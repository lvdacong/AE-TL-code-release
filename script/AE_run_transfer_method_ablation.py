"""Controlled transfer-method ablation for the final AE-TL evaluation.

The experiment compares reconstruction-only fine-tuning, two corrected freeze
strategies, a source-replay control, and three healthy-manifold alignment
methods (MMD, CORAL, and GRL-based adversarial alignment).  Damage samples are
never used for training, checkpoint selection, or hyperparameter selection.

Example
-------
    python AE_run_transfer_method_ablation.py --epochs 1000 --seeds 42 123 2026
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.lines import Line2D
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from AE_model_train_and_detect_auxiliary import (
    compute_anomaly_scores,
    compute_detection_metrics,
)
from AE_train_model_auxiliary import Autoencoder, apply_freeze_strategy


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "AD_preprocess_datasets_output"
MODEL_DIR = SCRIPT_DIR / "AE_model_train_and_detect_output"
SOURCE_CHECKPOINT = MODEL_DIR / "Damage_Repaired" / "pretrain" / "autoencoder.pth"
SOURCE_HEALTH = DATA_DIR / "health_original_2000" / "preprocessed_data_raw.npz"
SOURCE_TRAIN_SAMPLES = 1800
DEFAULT_OUTPUT = MODEL_DIR / "Transfer_Method_Ablation"


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    target_health: str
    target_damage: str
    l2_sp_alpha: float


SCENARIOS = (
    Scenario(
        "repair",
        "Local Structural Stiffening",
        "damage_repaired_12_original_500",
        "second_damage_12_original_100",
        0.005,
    ),
    Scenario(
        "relocation",
        "Sensor Relocation",
        "health_offset_count_1_2000",
        "first_damage_offset_count_1_100",
        0.0,
    ),
    Scenario(
        "drift",
        "Temperature-conditioned Sensor Response",
        "health_temperature_response_span20_2000",
        "first_damage_temperature_response_span20_100",
        0.0,
    ),
)

METHODS = (
    "full",
    "bottom",
    "decoder",
    "replay",
    "mmd",
    "coral",
    "dann",
)

METHOD_LABELS = {
    "full": "Full FT",
    "bottom": "Bottom-frozen",
    "decoder": "Decoder-only",
    "replay": "Source replay",
    "mmd": "MMD",
    "coral": "CORAL",
    "dann": "GRL adversarial",
}

METHOD_COLORS = {
    "full": "#111111",
    "bottom": "#6E6E6E",
    "decoder": "#A6A6A6",
    "replay": "#8C6BB1",
    "mmd": "#2878B5",
    "coral": "#D98E04",
    "dann": "#C44E52",
}

METHOD_LINESTYLES = {
    "full": "-",
    "bottom": "--",
    "decoder": ":",
    "replay": "-.",
    "mmd": "-",
    "coral": "--",
    "dann": "-.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--rho", type=float, default=0.1,
                        help="Initial alignment/reconstruction loss ratio.")
    parser.add_argument("--train-samples", type=int, default=400)
    # The repair dataset has 500 healthy samples.  A 400/50/50 split keeps
    # training, validation, and final healthy-control samples non-overlapping
    # in every scenario.
    parser.add_argument("--val-samples", type=int, default=50)
    parser.add_argument("--control-samples", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scenario", choices=[s.key for s in SCENARIOS] + ["all"], default="all")
    parser.add_argument(
        "--plot-rho-sensitivity-only",
        action="store_true",
        help="Combine the completed rho=0.01, 0.1, and 1.0 runs without retraining.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_v(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        return data["V"].astype(np.float32)


def scenario_arrays(
    scenario: Scenario,
    train_samples: int,
    val_samples: int,
    control_samples: int,
) -> Dict[str, np.ndarray]:
    target_all = load_v(DATA_DIR / scenario.target_health / "preprocessed_data_raw.npz")
    damage = load_v(DATA_DIR / scenario.target_damage / "preprocessed_data_raw.npz")
    source_all = load_v(SOURCE_HEALTH)

    minimum = train_samples + control_samples + val_samples
    if target_all.shape[0] < minimum:
        raise ValueError(
            f"{scenario.label} has {target_all.shape[0]} healthy samples; "
            f"at least {minimum} are required for non-overlapping splits."
        )
    if source_all.shape[0] < SOURCE_TRAIN_SAMPLES:
        raise ValueError(
            f"Source healthy data have {source_all.shape[0]} samples; "
            f"the declared pre-training split requires {SOURCE_TRAIN_SAMPLES}."
        )

    return {
        "target_train": target_all[:train_samples],
        "target_control": target_all[train_samples:train_samples + control_samples],
        "target_val": target_all[-val_samples:],
        "target_damage": damage,
        # Reuse only the 1,800 samples that formed the declared source-domain
        # pre-training split; do not pull source-validation samples into the
        # replay/alignment methods.
        "source_train": source_all[:SOURCE_TRAIN_SAMPLES],
    }


def build_autoencoder(input_dim: int, device: torch.device) -> Autoencoder:
    model = Autoencoder(
        input_dim=input_dim,
        encoder_dims=[768, 384, 192],
        latent_dim=192,
        decoder_dims=[192, 384, 768],
        dropout=0.0,
        activation="relu",
    ).to(device)
    checkpoint = torch.load(SOURCE_CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint)
    return model


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, coefficient: float) -> torch.Tensor:
        ctx.coefficient = coefficient
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.coefficient * grad_output, None


class DomainDiscriminator(nn.Module):
    def __init__(self, latent_dim: int = 192) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, z: torch.Tensor, grl_coefficient: float) -> torch.Tensor:
        z_reversed = GradientReverse.apply(z, grl_coefficient)
        return self.network(z_reversed).squeeze(1)


def covariance_matrix(z: torch.Tensor) -> torch.Tensor:
    centered = z - z.mean(dim=0, keepdim=True)
    denominator = max(1, z.shape[0] - 1)
    return centered.T @ centered / denominator


def coral_loss(source_z: torch.Tensor, target_z: torch.Tensor) -> torch.Tensor:
    dimension = source_z.shape[1]
    difference = covariance_matrix(source_z) - covariance_matrix(target_z)
    return difference.pow(2).sum() / (4.0 * dimension * dimension)


def mmd_loss(
    source_z: torch.Tensor,
    target_z: torch.Tensor,
    base_bandwidth: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    combined = torch.cat([source_z, target_z], dim=0)
    squared_distances = torch.cdist(combined, combined, p=2).pow(2)
    if base_bandwidth is None:
        with torch.no_grad():
            positive = squared_distances[squared_distances > 0]
            if positive.numel() == 0:
                base_bandwidth = torch.tensor(1.0, device=combined.device)
            else:
                base_bandwidth = positive.median().clamp_min(1e-12)

    kernel = torch.zeros_like(squared_distances)
    for multiplier in (0.25, 0.5, 1.0, 2.0, 4.0):
        bandwidth = (base_bandwidth * multiplier).clamp_min(1e-12)
        kernel = kernel + torch.exp(-squared_distances / (2.0 * bandwidth))

    n_source = source_z.shape[0]
    k_ss = kernel[:n_source, :n_source]
    k_tt = kernel[n_source:, n_source:]
    k_st = kernel[:n_source, n_source:]
    value = k_ss.mean() + k_tt.mean() - 2.0 * k_st.mean()
    return value.clamp_min(0.0), base_bandwidth.detach()


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def configure_method(
    model: Autoencoder,
    method: str,
    device: torch.device,
) -> Optional[DomainDiscriminator]:
    if method == "bottom":
        apply_freeze_strategy(model, "bottom", verbose=False)
    elif method == "decoder":
        apply_freeze_strategy(model, "encoder", verbose=False)
    else:
        apply_freeze_strategy(model, "none", verbose=False)

    if method == "dann":
        return DomainDiscriminator().to(device)
    return None


def build_optimizer(
    model: Autoencoder,
    discriminator: Optional[DomainDiscriminator],
) -> Adam:
    groups: List[dict] = []
    encoder_parameters = [p for p in model.encoder.parameters() if p.requires_grad]
    decoder_parameters = [p for p in model.decoder.parameters() if p.requires_grad]
    if encoder_parameters:
        groups.append({"params": encoder_parameters, "lr": 1e-5})
    if decoder_parameters:
        groups.append({"params": decoder_parameters, "lr": 1e-3})
    if discriminator is not None:
        groups.append({"params": list(discriminator.parameters()), "lr": 1e-3})
    return Adam(groups, weight_decay=0.0)


def clone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def evaluate_detection(
    model: Autoencoder,
    damage: np.ndarray,
    control: np.ndarray,
    device: torch.device,
) -> Dict[str, float]:
    damage_scores = compute_anomaly_scores(damage, model, device, method="max")
    control_scores = compute_anomaly_scores(control, model, device, method="max")
    metrics = compute_detection_metrics(
        damage_scores,
        control_scores,
        n_bootstrap=0,
    )
    return {
        "auc": float(metrics["auc"]),
        "tpr_at_fpr": float(metrics["tpr_at_fpr"]),
        "best_f1": float(metrics["best_f1"]),
    }


def train_one(
    scenario: Scenario,
    arrays: Dict[str, np.ndarray],
    method: str,
    seed: int,
    epochs: int,
    batch_size: int,
    rho: float,
    device: torch.device,
) -> Tuple[List[dict], dict]:
    set_seed(seed)
    x_target = torch.from_numpy(arrays["target_train"]).to(device)
    x_val = torch.from_numpy(arrays["target_val"]).to(device)
    x_source = torch.from_numpy(arrays["source_train"]).to(device)

    model = build_autoencoder(x_target.shape[1], device)
    discriminator = configure_method(model, method, device)
    optimizer = build_optimizer(model, discriminator)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    reconstruction = nn.MSELoss()
    domain_criterion = nn.BCEWithLogitsLoss()
    reference_state = clone_state_dict(model)

    uses_source = method in {"replay", "mmd", "coral", "dann"}
    best_val = float("inf")
    best_epoch = 0
    best_state = clone_state_dict(model)
    alignment_scale: Optional[torch.Tensor] = None
    effective_alignment_weight = 0.0
    mmd_bandwidth: Optional[torch.Tensor] = None
    rows: List[dict] = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        if discriminator is not None:
            discriminator.train()
        permutation = torch.randperm(x_target.shape[0], device=device)
        target_reconstruction_sum = 0.0
        target_count = 0
        raw_alignment_sum = 0.0
        raw_alignment_batches = 0
        domain_correct = 0
        domain_count = 0

        for start in range(0, x_target.shape[0], batch_size):
            indices = permutation[start:start + batch_size]
            target_clean = x_target[indices]
            target_input = target_clean + torch.randn_like(target_clean) * 0.005
            target_z = model.encoder(target_input)
            target_reconstruction = model.decoder(target_z)
            target_loss = reconstruction(target_reconstruction, target_clean)
            optimisation_loss = target_loss
            raw_alignment = None

            if uses_source:
                source_indices = torch.randint(
                    low=0,
                    high=x_source.shape[0],
                    size=(target_clean.shape[0],),
                    device=device,
                )
                source_clean = x_source[source_indices]
                source_input = source_clean + torch.randn_like(source_clean) * 0.005
                source_z = model.encoder(source_input)
                source_reconstruction = model.decoder(source_z)
                source_loss = reconstruction(source_reconstruction, source_clean)
                reconstruction_loss = 0.5 * (target_loss + source_loss)
                optimisation_loss = reconstruction_loss

                alignment_count = min(128, source_z.shape[0], target_z.shape[0])
                source_align = source_z[:alignment_count]
                target_align = target_z[:alignment_count]

                if method == "mmd":
                    raw_alignment, mmd_bandwidth = mmd_loss(
                        source_align,
                        target_align,
                        mmd_bandwidth,
                    )
                elif method == "coral":
                    raw_alignment = coral_loss(source_align, target_align)
                elif method == "dann":
                    progress = ((epoch - 1) * math.ceil(x_target.shape[0] / batch_size) +
                                start // batch_size)
                    total_steps = max(1, epochs * math.ceil(x_target.shape[0] / batch_size) - 1)
                    p = progress / total_steps
                    grl_coefficient = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
                    domain_features = torch.cat([source_align, target_align], dim=0)
                    domain_labels = torch.cat([
                        torch.zeros(alignment_count, device=device),
                        torch.ones(alignment_count, device=device),
                    ])
                    domain_logits = discriminator(domain_features, grl_coefficient)
                    raw_alignment = domain_criterion(domain_logits, domain_labels)
                    domain_predictions = (domain_logits.detach() >= 0).to(domain_labels.dtype)
                    domain_correct += int((domain_predictions == domain_labels).sum().item())
                    domain_count += int(domain_labels.numel())

                if raw_alignment is not None:
                    raw_alignment_sum += float(raw_alignment.detach().item())
                    raw_alignment_batches += 1
                    if alignment_scale is None:
                        alignment_scale = (
                            reconstruction_loss.detach() /
                            raw_alignment.detach().clamp_min(1e-12)
                        )
                        effective_alignment_weight = float((rho * alignment_scale).item())
                    optimisation_loss = optimisation_loss + rho * alignment_scale * raw_alignment

            if scenario.l2_sp_alpha > 0:
                l2_sp = sum(
                    (parameter - reference_state[name]).pow(2).sum()
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                )
                optimisation_loss = optimisation_loss + scenario.l2_sp_alpha * l2_sp

            optimizer.zero_grad(set_to_none=True)
            optimisation_loss.backward()
            parameters: Iterable[torch.Tensor] = model.parameters()
            if discriminator is not None:
                parameters = list(model.parameters()) + list(discriminator.parameters())
            nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()

            target_reconstruction_sum += target_loss.item() * target_clean.shape[0]
            target_count += target_clean.shape[0]

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_prediction = model(x_val)
            val_loss = reconstruction(val_prediction, x_val).item()
        target_train_loss = target_reconstruction_sum / max(1, target_count)

        # Match the existing best-validation checkpoint rule while avoiding
        # redundant state copies for numerically negligible improvements.
        improvement_threshold = (
            max(1e-9, best_val * 1e-5)
            if math.isfinite(best_val) else 0.0
        )
        if (not math.isfinite(best_val) or
                val_loss < best_val - improvement_threshold or
                epoch == epochs):
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_state = clone_state_dict(model)

        rows.append({
            "scenario": scenario.key,
            "scenario_label": scenario.label,
            "method": method,
            "method_label": METHOD_LABELS[method],
            "seed": seed,
            "epoch": epoch,
            "target_train_reconstruction_mse": target_train_loss,
            "target_val_reconstruction_mse": val_loss,
            "raw_alignment_objective": (
                raw_alignment_sum / raw_alignment_batches
                if raw_alignment_batches else np.nan
            ),
            "domain_discriminator_accuracy": (
                domain_correct / domain_count if domain_count else np.nan
            ),
        })

    model.load_state_dict(best_state)
    detection = evaluate_detection(
        model,
        arrays["target_damage"],
        arrays["target_control"],
        device,
    )
    elapsed = time.perf_counter() - started
    alignment_history = [
        row["raw_alignment_objective"] for row in rows
        if math.isfinite(row["raw_alignment_objective"])
    ]
    domain_accuracy_history = [
        row["domain_discriminator_accuracy"] for row in rows
        if math.isfinite(row["domain_discriminator_accuracy"])
    ]
    summary = {
        "scenario": scenario.key,
        "scenario_label": scenario.label,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_reconstruction_mse": best_val,
        "last_val_reconstruction_mse": rows[-1]["target_val_reconstruction_mse"],
        "auc": detection["auc"],
        "tpr_at_fpr": detection["tpr_at_fpr"],
        "best_f1": detection["best_f1"],
        "ae_trainable_parameters": trainable_parameter_count(model),
        "domain_discriminator_parameters": (
            sum(p.numel() for p in discriminator.parameters())
            if discriminator is not None else 0
        ),
        "uses_source_replay": uses_source,
        "rho": rho if method in {"mmd", "coral", "dann"} else 0.0,
        "effective_alignment_weight": effective_alignment_weight,
        "initial_raw_alignment_objective": (
            alignment_history[0] if alignment_history else np.nan
        ),
        "final_raw_alignment_objective": (
            alignment_history[-1] if alignment_history else np.nan
        ),
        "initial_domain_discriminator_accuracy": (
            domain_accuracy_history[0] if domain_accuracy_history else np.nan
        ),
        "final_domain_discriminator_accuracy": (
            domain_accuracy_history[-1] if domain_accuracy_history else np.nan
        ),
        "elapsed_seconds": elapsed,
    }
    return rows, summary


def method_order(methods: Iterable[str]) -> List[str]:
    selected = set(methods)
    return [method for method in METHODS if method in selected]


def load_preserved_rows(
    csv_path: Path,
    selected_scenarios: Iterable[Scenario],
    selected_methods: Iterable[str],
) -> List[dict]:
    """Keep completed rows outside the requested scenario-method cells."""
    if not csv_path.exists():
        return []
    frame = pd.read_csv(csv_path)
    scenario_keys = {scenario.key for scenario in selected_scenarios}
    method_keys = set(selected_methods)
    replace = frame["scenario"].isin(scenario_keys) & frame["method"].isin(method_keys)
    return frame.loc[~replace].to_dict("records")


def has_complete_plot_grid(summary: pd.DataFrame, methods: Iterable[str]) -> bool:
    """Return whether every scenario-method cell has at least one run."""
    for scenario in SCENARIOS:
        for method in methods:
            if not (
                (summary["scenario"] == scenario.key) &
                (summary["method"] == method)
            ).any():
                return False
    return True


def plot_training_curves(curves: pd.DataFrame, output_dir: Path, methods: List[str]) -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 8,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), sharex=True, constrained_layout=True)
    for axis, scenario in zip(axes, SCENARIOS):
        scenario_frame = curves[curves["scenario"] == scenario.key]
        for method in methods:
            method_frame = scenario_frame[scenario_frame["method"] == method]
            grouped = method_frame.groupby("epoch")["target_val_reconstruction_mse"]
            mean = grouped.mean()
            std = grouped.std(ddof=1).fillna(0.0)
            # A short rolling mean improves readability without changing the
            # saved raw epoch-level CSV used for numerical reporting.
            mean_plot = mean.rolling(window=11, min_periods=1, center=True).mean()
            std_plot = std.rolling(window=11, min_periods=1, center=True).mean()
            x = mean_plot.index.to_numpy()
            y = mean_plot.to_numpy()
            s = std_plot.to_numpy()
            axis.plot(
                x,
                y,
                color=METHOD_COLORS[method],
                linestyle=METHOD_LINESTYLES[method],
                linewidth=1.05,
            )
            if method_frame["seed"].nunique() > 1:
                axis.fill_between(
                    x,
                    np.maximum(y - s, 1e-12),
                    y + s,
                    color=METHOD_COLORS[method],
                    alpha=0.10,
                    linewidth=0,
                )
        axis.set_yscale("log")
        # A mean-minus-standard-deviation band can become negative during the
        # early transient.  Keep the log-axis limits tied to the actually
        # observed positive validation losses instead of the numerical floor
        # used only to make fill_between log-safe.
        observed = scenario_frame["target_val_reconstruction_mse"].to_numpy()
        observed = observed[np.isfinite(observed) & (observed > 0)]
        if observed.size:
            axis.set_ylim(observed.min() * 0.75, observed.max() * 1.25)
        axis.set_title(scenario.label, fontsize=8.5)
        axis.set_xlabel("Fine-tuning epoch")
        axis.grid(False)
    axes[0].set_ylabel("Target healthy validation MSE")
    legend_handles = [
        Line2D(
            [0], [0],
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=1.2,
            label=METHOD_LABELS[method],
        )
        for method in methods
    ]
    fig.legend(
        handles=legend_handles,
        loc="outside upper center",
        ncol=4,
        frameon=False,
        fontsize=7.2,
        handlelength=2.4,
    )
    fig.savefig(output_dir / "fig_transfer_method_ablation_curves.png", dpi=300)
    fig.savefig(output_dir / "fig_transfer_method_ablation_curves.pdf")
    plt.close(fig)


def plot_metrics(summary: pd.DataFrame, output_dir: Path, methods: List[str]) -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 8,
        "axes.linewidth": 0.7,
    })
    scenario_names = ["Local\nstiffening", "Relocation", "Temperature\nresponse"]
    method_names = [METHOD_LABELS[method] for method in methods]
    auc_means = np.empty((len(methods), len(SCENARIOS)))
    auc_stds = np.empty_like(auc_means)
    val_means = np.empty_like(auc_means)
    val_stds = np.empty_like(auc_means)

    for column, scenario in enumerate(SCENARIOS):
        frame = summary[summary["scenario"] == scenario.key]
        for row, method in enumerate(methods):
            method_frame = frame[frame["method"] == method]
            auc_means[row, column] = method_frame["auc"].mean()
            auc_stds[row, column] = method_frame["auc"].std(ddof=1)
            val_means[row, column] = (
                method_frame["best_val_reconstruction_mse"].mean() * 1e3
            )
            val_stds[row, column] = (
                method_frame["best_val_reconstruction_mse"].std(ddof=1) * 1e3
            )

    fig, axes = plt.subplots(
        1, 2,
        figsize=(7.2, 3.25),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.0]},
    )
    matrices = [val_means, auc_means]
    deviations = [val_stds, auc_stds]
    titles = [
        r"Best target validation MSE ($\times 10^{-3}$)",
        "Max-channel AUC",
    ]
    colormaps = ["YlGn_r", "YlGn"]

    for panel, (axis, matrix, deviation, title, cmap) in enumerate(
        zip(axes, matrices, deviations, titles, colormaps)
    ):
        finite = matrix[np.isfinite(matrix)]
        margin = max((finite.max() - finite.min()) * 0.06, 1e-9)
        image = axis.imshow(
            matrix,
            aspect="auto",
            cmap=cmap,
            vmin=finite.min() - margin,
            vmax=finite.max() + margin,
        )
        axis.set_title(title, fontsize=9)
        axis.set_xticks(np.arange(len(SCENARIOS)), labels=scenario_names)
        axis.set_yticks(np.arange(len(methods)))
        if panel == 0:
            axis.set_yticklabels(method_names)
        else:
            axis.set_yticklabels([])
            axis.tick_params(axis="y", length=0)
        axis.tick_params(axis="both", length=0)
        axis.set_xticks(np.arange(-0.5, len(SCENARIOS), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(methods), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.2)
        axis.tick_params(which="minor", bottom=False, left=False)
        for spine in axis.spines.values():
            spine.set_visible(False)

        best_values = (
            np.min(matrix, axis=0) if panel == 0
            else np.max(matrix, axis=0)
        )
        normalizer = image.norm
        for row in range(len(methods)):
            for column in range(len(SCENARIOS)):
                rgba = image.cmap(normalizer(matrix[row, column]))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                color = "white" if luminance < 0.48 else "black"
                if panel == 0:
                    label = f"{matrix[row, column]:.4f}\n$\\pm${deviation[row, column]:.4f}"
                else:
                    label = f"{matrix[row, column]:.4f}\n$\\pm${deviation[row, column]:.4f}"
                axis.text(
                    column,
                    row,
                    label,
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=6.6,
                    fontweight=(
                        "bold"
                        if np.isclose(
                            matrix[row, column],
                            best_values[column],
                            rtol=0.0,
                            atol=(5e-5 if panel == 1 else 5e-4),
                        )
                        else "normal"
                    ),
                )
        colorbar = fig.colorbar(image, ax=axis, fraction=0.045, pad=0.025)
        colorbar.ax.tick_params(labelsize=6.5, width=0.6, length=2)

    fig.savefig(output_dir / "fig_transfer_method_ablation_metrics.png", dpi=300)
    fig.savefig(output_dir / "fig_transfer_method_ablation_metrics.pdf")
    plt.close(fig)


def plot_rho_sensitivity(output_dir: Path) -> None:
    run_paths = {
        0.01: MODEL_DIR / "Transfer_Method_Ablation_Rho_0p01" / "run_summary.csv",
        0.1: MODEL_DIR / "Transfer_Method_Ablation" / "run_summary.csv",
        1.0: MODEL_DIR / "Transfer_Method_Ablation_Rho_1p0" / "run_summary.csv",
    }
    frames: List[pd.DataFrame] = []
    alignment_methods = ["mmd", "coral", "dann"]
    for rho, path in run_paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame = frame[frame["method"].isin(alignment_methods)].copy()
        frame["rho"] = rho
        frames.append(frame)

    runs = pd.concat(frames, ignore_index=True)
    aggregate = (
        runs.groupby(
            ["scenario", "scenario_label", "method", "method_label", "rho"],
            as_index=False,
        )
        .agg(
            n_seeds=("seed", "nunique"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", lambda values: values.std(ddof=1)),
            best_val_reconstruction_mse_mean=(
                "best_val_reconstruction_mse", "mean"
            ),
            best_val_reconstruction_mse_std=(
                "best_val_reconstruction_mse",
                lambda values: values.std(ddof=1),
            ),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(output_dir / "rho_sensitivity_summary.csv", index=False)

    main_summary = pd.read_csv(
        MODEL_DIR / "Transfer_Method_Ablation" / "run_summary.csv"
    )
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 8,
        "axes.linewidth": 0.7,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), constrained_layout=True)
    rho_values = [0.01, 0.1, 1.0]
    rho_labels = ["0.01", "0.1", "1.0"]
    method_names = [METHOD_LABELS[method] for method in alignment_methods]

    for panel, (axis, scenario) in enumerate(zip(axes, SCENARIOS)):
        matrix = np.empty((len(alignment_methods), len(rho_values)))
        deviation = np.empty_like(matrix)
        for row, method in enumerate(alignment_methods):
            for column, rho in enumerate(rho_values):
                selected = aggregate[
                    (aggregate["scenario"] == scenario.key) &
                    (aggregate["method"] == method) &
                    np.isclose(aggregate["rho"], rho)
                ].iloc[0]
                matrix[row, column] = selected["auc_mean"]
                deviation[row, column] = selected["auc_std"]

        references = main_summary[main_summary["scenario"] == scenario.key]
        full_auc = references[references["method"] == "full"]["auc"].mean()
        replay_auc = references[references["method"] == "replay"]["auc"].mean()
        finite = matrix[np.isfinite(matrix)]
        # Include the two title references in the colour range so that cell
        # colours directly show where the alignment variants sit relative to
        # Full FT and source replay, even though the references are not
        # repeated as heatmap cells.
        combined = np.concatenate([finite, np.array([full_auc, replay_auc])])
        margin = max((combined.max() - combined.min()) * 0.06, 1e-9)
        heatmap = axis.imshow(
            matrix,
            aspect="auto",
            cmap="YlGn",
            vmin=combined.min() - margin,
            vmax=combined.max() + margin,
        )
        axis.set_title(
            f"{scenario.label}\nFull FT {full_auc:.4f} | Replay {replay_auc:.4f}",
            fontsize=8.2,
        )
        axis.set_xticks(np.arange(len(rho_values)), labels=rho_labels)
        axis.set_xlabel(r"Alignment strength $\rho$")
        axis.set_yticks(np.arange(len(alignment_methods)))
        if panel == 0:
            axis.set_yticklabels(method_names)
        else:
            axis.set_yticklabels([])
            axis.tick_params(axis="y", length=0)
        axis.tick_params(axis="both", length=0)
        axis.set_xticks(np.arange(-0.5, len(rho_values), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(alignment_methods), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.2)
        axis.tick_params(which="minor", bottom=False, left=False)
        for spine in axis.spines.values():
            spine.set_visible(False)

        normalizer = heatmap.norm
        best_auc = matrix.max()
        for row in range(len(alignment_methods)):
            for column in range(len(rho_values)):
                rgba = heatmap.cmap(normalizer(matrix[row, column]))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.4f}\n$\\pm${deviation[row, column]:.4f}",
                    ha="center",
                    va="center",
                    color=("white" if luminance < 0.48 else "black"),
                    fontsize=6.4,
                    fontweight=(
                        "bold"
                        if np.isclose(
                            matrix[row, column],
                            best_auc,
                            rtol=0.0,
                            atol=5e-5,
                        )
                        else "normal"
                    ),
                )
        colorbar = fig.colorbar(heatmap, ax=axis, fraction=0.05, pad=0.025)
        colorbar.ax.tick_params(labelsize=6.2, width=0.6, length=2)

    fig.savefig(output_dir / "fig_transfer_alignment_rho_sensitivity.png", dpi=300)
    fig.savefig(output_dir / "fig_transfer_alignment_rho_sensitivity.pdf")
    plt.close(fig)


def aggregate_summary(summary: pd.DataFrame, methods: List[str]) -> pd.DataFrame:
    metrics = [
        "best_val_reconstruction_mse",
        "auc",
        "tpr_at_fpr",
        "best_f1",
        "best_epoch",
        "elapsed_seconds",
    ]
    rows: List[dict] = []
    for scenario in SCENARIOS:
        for method in methods:
            frame = summary[
                (summary["scenario"] == scenario.key) &
                (summary["method"] == method)
            ]
            row = {
                "scenario": scenario.key,
                "scenario_label": scenario.label,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "n_seeds": frame["seed"].nunique(),
            }
            for metric in metrics:
                row[f"{metric}_mean"] = frame[metric].mean()
                row[f"{metric}_std"] = (
                    frame[metric].std(ddof=1) if len(frame) > 1 else 0.0
                )
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if not SOURCE_CHECKPOINT.exists():
        raise FileNotFoundError(SOURCE_CHECKPOINT)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_rho_sensitivity_only:
        plot_rho_sensitivity(args.output_dir)
        print(f"[done] {args.output_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    if device.type == "cuda":
        print(f"[gpu] {torch.cuda.get_device_name(0)}")

    selected_scenarios = [
        scenario for scenario in SCENARIOS
        if args.scenario == "all" or scenario.key == args.scenario
    ]
    methods = method_order(args.methods)
    all_curve_rows = load_preserved_rows(
        args.output_dir / "epoch_curves.csv", selected_scenarios, methods
    )
    all_summary_rows = load_preserved_rows(
        args.output_dir / "run_summary.csv", selected_scenarios, methods
    )

    config = {
        "epochs": args.epochs,
        "seeds": args.seeds,
        "methods": methods,
        "rho": args.rho,
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "control_samples": args.control_samples,
        "batch_size": args.batch_size,
        "encoder_lr": 1e-5,
        "decoder_lr": 1e-3,
        "noise_std": 0.005,
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_health": str(SOURCE_HEALTH),
        "source_train_samples": SOURCE_TRAIN_SAMPLES,
        "scenarios": [asdict(scenario) for scenario in SCENARIOS],
        "updated_scenarios": [scenario.key for scenario in selected_scenarios],
        "incremental_merge": bool(all_curve_rows or all_summary_rows),
        "data_split_policy": (
            "identical deterministic positional split for every method: first "
            "train_samples rows for training, the immediately following "
            "control_samples rows for healthy testing, and the last val_samples "
            "rows for validation"
        ),
        "temperature_response": {
            "temperature_span_c": 20.0,
            "healthy_dataset": "health_temperature_response_span20_2000",
            "damage_dataset": "first_damage_temperature_response_span20_100",
            "condition_reuse": (
                "healthy and damaged data use the same fixed 100-condition "
                "sequence, repeated by sample_index modulo 100"
            ),
        },
    }
    (args.output_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    for scenario in selected_scenarios:
        arrays = scenario_arrays(
            scenario,
            args.train_samples,
            args.val_samples,
            args.control_samples,
        )
        for method in methods:
            for seed in args.seeds:
                print(
                    f"\n[{scenario.label}] {METHOD_LABELS[method]} | "
                    f"seed={seed} | epochs={args.epochs}"
                )
                curve_rows, summary = train_one(
                    scenario=scenario,
                    arrays=arrays,
                    method=method,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    rho=args.rho,
                    device=device,
                )
                all_curve_rows.extend(curve_rows)
                all_summary_rows.append(summary)
                print(
                    f"  val={summary['best_val_reconstruction_mse']:.6g} | "
                    f"AUC={summary['auc']:.4f} | "
                    f"time={summary['elapsed_seconds']:.1f}s"
                )

                # Persist after every run so an interruption does not discard
                # completed evidence.
                pd.DataFrame(all_curve_rows).to_csv(
                    args.output_dir / "epoch_curves.csv", index=False
                )
                pd.DataFrame(all_summary_rows).to_csv(
                    args.output_dir / "run_summary.csv", index=False
                )

    curves = pd.DataFrame(all_curve_rows)
    summary = pd.DataFrame(all_summary_rows)
    plotted_methods = method_order(summary["method"].dropna().unique())
    aggregate = aggregate_summary(summary, plotted_methods)
    aggregate.to_csv(args.output_dir / "aggregate_summary.csv", index=False)

    if has_complete_plot_grid(summary, plotted_methods):
        plot_training_curves(curves, args.output_dir, plotted_methods)
        plot_metrics(summary, args.output_dir, plotted_methods)

    print("\n[aggregate]")
    print(aggregate.to_string(index=False))
    print(f"\n[done] {args.output_dir}")


if __name__ == "__main__":
    main()
