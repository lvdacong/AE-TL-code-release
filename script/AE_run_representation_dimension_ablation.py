"""Controlled PCA/AE representation-dimension ablation.

This experiment varies only the PCA component count or the AE bottleneck
dimension.  The source/target data, surrounding AE hidden widths, optimiser,
learning-rate schedule, epoch budget, and detection score remain fixed.
Damage samples are used only for final evaluation.

The 192-dimensional AE uses the fixed reference source checkpoint.  Each
other AE architecture is source-pretrained once with seed 42; three target
fine-tuning seeds quantify target-adaptation variability from that fixed
checkpoint.

Example
-------
    python AE_run_representation_dimension_ablation.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from AE_model_train_and_detect_auxiliary import (
    compute_anomaly_scores,
    compute_detection_metrics,
)
from AE_train_model_auxiliary import Autoencoder


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "AD_preprocess_datasets_output"
MODEL_DIR = SCRIPT_DIR / "AE_model_train_and_detect_output"
OUTPUT_DIR = MODEL_DIR / "Representation_Dimension_Ablation"
SOURCE_REFERENCE_CHECKPOINT = (
    MODEL_DIR / "Damage_Repaired" / "pretrain" / "autoencoder.pth"
)
SOURCE_HEALTH = DATA_DIR / "health_original_2000" / "preprocessed_data_raw.npz"

DEFAULT_DIMS = (32, 64, 96, 128, 160, 192)
DEFAULT_SEEDS = (42, 123, 2026)
SOURCE_SEED = 42
SOURCE_TRAIN_SAMPLES = 1800
SOURCE_VAL_SAMPLES = 200
TARGET_TRAIN_SAMPLES = 400
TARGET_VAL_SAMPLES = 50
TARGET_CONTROL_SAMPLES = 50
BATCH_SIZE = 256
PRETRAIN_EPOCHS = 2000
FINETUNE_EPOCHS = 1000


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
        "Temperature-conditioned Response",
        "health_temperature_response_span20_2000",
        "first_damage_temperature_response_span20_100",
        0.0,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=int, nargs="+", default=list(DEFAULT_DIMS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--pretrain-epochs", type=int, default=PRETRAIN_EPOCHS)
    parser.add_argument("--finetune-epochs", type=int, default=FINETUNE_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_v(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        return data["V"].astype(np.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_arrays() -> Dict[str, np.ndarray]:
    values = load_v(SOURCE_HEALTH)
    required = SOURCE_TRAIN_SAMPLES + SOURCE_VAL_SAMPLES
    if values.shape[0] < required:
        raise ValueError(f"Source data have {values.shape[0]} rows; {required} required")
    return {
        "train": values[:SOURCE_TRAIN_SAMPLES],
        "val": values[-SOURCE_VAL_SAMPLES:],
    }


def scenario_arrays(scenario: Scenario) -> Dict[str, np.ndarray]:
    health_path = DATA_DIR / scenario.target_health / "preprocessed_data_raw.npz"
    damage_path = DATA_DIR / scenario.target_damage / "preprocessed_data_raw.npz"
    health = load_v(health_path)
    damage = load_v(damage_path)
    required = TARGET_TRAIN_SAMPLES + TARGET_CONTROL_SAMPLES + TARGET_VAL_SAMPLES
    if health.shape[0] < required:
        raise ValueError(
            f"{scenario.label} has {health.shape[0]} healthy rows; {required} required"
        )
    return {
        "train": health[:TARGET_TRAIN_SAMPLES],
        "control": health[
            TARGET_TRAIN_SAMPLES:TARGET_TRAIN_SAMPLES + TARGET_CONTROL_SAMPLES
        ],
        "val": health[-TARGET_VAL_SAMPLES:],
        "damage": damage,
    }


def ae_config(latent_dim: int) -> dict:
    return {
        "input_dim": 252,
        "encoder_dims": [768, 384, 192],
        "latent_dim": latent_dim,
        "decoder_dims": [192, 384, 768],
        "dropout": 0.0,
        "activation": "relu",
    }


def build_model(latent_dim: int, device: torch.device) -> Autoencoder:
    config = ae_config(latent_dim)
    return Autoencoder(
        input_dim=config["input_dim"],
        encoder_dims=config["encoder_dims"],
        latent_dim=config["latent_dim"],
        decoder_dims=config["decoder_dims"],
        dropout=config["dropout"],
        activation=config["activation"],
    ).to(device)


def clone_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def reconstruction_mse(model: nn.Module, values: np.ndarray, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(values).to(device)
        return float(nn.functional.mse_loss(model(tensor), tensor).item())


def fit_source_model(
    latent_dim: int,
    arrays: Dict[str, np.ndarray],
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[Autoencoder, List[dict], dict]:
    set_seed(SOURCE_SEED)
    model = build_model(latent_dim, device)
    train = torch.from_numpy(arrays["train"]).to(device)
    val = torch.from_numpy(arrays["val"]).to(device)
    optimizer = Adam(model.parameters(), lr=3e-4, weight_decay=0.0)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_epoch = 0
    best_state = clone_state(model)
    rows: List[dict] = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(train.shape[0], device=device)
        running = 0.0
        count = 0
        for start in range(0, train.shape[0], batch_size):
            clean = train[permutation[start:start + batch_size]]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(clean), clean)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.item()) * clean.shape[0]
            count += clean.shape[0]
        scheduler.step()
        train_loss = running / max(1, count)
        val_loss = reconstruction_mse(model, arrays["val"], device)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = clone_state(model)
        rows.append({
            "stage": "source_pretrain",
            "scenario": "source",
            "latent_dim": latent_dim,
            "seed": SOURCE_SEED,
            "epoch": epoch,
            "train_reconstruction_mse": train_loss,
            "val_reconstruction_mse": val_loss,
        })

    model.load_state_dict(best_state)
    return model, rows, {
        "latent_dim": latent_dim,
        "source_seed": SOURCE_SEED,
        "best_epoch": best_epoch,
        "best_source_val_reconstruction_mse": best_val,
        "parameter_count": parameter_count(model),
        "elapsed_seconds": time.perf_counter() - started,
    }


def fit_target_model(
    source_state: Dict[str, torch.Tensor],
    latent_dim: int,
    scenario: Scenario,
    arrays: Dict[str, np.ndarray],
    seed: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[List[dict], dict]:
    set_seed(seed)
    model = build_model(latent_dim, device)
    model.load_state_dict(source_state)
    reference_state = clone_state(model)
    train = torch.from_numpy(arrays["train"]).to(device)
    val = torch.from_numpy(arrays["val"]).to(device)
    optimizer = Adam([
        {"params": list(model.encoder.parameters()), "lr": 1e-5},
        {"params": list(model.decoder.parameters()), "lr": 1e-3},
    ], weight_decay=0.0)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_epoch = 0
    best_state = clone_state(model)
    rows: List[dict] = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(train.shape[0], device=device)
        running = 0.0
        count = 0
        for start in range(0, train.shape[0], batch_size):
            clean = train[permutation[start:start + batch_size]]
            noisy = clean + torch.randn_like(clean) * 0.005
            optimizer.zero_grad(set_to_none=True)
            reconstruction_loss = criterion(model(noisy), clean)
            optimisation_loss = reconstruction_loss
            if scenario.l2_sp_alpha > 0:
                l2_sp = sum(
                    (parameter - reference_state[name]).pow(2).sum()
                    for name, parameter in model.named_parameters()
                )
                optimisation_loss = optimisation_loss + scenario.l2_sp_alpha * l2_sp
            optimisation_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(reconstruction_loss.item()) * clean.shape[0]
            count += clean.shape[0]
        scheduler.step()
        train_loss = running / max(1, count)
        val_loss = reconstruction_mse(model, arrays["val"], device)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = clone_state(model)
        rows.append({
            "stage": "target_finetune",
            "scenario": scenario.key,
            "latent_dim": latent_dim,
            "seed": seed,
            "epoch": epoch,
            "train_reconstruction_mse": train_loss,
            "val_reconstruction_mse": val_loss,
        })

    model.load_state_dict(best_state)
    damage_scores = compute_anomaly_scores(arrays["damage"], model, device, method="max")
    control_scores = compute_anomaly_scores(arrays["control"], model, device, method="max")
    metrics = compute_detection_metrics(damage_scores, control_scores, n_bootstrap=0)
    return rows, {
        "method": "AE-TL",
        "scenario": scenario.key,
        "scenario_label": scenario.label,
        "representation_dim": latent_dim,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_reconstruction_mse": best_val,
        "auc": float(metrics["auc"]),
        "tpr_at_fpr": float(metrics["tpr_at_fpr"]),
        "best_f1": float(metrics["best_f1"]),
        "parameter_count": parameter_count(model),
        "elapsed_seconds": time.perf_counter() - started,
    }


def evaluate_pca(
    latent_dim: int,
    scenario: Scenario,
    arrays: Dict[str, np.ndarray],
) -> dict:
    started = time.perf_counter()
    model = PCA(n_components=latent_dim, svd_solver="full")
    model.fit(arrays["train"])

    def reconstruct(values: np.ndarray) -> np.ndarray:
        return model.inverse_transform(model.transform(values))

    val_reconstruction = reconstruct(arrays["val"])
    damage_reconstruction = reconstruct(arrays["damage"])
    control_reconstruction = reconstruct(arrays["control"])
    damage_scores = np.max(np.abs(arrays["damage"] - damage_reconstruction), axis=1)
    control_scores = np.max(np.abs(arrays["control"] - control_reconstruction), axis=1)
    metrics = compute_detection_metrics(damage_scores, control_scores, n_bootstrap=0)
    return {
        "method": "PCA",
        "scenario": scenario.key,
        "scenario_label": scenario.label,
        "representation_dim": latent_dim,
        "seed": np.nan,
        "best_epoch": np.nan,
        "best_val_reconstruction_mse": float(
            np.mean((arrays["val"] - val_reconstruction) ** 2)
        ),
        "auc": float(metrics["auc"]),
        "tpr_at_fpr": float(metrics["tpr_at_fpr"]),
        "best_f1": float(metrics["best_f1"]),
        "parameter_count": np.nan,
        "explained_variance_ratio": float(model.explained_variance_ratio_.sum()),
        "elapsed_seconds": time.perf_counter() - started,
    }


def load_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict("records")


def save_rows(path: Path, rows: Iterable[dict]) -> None:
    frame = pd.DataFrame(list(rows))
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def has_ae_run(rows: List[dict], scenario: str, dim: int, seed: int) -> bool:
    return any(
        row.get("method") == "AE-TL"
        and row.get("scenario") == scenario
        and int(row.get("representation_dim")) == dim
        and int(float(row.get("seed"))) == seed
        for row in rows
    )


def has_pca_run(rows: List[dict], scenario: str, dim: int) -> bool:
    return any(
        row.get("method") == "PCA"
        and row.get("scenario") == scenario
        and int(row.get("representation_dim")) == dim
        for row in rows
    )


def aggregate_results(rows: pd.DataFrame) -> pd.DataFrame:
    ae = rows[rows["method"] == "AE-TL"].copy()
    pca = rows[rows["method"] == "PCA"].copy()
    metrics = ["best_val_reconstruction_mse", "auc", "tpr_at_fpr", "best_f1"]
    ae_agg = ae.groupby(
        ["method", "scenario", "scenario_label", "representation_dim"], as_index=False
    )[metrics].agg(["mean", "std"])
    ae_agg.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        for column in ae_agg.columns.to_flat_index()
    ]
    pca_agg = pca[[
        "method", "scenario", "scenario_label", "representation_dim",
        *metrics, "explained_variance_ratio",
    ]].copy()
    for metric in metrics:
        pca_agg[f"{metric}_mean"] = pca_agg[metric]
        pca_agg[f"{metric}_std"] = 0.0
    pca_agg = pca_agg[[
        "method", "scenario", "scenario_label", "representation_dim",
        *[name for metric in metrics for name in (f"{metric}_mean", f"{metric}_std")],
        "explained_variance_ratio",
    ]]
    ae_agg["explained_variance_ratio"] = np.nan
    return pd.concat([ae_agg, pca_agg], ignore_index=True, sort=False)


def plot_comparison(aggregate: pd.DataFrame, output_dir: Path, dims: Tuple[int, ...]) -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.7,
        "axes.grid": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })
    colors = {"AE-TL": "#7BA7BC", "PCA": "#C97A6C"}
    markers = {"AE-TL": "o", "PCA": "s"}
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), constrained_layout=True)
    handles = []
    labels = []

    for column, scenario in enumerate(SCENARIOS):
        scenario_frame = aggregate[aggregate["scenario"] == scenario.key]
        for row_index, metric in enumerate(("auc", "best_val_reconstruction_mse")):
            axis = axes[row_index, column]
            for method in ("AE-TL", "PCA"):
                frame = scenario_frame[scenario_frame["method"] == method].sort_values(
                    "representation_dim"
                )
                x = frame["representation_dim"].to_numpy(dtype=float)
                y = frame[f"{metric}_mean"].to_numpy(dtype=float)
                error = frame[f"{metric}_std"].fillna(0.0).to_numpy(dtype=float)
                line, = axis.plot(
                    x,
                    y,
                    color=colors[method],
                    marker=markers[method],
                    markersize=3.2,
                    linewidth=1.1,
                    label=method,
                )
                if method == "AE-TL":
                    axis.fill_between(x, y - error, y + error, color=colors[method], alpha=0.18)
                if column == 0 and row_index == 0:
                    handles.append(line)
                    labels.append(method)
            axis.axvline(192, color="#777777", linestyle=":", linewidth=0.8)
            axis.set_xticks(dims)
            axis.tick_params(direction="in")
            axis.grid(False)
            if row_index == 0:
                axis.set_title(scenario.label)
                axis.set_ylim(min(0.98, axis.get_ylim()[0]), 1.005)
            else:
                axis.set_yscale("log")
                axis.set_xlabel("Representation dimension")
        axes[0, 0].set_ylabel("Max-channel AUC")
        axes[1, 0].set_ylabel("Healthy validation MSE")

    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=2,
        frameon=False,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"fig_representation_dimension_comparison.{suffix}", dpi=300)
    plt.close(fig)


def plot_auc_by_scenario(
    aggregate: pd.DataFrame,
    output_dir: Path,
    dims: Tuple[int, ...],
) -> None:
    """Compare AE-TL and PCA dimension ablations in separate scenario panels."""
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.7,
        "axes.grid": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })
    colours = {"AE-TL": "#7BA7BC", "PCA": "#C97A6C"}
    styles = {"AE-TL": ("-", "o"), "PCA": ("--", "s")}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), constrained_layout=True)
    handles = []
    labels = []

    for axis, scenario in zip(axes, SCENARIOS):
        scenario_frame = aggregate[aggregate["scenario"] == scenario.key]
        panel_values = []
        for method in ("AE-TL", "PCA"):
            frame = scenario_frame[scenario_frame["method"] == method].sort_values(
                "representation_dim"
            )
            x = frame["representation_dim"].to_numpy(dtype=float)
            y = frame["auc_mean"].to_numpy(dtype=float)
            error = frame["auc_std"].fillna(0.0).to_numpy(dtype=float)
            linestyle, marker = styles[method]
            line, = axis.plot(
                x,
                y,
                color=colours[method],
                linestyle=linestyle,
                marker=marker,
                markersize=3.2,
                linewidth=1.1,
                label=method,
            )
            if method == "AE-TL":
                axis.fill_between(
                    x,
                    y - error,
                    y + error,
                    color=colours[method],
                    alpha=0.18,
                    linewidth=0,
                )
            if axis is axes[0]:
                handles.append(line)
                labels.append(method)
            panel_values.extend((y - error).tolist())
            panel_values.extend((y + error).tolist())

        lower = min(panel_values)
        padding = max(0.003, (1.0 - lower) * 0.06)
        axis.set_ylim(lower - padding, 1.003)
        axis.axvline(192, color="#777777", linestyle=":", linewidth=0.8)
        axis.set_title({
            "repair": "Local stiffening",
            "relocation": "Relocation",
            "drift": "Temperature response",
        }[scenario.key])
        axis.set_xlabel("Representation dimension")
        axis.set_xticks(dims)
        axis.tick_params(direction="in")
        axis.grid(False)

    axes[0].set_ylabel("Max-channel AUC")
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=2,
        frameon=False,
        handlelength=2.4,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"fig_representation_dimension_auc_by_scenario.{suffix}",
            dpi=300,
        )
    plt.close(fig)


def plot_training_curves(
    curves: pd.DataFrame,
    output_dir: Path,
    dims: Tuple[int, ...],
) -> None:
    target = curves[curves["stage"] == "target_finetune"].copy()
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "axes.grid": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })
    colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.88, len(dims)))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), constrained_layout=True)
    handles = []
    labels = []
    for axis, scenario in zip(axes, SCENARIOS):
        frame = target[target["scenario"] == scenario.key]
        for color, dim in zip(colors, dims):
            dim_frame = frame[frame["latent_dim"] == dim]
            grouped = dim_frame.groupby("epoch")["val_reconstruction_mse"]
            mean = grouped.mean().rolling(11, min_periods=1, center=True).mean()
            std = grouped.std(ddof=1).fillna(0.0).rolling(11, min_periods=1, center=True).mean()
            x = mean.index.to_numpy(dtype=float)
            y = mean.to_numpy(dtype=float)
            err = std.to_numpy(dtype=float)
            line, = axis.plot(x, y, color=color, linewidth=0.9, label=str(dim))
            axis.fill_between(x, y - err, y + err, color=color, alpha=0.10)
            if axis is axes[0]:
                handles.append(line)
                labels.append(str(dim))
        axis.set_title({
            "repair": "Local stiffening",
            "relocation": "Relocation",
            "drift": "Temperature response",
        }[scenario.key])
        axis.set_xlabel("Fine-tuning epoch")
        axis.set_yscale("log")
        axis.tick_params(direction="in")
        axis.grid(False)
    axes[0].set_ylabel("Healthy validation MSE")
    fig.legend(
        handles,
        labels,
        title="AE bottleneck",
        loc="outside upper center",
        ncol=len(dims),
        frameon=False,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"fig_ae_bottleneck_training_curves.{suffix}", dpi=300)
    plt.close(fig)


def plot_training_curves_with_pca(
    curves: pd.DataFrame,
    aggregate: pd.DataFrame,
    output_dir: Path,
    dims: Tuple[int, ...],
) -> None:
    """Overlay PCA validation-MSE levels on the three AE fine-tuning panels."""
    target = curves[curves["stage"] == "target_finetune"].copy()
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "axes.grid": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })
    colours = plt.get_cmap("viridis")(np.linspace(0.12, 0.88, len(dims)))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), constrained_layout=True)
    dimension_handles = []
    dimension_labels = []

    for axis, scenario in zip(axes, SCENARIOS):
        frame = target[target["scenario"] == scenario.key]
        pca_frame = aggregate[
            (aggregate["scenario"] == scenario.key)
            & (aggregate["method"] == "PCA")
        ].set_index("representation_dim")
        panel_values = []

        for colour, dim in zip(colours, dims):
            dim_frame = frame[frame["latent_dim"] == dim]
            grouped = dim_frame.groupby("epoch")["val_reconstruction_mse"]
            mean = grouped.mean().rolling(11, min_periods=1, center=True).mean()
            std = (
                grouped.std(ddof=1)
                .fillna(0.0)
                .rolling(11, min_periods=1, center=True)
                .mean()
            )
            x = mean.index.to_numpy(dtype=float)
            y = mean.to_numpy(dtype=float)
            err = std.to_numpy(dtype=float)
            line, = axis.plot(
                x,
                y,
                color=colour,
                linewidth=0.9,
                zorder=3,
            )
            axis.fill_between(
                x,
                np.maximum(y - err, np.finfo(float).tiny),
                y + err,
                color=colour,
                alpha=0.08,
                linewidth=0,
                zorder=2,
            )
            pca_mse = float(
                pca_frame.loc[dim, "best_val_reconstruction_mse_mean"]
            )
            axis.axhline(
                pca_mse,
                color=colour,
                linestyle="--",
                linewidth=0.75,
                alpha=0.85,
                zorder=1,
            )
            panel_values.extend(y.tolist())
            panel_values.append(pca_mse)
            if axis is axes[0]:
                dimension_handles.append(line)
                dimension_labels.append(str(dim))

        positive_values = np.asarray(panel_values, dtype=float)
        positive_values = positive_values[positive_values > 0]
        axis.set_ylim(positive_values.min() * 0.65, positive_values.max() * 1.25)
        axis.set_title(scenario.label)
        axis.set_xlabel("Fine-tuning epoch")
        axis.set_yscale("log")
        axis.tick_params(direction="in")
        axis.grid(False)

    axes[0].set_ylabel("Healthy validation MSE")
    method_handles = [
        Line2D([0], [0], color="#333333", linewidth=1.0, linestyle="-"),
        Line2D([0], [0], color="#333333", linewidth=0.9, linestyle="--"),
    ]
    fig.legend(
        dimension_handles + method_handles,
        dimension_labels + ["AE-TL", "PCA"],
        loc="outside upper center",
        ncol=len(dims) + 2,
        frameon=False,
        columnspacing=0.9,
        handlelength=1.8,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"fig_ae_pca_dimension_mse_curves.{suffix}",
            dpi=300,
        )
    plt.close(fig)


def plot_method_overlay(
    aggregate: pd.DataFrame,
    output_dir: Path,
    dims: Tuple[int, ...],
) -> None:
    """Overlay all three scenarios; colour encodes scenario, style encodes method."""
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 8,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "axes.grid": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })
    scenario_colours = {
        "repair": "#6B8EAE",
        "relocation": "#8EAE8B",
        "drift": "#C49A6C",
    }
    method_styles = {
        "AE-TL": ("-", "o"),
        "PCA": ("--", "s"),
    }
    fig, axis = plt.subplots(figsize=(5.2, 3.05), constrained_layout=True)
    for scenario in SCENARIOS:
        scenario_frame = aggregate[aggregate["scenario"] == scenario.key]
        for method in ("AE-TL", "PCA"):
            frame = scenario_frame[scenario_frame["method"] == method].sort_values(
                "representation_dim"
            )
            x = frame["representation_dim"].to_numpy(dtype=float)
            y = frame["auc_mean"].to_numpy(dtype=float)
            error = frame["auc_std"].fillna(0.0).to_numpy(dtype=float)
            linestyle, marker = method_styles[method]
            axis.plot(
                x,
                y,
                color=scenario_colours[scenario.key],
                linestyle=linestyle,
                marker=marker,
                markersize=3.2,
                linewidth=1.1,
                label=f"{scenario.label} — {method}",
            )
            if method == "AE-TL":
                axis.fill_between(
                    x,
                    y - error,
                    y + error,
                    color=scenario_colours[scenario.key],
                    alpha=0.12,
                    linewidth=0,
                )
    axis.axvline(192, color="#777777", linestyle=":", linewidth=0.8)
    axis.set_xlabel("Representation dimension")
    axis.set_ylabel("Max-channel AUC")
    axis.set_xticks(dims)
    axis.set_ylim(0.90, 1.005)
    axis.tick_params(direction="in")
    axis.grid(False)
    fig.legend(
        loc="outside upper center",
        ncol=2,
        frameon=False,
        columnspacing=1.0,
        handlelength=2.4,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"fig_representation_dimension_overlay.{suffix}", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dims = tuple(sorted(set(args.dims)))
    seeds = tuple(args.seeds)
    if not dims or any(dim <= 0 or dim > 192 for dim in dims):
        raise ValueError("All dimensions must be in [1, 192]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    if device.type == "cuda":
        print(f"[gpu] {torch.cuda.get_device_name(0)}")

    source = source_arrays()
    scenario_data = {scenario.key: scenario_arrays(scenario) for scenario in SCENARIOS}
    summary_path = args.output_dir / "run_summary.csv"
    curves_path = args.output_dir / "epoch_curves.csv"
    source_summary_path = args.output_dir / "source_pretraining_summary.csv"
    summaries = [] if args.force else load_rows(summary_path)
    curves = [] if args.force else load_rows(curves_path)
    source_summaries = [] if args.force else load_rows(source_summary_path)

    source_states: Dict[int, Dict[str, torch.Tensor]] = {}
    for dim in dims:
        checkpoint = checkpoint_dir / f"source_latent_{dim}_seed_{SOURCE_SEED}.pth"
        if dim == 192 and not args.force:
            if not SOURCE_REFERENCE_CHECKPOINT.exists():
                raise FileNotFoundError(SOURCE_REFERENCE_CHECKPOINT)
            state = torch.load(SOURCE_REFERENCE_CHECKPOINT, map_location=device)
            model = build_model(dim, device)
            model.load_state_dict(state)
            source_states[dim] = clone_state(model)
            if not any(int(row.get("latent_dim")) == dim for row in source_summaries):
                source_summaries.append({
                    "latent_dim": dim,
                    "source_seed": SOURCE_SEED,
                    "best_epoch": np.nan,
                    "best_source_val_reconstruction_mse": reconstruction_mse(
                        model, source["val"], device
                    ),
                    "parameter_count": parameter_count(model),
                    "elapsed_seconds": 0.0,
                    "checkpoint_origin": "fixed_reference_source_checkpoint",
                    "checkpoint_sha256": sha256(SOURCE_REFERENCE_CHECKPOINT),
                })
                save_rows(source_summary_path, source_summaries)
            del model
            continue

        if checkpoint.exists() and not args.force:
            model = build_model(dim, device)
            model.load_state_dict(torch.load(checkpoint, map_location=device))
            source_states[dim] = clone_state(model)
            del model
            continue

        print(f"[source pretrain] latent={dim}")
        model, pretrain_curves, pretrain_summary = fit_source_model(
            dim, source, args.pretrain_epochs, args.batch_size, device
        )
        torch.save(model.state_dict(), checkpoint)
        source_states[dim] = clone_state(model)
        pretrain_summary.update({
            "checkpoint_origin": "dimension_ablation_pretraining",
            "checkpoint_sha256": sha256(checkpoint),
        })
        source_summaries = [
            row for row in source_summaries if int(row.get("latent_dim")) != dim
        ]
        source_summaries.append(pretrain_summary)
        curves.extend(pretrain_curves)
        save_rows(source_summary_path, source_summaries)
        save_rows(curves_path, curves)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    for scenario in SCENARIOS:
        arrays = scenario_data[scenario.key]
        for dim in dims:
            if args.force or not has_pca_run(summaries, scenario.key, dim):
                print(f"[PCA] scenario={scenario.key} dim={dim}")
                summaries.append(evaluate_pca(dim, scenario, arrays))
                save_rows(summary_path, summaries)
            for seed in seeds:
                if not args.force and has_ae_run(summaries, scenario.key, dim, seed):
                    continue
                print(f"[AE-TL] scenario={scenario.key} dim={dim} seed={seed}")
                run_curves, run_summary = fit_target_model(
                    source_states[dim],
                    dim,
                    scenario,
                    arrays,
                    seed,
                    args.finetune_epochs,
                    args.batch_size,
                    device,
                )
                summaries.append(run_summary)
                curves.extend(run_curves)
                save_rows(summary_path, summaries)
                save_rows(curves_path, curves)
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

    summary_frame = pd.DataFrame(summaries)
    curves_frame = pd.DataFrame(curves)
    expected_ae = len(SCENARIOS) * len(dims) * len(seeds)
    expected_pca = len(SCENARIOS) * len(dims)
    ae_rows = summary_frame[summary_frame["method"] == "AE-TL"]
    pca_rows = summary_frame[summary_frame["method"] == "PCA"]
    if len(ae_rows) != expected_ae or len(pca_rows) != expected_pca:
        raise AssertionError(
            f"Incomplete result grid: AE-TL {len(ae_rows)}/{expected_ae}, "
            f"PCA {len(pca_rows)}/{expected_pca}"
        )
    duplicate_columns = ["method", "scenario", "representation_dim", "seed"]
    if summary_frame.duplicated(duplicate_columns, keep=False).any():
        raise AssertionError("Duplicate method/scenario/dimension/seed result rows detected")
    required_numeric = [
        "best_val_reconstruction_mse", "auc", "tpr_at_fpr", "best_f1"
    ]
    if not np.isfinite(summary_frame[required_numeric].to_numpy(dtype=float)).all():
        raise AssertionError("Non-finite primary metric detected")
    aggregate = aggregate_results(summary_frame)
    aggregate.to_csv(
        args.output_dir / "aggregate_summary.csv", index=False, encoding="utf-8-sig"
    )
    plot_comparison(aggregate, args.output_dir, dims)
    plot_auc_by_scenario(aggregate, args.output_dir, dims)
    plot_training_curves(curves_frame, args.output_dir, dims)
    plot_training_curves_with_pca(curves_frame, aggregate, args.output_dir, dims)
    plot_method_overlay(aggregate, args.output_dir, dims)

    config = {
        "experiment": "PCA and AE representation-dimension ablation",
        "dimensions": list(dims),
        "source_pretraining_seed": SOURCE_SEED,
        "target_finetuning_seeds": list(seeds),
        "source_train_samples": SOURCE_TRAIN_SAMPLES,
        "source_val_samples": SOURCE_VAL_SAMPLES,
        "target_train_samples": TARGET_TRAIN_SAMPLES,
        "target_val_samples": TARGET_VAL_SAMPLES,
        "target_control_samples": TARGET_CONTROL_SAMPLES,
        "damage_samples": {scenario.key: len(scenario_data[scenario.key]["damage"]) for scenario in SCENARIOS},
        "pretrain_epochs": args.pretrain_epochs,
        "finetune_epochs": args.finetune_epochs,
        "batch_size": args.batch_size,
        "source_learning_rate": 3e-4,
        "encoder_finetune_learning_rate": 1e-5,
        "decoder_finetune_learning_rate": 1e-3,
        "cosine_eta_min": 1e-6,
        "input_noise_std": 0.005,
        "gradient_clip": 1.0,
        "ae_surrounding_widths": {
            "encoder": [768, 384, 192],
            "decoder": [192, 384, 768],
        },
        "scenarios": [asdict(scenario) for scenario in SCENARIOS],
        "data_sha256": {
            "source_health": sha256(SOURCE_HEALTH),
            **{
                f"{scenario.key}_health": sha256(
                    DATA_DIR / scenario.target_health / "preprocessed_data_raw.npz"
                )
                for scenario in SCENARIOS
            },
            **{
                f"{scenario.key}_damage": sha256(
                    DATA_DIR / scenario.target_damage / "preprocessed_data_raw.npz"
                )
                for scenario in SCENARIOS
            },
        },
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
    }
    (args.output_dir / "experiment_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[done] {args.output_dir}")


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
