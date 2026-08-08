"""Target-sample-budget comparison between source reuse, AE-TL, and PCA.

Two controlled views are generated:

1. A strict 32-dimensional comparison, where AE-TL and target-refitted PCA use
   exactly the same target healthy subsets at budgets 50, 100, 200, and 400.
2. A 192-dimensional deployment comparison, where a frozen source PCA is
   compared with AE-TL at all budgets and target-refitted PCA is included only
   when the target sample count is sufficient to estimate 192 components.

Target validation, healthy control, and damage evaluation sets are fixed across
all budgets. Damage labels are never used for fitting or model selection.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

# Required by PyTorch for reproducible CUDA matrix multiplications when
# deterministic algorithms are enabled by the shared experiment utilities.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

import AE_run_representation_dimension_ablation as base


OUTPUT_DIR = (
    base.MODEL_DIR / "Target_Sample_Budget_Ablation"
)
DEFAULT_BUDGETS = (50, 100, 200, 400)
DEFAULT_DIMS = (32, 192)
DEFAULT_SEEDS = base.DEFAULT_SEEDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--dims", type=int, nargs="+", default=list(DEFAULT_DIMS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--finetune-epochs", type=int, default=base.FINETUNE_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=base.BATCH_SIZE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict("records")


def save_rows(path: Path, rows: Iterable[dict]) -> None:
    pd.DataFrame(list(rows)).to_csv(path, index=False, encoding="utf-8-sig")


def source_checkpoint(latent_dim: int) -> Path:
    if latent_dim == 192:
        return base.SOURCE_REFERENCE_CHECKPOINT
    return (
        base.OUTPUT_DIR
        / "checkpoints"
        / f"source_latent_{latent_dim}_seed_{base.SOURCE_SEED}.pth"
    )


def load_source_state(latent_dim: int, device: torch.device) -> Dict[str, torch.Tensor]:
    path = source_checkpoint(latent_dim)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing fixed source checkpoint for latent dimension {latent_dim}: {path}"
        )
    return torch.load(path, map_location=device)


def subset_indices(pool_size: int, budget: int, seed: int) -> np.ndarray:
    if budget > pool_size:
        raise ValueError(f"Budget {budget} exceeds the fixed pool size {pool_size}")
    permutation = np.random.default_rng(seed).permutation(pool_size)
    return permutation[:budget]


def detection_metrics_from_reconstruction(
    arrays: Dict[str, np.ndarray],
    reconstruct,
) -> dict:
    val_reconstruction = reconstruct(arrays["val"])
    damage_reconstruction = reconstruct(arrays["damage"])
    control_reconstruction = reconstruct(arrays["control"])
    damage_scores = np.max(
        np.abs(arrays["damage"] - damage_reconstruction), axis=1
    )
    control_scores = np.max(
        np.abs(arrays["control"] - control_reconstruction), axis=1
    )
    metrics = base.compute_detection_metrics(
        damage_scores, control_scores, n_bootstrap=0
    )
    return {
        "val_reconstruction_mse": float(
            np.mean((arrays["val"] - val_reconstruction) ** 2)
        ),
        "auc": float(metrics["auc"]),
        "tpr_at_fpr": float(metrics["tpr_at_fpr"]),
        "best_f1": float(metrics["best_f1"]),
    }


def evaluate_ae_state(
    source_state: Dict[str, torch.Tensor],
    latent_dim: int,
    arrays: Dict[str, np.ndarray],
    device: torch.device,
) -> dict:
    model = base.build_model(latent_dim, device)
    model.load_state_dict(source_state)
    model.eval()

    def reconstruct(values: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.from_numpy(values).to(device)
            return model(tensor).detach().cpu().numpy()

    return detection_metrics_from_reconstruction(arrays, reconstruct)


def evaluate_pca(
    fit_values: np.ndarray,
    latent_dim: int,
    arrays: Dict[str, np.ndarray],
) -> dict:
    if fit_values.shape[0] < latent_dim:
        raise ValueError(
            f"PCA-{latent_dim} requires at least {latent_dim} samples; "
            f"received {fit_values.shape[0]}"
        )
    model = PCA(n_components=latent_dim, svd_solver="full")
    model.fit(fit_values)

    def reconstruct(values: np.ndarray) -> np.ndarray:
        return model.inverse_transform(model.transform(values))

    result = detection_metrics_from_reconstruction(arrays, reconstruct)
    result["explained_variance_ratio"] = float(
        model.explained_variance_ratio_.sum()
    )
    return result


def row_exists(
    rows: Sequence[dict],
    method: str,
    scenario: str,
    latent_dim: int,
    target_samples: int,
    subset_seed: int,
) -> bool:
    return any(
        row.get("method") == method
        and row.get("scenario") == scenario
        and int(row.get("representation_dim")) == latent_dim
        and int(row.get("target_samples")) == target_samples
        and int(float(row.get("subset_seed"))) == subset_seed
        for row in rows
    )


def reuse_full_budget_ae_rows(
    rows: List[dict],
    latent_dim: int,
    seeds: Sequence[int],
) -> None:
    source_path = base.OUTPUT_DIR / "run_summary.csv"
    if not source_path.exists():
        return
    existing = pd.read_csv(source_path)
    existing = existing[
        (existing["method"] == "AE-TL")
        & (existing["representation_dim"] == latent_dim)
    ]
    for scenario in base.SCENARIOS:
        for seed in seeds:
            if row_exists(rows, "AE-TL", scenario.key, latent_dim, 400, seed):
                continue
            match = existing[
                (existing["scenario"] == scenario.key)
                & (existing["seed"] == seed)
            ]
            if len(match) != 1:
                continue
            source = match.iloc[0]
            rows.append({
                "method": "AE-TL",
                "calibration": "target_finetune",
                "scenario": scenario.key,
                "scenario_label": scenario.label,
                "representation_dim": latent_dim,
                "target_samples": 400,
                "subset_seed": seed,
                "val_reconstruction_mse": float(
                    source["best_val_reconstruction_mse"]
                ),
                "auc": float(source["auc"]),
                "tpr_at_fpr": float(source["tpr_at_fpr"]),
                "best_f1": float(source["best_f1"]),
                "explained_variance_ratio": np.nan,
                "reused_from_representation_ablation": True,
            })


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    metrics = ["val_reconstruction_mse", "auc", "tpr_at_fpr", "best_f1"]
    grouped = rows.groupby(
        [
            "method",
            "calibration",
            "scenario",
            "scenario_label",
            "representation_dim",
            "target_samples",
        ],
        as_index=False,
    )[metrics].agg(["mean", "std", "count"])
    grouped.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        for column in grouped.columns.to_flat_index()
    ]
    return grouped


def set_plot_style() -> None:
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


def plot_common32(aggregated: pd.DataFrame, output_dir: Path) -> None:
    set_plot_style()
    frame = aggregated[aggregated["representation_dim"] == 32]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), constrained_layout=True)
    styles = {
        "AE-TL": ("#7BA7BC", "-", "o"),
        "PCA": ("#C97A6C", "--", "s"),
    }
    handles = []
    labels = []
    for axis, scenario in zip(axes, base.SCENARIOS):
        scenario_frame = frame[frame["scenario"] == scenario.key]
        for method in ("AE-TL", "PCA"):
            method_frame = scenario_frame[scenario_frame["method"] == method].sort_values(
                "target_samples"
            )
            x = method_frame["target_samples"].to_numpy(dtype=float)
            y = method_frame["auc_mean"].to_numpy(dtype=float)
            error = method_frame["auc_std"].fillna(0.0).to_numpy(dtype=float)
            colour, linestyle, marker = styles[method]
            line, = axis.plot(
                x,
                y,
                color=colour,
                linestyle=linestyle,
                marker=marker,
                markersize=3.2,
                linewidth=1.1,
                label=method,
            )
            axis.fill_between(
                x,
                y - error,
                y + error,
                color=colour,
                alpha=0.15,
                linewidth=0,
            )
            if axis is axes[0]:
                handles.append(line)
                labels.append(method)
        axis.set_title(scenario.label)
        axis.set_xlabel("Target healthy samples")
        axis.set_xticks([0, *DEFAULT_BUDGETS])
        axis.set_xticklabels(["Source", "50", "100", "200", "400"])
        axis.set_ylim(max(0.0, scenario_frame["auc_mean"].min() - 0.03), 1.005)
        axis.grid(False)
    axes[0].set_ylabel("Max-channel AUC")
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"fig_common32_target_budget_auc.{suffix}", dpi=300)
    plt.close(fig)


def prepare_pca64_rows(
    source: Dict[str, np.ndarray],
    scenario_data: Dict[str, Dict[str, np.ndarray]],
    budgets: Sequence[int],
    seeds: Sequence[int],
    output_dir: Path,
    force: bool,
) -> pd.DataFrame:
    path = output_dir / "pca64_run_summary.csv"
    rows = [] if force else load_rows(path)
    latent_dim = 64

    for scenario in base.SCENARIOS:
        arrays = scenario_data[scenario.key]
        if not row_exists(rows, "PCA-64", scenario.key, latent_dim, 0, -1):
            result = evaluate_pca(source["train"], latent_dim, arrays)
            rows.append({
                "method": "PCA-64",
                "calibration": "source_frozen",
                "scenario": scenario.key,
                "scenario_label": scenario.label,
                "representation_dim": latent_dim,
                "target_samples": 0,
                "subset_seed": -1,
                **result,
                "reused_from_representation_ablation": False,
            })
            save_rows(path, rows)

        for budget in budgets:
            if budget <= latent_dim:
                continue
            for seed in seeds:
                if row_exists(
                    rows, "PCA-64", scenario.key, latent_dim, budget, seed
                ):
                    continue
                indices = subset_indices(base.TARGET_TRAIN_SAMPLES, budget, seed)
                result = evaluate_pca(arrays["train"][indices], latent_dim, arrays)
                rows.append({
                    "method": "PCA-64",
                    "calibration": "target_refit",
                    "scenario": scenario.key,
                    "scenario_label": scenario.label,
                    "representation_dim": latent_dim,
                    "target_samples": budget,
                    "subset_seed": seed,
                    **result,
                    "reused_from_representation_ablation": False,
                })
                save_rows(path, rows)

    frame = pd.DataFrame(rows).drop_duplicates(
        ["method", "scenario", "representation_dim", "target_samples", "subset_seed"],
        keep="last",
    ).sort_values(["scenario", "target_samples", "subset_seed"])
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return frame


def plot_common32_with_pca64(
    aggregated: pd.DataFrame,
    pca64_aggregated: pd.DataFrame,
    output_dir: Path,
) -> None:
    set_plot_style()
    frame = aggregated[aggregated["representation_dim"] == 32]
    sample_axis = np.asarray([0, *DEFAULT_BUDGETS], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), constrained_layout=True)
    styles = {
        "AE-TL-32": ("#7BA7BC", "-", "o"),
        "PCA-32": ("#C97A6C", "--", "s"),
        "PCA-64": ("#8F83B8", "-.", "^"),
    }
    handles = []
    labels = []

    for axis, scenario in zip(axes, base.SCENARIOS):
        scenario_frame = frame[frame["scenario"] == scenario.key]
        plotted_values = []
        for method, label in (("AE-TL", "AE-TL-32"), ("PCA", "PCA-32")):
            method_frame = scenario_frame[scenario_frame["method"] == method].sort_values(
                "target_samples"
            )
            x = method_frame["target_samples"].to_numpy(dtype=float)
            y = method_frame["auc_mean"].to_numpy(dtype=float)
            error = method_frame["auc_std"].fillna(0.0).to_numpy(dtype=float)
            colour, linestyle, marker = styles[label]
            line, = axis.plot(
                x,
                y,
                color=colour,
                linestyle=linestyle,
                marker=marker,
                markersize=3.2,
                linewidth=1.1,
            )
            axis.fill_between(
                x,
                y - error,
                y + error,
                color=colour,
                alpha=0.15,
                linewidth=0,
            )
            plotted_values.extend((y - error).tolist())
            if axis is axes[0]:
                handles.append(line)
                labels.append(label)

        pca64 = pca64_aggregated[
            pca64_aggregated["scenario"] == scenario.key
        ].set_index("target_samples")
        y64 = np.full(sample_axis.shape, np.nan, dtype=float)
        error64 = np.full(sample_axis.shape, np.nan, dtype=float)
        for index, sample_count in enumerate(sample_axis.astype(int)):
            if sample_count not in pca64.index:
                continue
            row = pca64.loc[sample_count]
            y64[index] = float(row["auc_mean"])
            error64[index] = float(row["auc_std"]) if pd.notna(row["auc_std"]) else 0.0
        colour, linestyle, marker = styles["PCA-64"]
        line64, = axis.plot(
            sample_axis,
            y64,
            color=colour,
            linestyle=linestyle,
            marker=marker,
            markersize=3.2,
            linewidth=1.1,
        )
        axis.fill_between(
            sample_axis,
            y64 - error64,
            y64 + error64,
            color=colour,
            alpha=0.12,
            linewidth=0,
        )
        plotted_values.extend((y64 - error64)[np.isfinite(y64)].tolist())
        if axis is axes[0]:
            handles.append(line64)
            labels.append("PCA-64")

        axis.set_title(scenario.label)
        axis.set_xlabel("Target healthy samples")
        axis.set_xticks(sample_axis)
        axis.set_xticklabels(["Source", "50", "100", "200", "400"])
        axis.set_ylim(max(0.0, min(plotted_values) - 0.03), 1.005)
        axis.grid(False)

    axes[0].set_ylabel("Max-channel AUC")
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"fig_ae_pca32_pca64_target_budget_auc.{suffix}", dpi=300
        )
    plt.close(fig)


def plot_pca_adaptation_modes(
    aggregated: pd.DataFrame,
    pca64_aggregated: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot AE-TL with target-adapted and frozen PCA at two PCA ranks."""

    set_plot_style()
    sample_axis = np.asarray([0, *DEFAULT_BUDGETS], dtype=float)
    sample_counts = sample_axis.astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), constrained_layout=True)
    styles = {
        "AE-TL": ("#557F96", "-", "o", "#557F96"),
        "Target-adapted PCA (lower rank)": ("#C56F5C", "-", "s", "#C56F5C"),
        "Target-adapted PCA (higher rank)": ("#7565A8", "-", "^", "#7565A8"),
        "Frozen source PCA (lower rank)": ("#C56F5C", "--", "s", "white"),
        "Frozen source PCA (higher rank)": ("#7565A8", "--", "^", "white"),
    }
    # Matplotlib fills multi-column legends column-first; this sequence yields
    # AE-TL plus both adapted PCA curves on row one and both frozen curves on row two.
    legend_layout_order = [
        "AE-TL",
        "Frozen source PCA (lower rank)",
        "Target-adapted PCA (lower rank)",
        "Frozen source PCA (higher rank)",
        "Target-adapted PCA (higher rank)",
    ]
    legend_handles = {}

    def point(
        frame: pd.DataFrame,
        scenario_key: str,
        method: str,
        calibration: str,
        representation_dim: int,
        target_samples: int,
    ) -> Tuple[float, float]:
        match = frame[
            (frame["scenario"] == scenario_key)
            & (frame["method"] == method)
            & (frame["calibration"] == calibration)
            & (frame["representation_dim"] == representation_dim)
            & (frame["target_samples"] == target_samples)
        ]
        if len(match) != 1:
            raise ValueError(
                "Expected one aggregate row for "
                f"{scenario_key}, {method}, {calibration}, "
                f"dimension={representation_dim}, target_samples={target_samples}; "
                f"found {len(match)}"
            )
        row = match.iloc[0]
        error = float(row["auc_std"]) if pd.notna(row["auc_std"]) else 0.0
        return float(row["auc_mean"]), error

    for axis, scenario in zip(axes, base.SCENARIOS):
        ae_y = np.full(sample_axis.shape, np.nan, dtype=float)
        ae_error = np.full(sample_axis.shape, np.nan, dtype=float)
        lower_adapted_y = np.full(sample_axis.shape, np.nan, dtype=float)
        lower_adapted_error = np.full(sample_axis.shape, np.nan, dtype=float)
        higher_adapted_y = np.full(sample_axis.shape, np.nan, dtype=float)
        higher_adapted_error = np.full(sample_axis.shape, np.nan, dtype=float)

        ae_y[0], ae_error[0] = point(
            aggregated, scenario.key, "AE-TL", "source_frozen", 32, 0
        )
        lower_adapted_y[0], lower_adapted_error[0] = point(
            aggregated, scenario.key, "PCA", "source_frozen", 32, 0
        )
        higher_adapted_y[0], higher_adapted_error[0] = point(
            pca64_aggregated, scenario.key, "PCA-64", "source_frozen", 64, 0
        )

        for index, sample_count in enumerate(sample_counts[1:], start=1):
            ae_y[index], ae_error[index] = point(
                aggregated,
                scenario.key,
                "AE-TL",
                "target_finetune",
                32,
                sample_count,
            )
            lower_adapted_y[index], lower_adapted_error[index] = point(
                aggregated,
                scenario.key,
                "PCA",
                "target_refit",
                32,
                sample_count,
            )
            if sample_count > 64:
                higher_adapted_y[index], higher_adapted_error[index] = point(
                    pca64_aggregated,
                    scenario.key,
                    "PCA-64",
                    "target_refit",
                    64,
                    sample_count,
                )

        lower_frozen_y = np.full(sample_axis.shape, lower_adapted_y[0])
        higher_frozen_y = np.full(sample_axis.shape, higher_adapted_y[0])
        series = {
            "AE-TL": (ae_y, ae_error),
            "Target-adapted PCA (lower rank)": (
                lower_adapted_y,
                lower_adapted_error,
            ),
            "Target-adapted PCA (higher rank)": (
                higher_adapted_y,
                higher_adapted_error,
            ),
            "Frozen source PCA (lower rank)": (
                lower_frozen_y,
                np.zeros(sample_axis.shape),
            ),
            "Frozen source PCA (higher rank)": (
                higher_frozen_y,
                np.zeros(sample_axis.shape),
            ),
        }

        plotted_values = []
        plot_order = [
            "Frozen source PCA (lower rank)",
            "Frozen source PCA (higher rank)",
            "Target-adapted PCA (lower rank)",
            "Target-adapted PCA (higher rank)",
            "AE-TL",
        ]
        for label in plot_order:
            y, error = series[label]
            colour, linestyle, marker, marker_face = styles[label]
            is_frozen = label.startswith("Frozen")
            line, = axis.plot(
                sample_axis,
                y,
                color=colour,
                linestyle=linestyle,
                marker=marker,
                markerfacecolor=marker_face,
                markeredgecolor=colour,
                markeredgewidth=0.7,
                markersize=3.1,
                linewidth=0.95,
                zorder=2 if is_frozen else 3,
            )
            if not is_frozen:
                axis.fill_between(
                    sample_axis,
                    y - error,
                    y + error,
                    color=colour,
                    alpha=0.10,
                    linewidth=0,
                    zorder=1,
                )
            finite = np.isfinite(y - error)
            plotted_values.extend((y - error)[finite].tolist())
            if axis is axes[0]:
                legend_handles[label] = line

        axis.set_title(scenario.label)
        axis.set_xlabel("Target healthy samples")
        axis.set_xticks(sample_axis)
        axis.set_xticklabels(["Source", "50", "100", "200", "400"])
        axis.set_ylim(max(0.0, min(plotted_values) - 0.03), 1.005)
        axis.grid(False)

    axes[0].set_ylabel("Max-channel AUC")
    fig.legend(
        [legend_handles[label] for label in legend_layout_order],
        legend_layout_order,
        loc="outside upper center",
        ncol=3,
        frameon=False,
        fontsize=6.2,
        columnspacing=1.0,
        handlelength=2.2,
        handletextpad=0.45,
        borderaxespad=0.15,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"fig_ae_pca_adaptation_modes_target_budget_auc.{suffix}",
            dpi=300,
        )
    plt.close(fig)


def plot_deployment192(aggregated: pd.DataFrame, output_dir: Path) -> None:
    set_plot_style()
    frame = aggregated[aggregated["representation_dim"] == 192]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), constrained_layout=True)
    handles = []
    labels = []
    for axis, scenario in zip(axes, base.SCENARIOS):
        scenario_frame = frame[frame["scenario"] == scenario.key]
        ae = scenario_frame[scenario_frame["method"] == "AE-TL"].sort_values(
            "target_samples"
        )
        line, = axis.plot(
            ae["target_samples"],
            ae["auc_mean"],
            color="#7BA7BC",
            linestyle="-",
            marker="o",
            markersize=3.2,
            linewidth=1.1,
            label="AE route",
        )
        ae_error = ae["auc_std"].fillna(0.0).to_numpy(dtype=float)
        axis.fill_between(
            ae["target_samples"].to_numpy(dtype=float),
            ae["auc_mean"].to_numpy(dtype=float) - ae_error,
            ae["auc_mean"].to_numpy(dtype=float) + ae_error,
            color="#7BA7BC",
            alpha=0.15,
            linewidth=0,
        )
        frozen = scenario_frame[
            (scenario_frame["method"] == "PCA")
            & (scenario_frame["calibration"] == "source_frozen")
        ]
        frozen_auc = float(frozen["auc_mean"].iloc[0])
        frozen_line = axis.axhline(
            frozen_auc,
            color="#777777",
            linestyle=":",
            linewidth=1.0,
            label="Frozen source PCA",
        )
        target_pca = scenario_frame[
            (scenario_frame["method"] == "PCA")
            & (scenario_frame["calibration"] == "target_refit")
        ].sort_values("target_samples")
        pca_line, = axis.plot(
            target_pca["target_samples"],
            target_pca["auc_mean"],
            color="#C97A6C",
            linestyle="--",
            marker="s",
            markersize=3.2,
            linewidth=1.1,
            label="Target-refitted PCA",
        )
        pca_error = target_pca["auc_std"].fillna(0.0).to_numpy(dtype=float)
        axis.fill_between(
            target_pca["target_samples"].to_numpy(dtype=float),
            target_pca["auc_mean"].to_numpy(dtype=float) - pca_error,
            target_pca["auc_mean"].to_numpy(dtype=float) + pca_error,
            color="#C97A6C",
            alpha=0.15,
            linewidth=0,
        )
        if axis is axes[0]:
            handles.extend([line, frozen_line, pca_line])
            labels.extend(["AE route", "Frozen source PCA", "Target-refitted PCA"])
        values = scenario_frame["auc_mean"].to_numpy(dtype=float)
        axis.set_ylim(max(0.0, values.min() - 0.03), 1.005)
        axis.set_title(scenario.label)
        axis.set_xlabel("Target healthy samples")
        axis.set_xticks([0, *DEFAULT_BUDGETS])
        axis.set_xticklabels(["Source", "50", "100", "200", "400"])
        axis.grid(False)
    axes[0].set_ylabel("Max-channel AUC")
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"fig_deployment192_target_budget_auc.{suffix}", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    budgets = tuple(sorted(set(args.budgets)))
    dims = tuple(sorted(set(args.dims)))
    seeds = tuple(args.seeds)
    if any(budget <= 0 or budget > base.TARGET_TRAIN_SAMPLES for budget in budgets):
        raise ValueError("Target budgets must be in [1, 400]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "run_summary.csv"
    rows = [] if args.force else load_rows(summary_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    source = base.source_arrays()
    scenario_data = {
        scenario.key: base.scenario_arrays(scenario) for scenario in base.SCENARIOS
    }

    for latent_dim in dims:
        source_state = load_source_state(latent_dim, device)
        reuse_full_budget_ae_rows(rows, latent_dim, seeds)
        for scenario in base.SCENARIOS:
            arrays = scenario_data[scenario.key]
            source_seed = -1
            if not row_exists(
                rows, "AE-TL", scenario.key, latent_dim, 0, source_seed
            ):
                result = evaluate_ae_state(
                    source_state, latent_dim, arrays, device
                )
                rows.append({
                    "method": "AE-TL",
                    "calibration": "source_frozen",
                    "scenario": scenario.key,
                    "scenario_label": scenario.label,
                    "representation_dim": latent_dim,
                    "target_samples": 0,
                    "subset_seed": source_seed,
                    **result,
                    "explained_variance_ratio": np.nan,
                    "reused_from_representation_ablation": False,
                })
                save_rows(summary_path, rows)

            if not row_exists(
                rows, "PCA", scenario.key, latent_dim, 0, source_seed
            ):
                result = evaluate_pca(source["train"], latent_dim, arrays)
                rows.append({
                    "method": "PCA",
                    "calibration": "source_frozen",
                    "scenario": scenario.key,
                    "scenario_label": scenario.label,
                    "representation_dim": latent_dim,
                    "target_samples": 0,
                    "subset_seed": source_seed,
                    **result,
                    "reused_from_representation_ablation": False,
                })
                save_rows(summary_path, rows)

            for budget in budgets:
                for seed in seeds:
                    indices = subset_indices(
                        base.TARGET_TRAIN_SAMPLES, budget, seed
                    )
                    subset_arrays = dict(arrays)
                    subset_arrays["train"] = arrays["train"][indices]

                    if not row_exists(
                        rows, "AE-TL", scenario.key, latent_dim, budget, seed
                    ):
                        _, summary = base.fit_target_model(
                            source_state,
                            latent_dim,
                            scenario,
                            subset_arrays,
                            seed,
                            args.finetune_epochs,
                            args.batch_size,
                            device,
                        )
                        rows.append({
                            "method": "AE-TL",
                            "calibration": "target_finetune",
                            "scenario": scenario.key,
                            "scenario_label": scenario.label,
                            "representation_dim": latent_dim,
                            "target_samples": budget,
                            "subset_seed": seed,
                            "val_reconstruction_mse": summary[
                                "best_val_reconstruction_mse"
                            ],
                            "auc": summary["auc"],
                            "tpr_at_fpr": summary["tpr_at_fpr"],
                            "best_f1": summary["best_f1"],
                            "explained_variance_ratio": np.nan,
                            "reused_from_representation_ablation": False,
                        })
                        save_rows(summary_path, rows)

                    if budget >= latent_dim and not row_exists(
                        rows, "PCA", scenario.key, latent_dim, budget, seed
                    ):
                        result = evaluate_pca(
                            subset_arrays["train"], latent_dim, arrays
                        )
                        rows.append({
                            "method": "PCA",
                            "calibration": "target_refit",
                            "scenario": scenario.key,
                            "scenario_label": scenario.label,
                            "representation_dim": latent_dim,
                            "target_samples": budget,
                            "subset_seed": seed,
                            **result,
                            "reused_from_representation_ablation": False,
                        })
                        save_rows(summary_path, rows)

                    if not row_exists(
                        rows,
                        "PCA-Pooled",
                        scenario.key,
                        latent_dim,
                        budget,
                        seed,
                    ):
                        pooled_fit = np.concatenate(
                            [source["train"], subset_arrays["train"]], axis=0
                        )
                        result = evaluate_pca(pooled_fit, latent_dim, arrays)
                        rows.append({
                            "method": "PCA-Pooled",
                            "calibration": "source_target_refit",
                            "scenario": scenario.key,
                            "scenario_label": scenario.label,
                            "representation_dim": latent_dim,
                            "target_samples": budget,
                            "subset_seed": seed,
                            **result,
                            "reused_from_representation_ablation": False,
                        })
                        save_rows(summary_path, rows)

    result_frame = pd.DataFrame(rows)
    result_frame = result_frame.drop_duplicates(
        [
            "method",
            "scenario",
            "representation_dim",
            "target_samples",
            "subset_seed",
        ],
        keep="last",
    ).sort_values(
        [
            "representation_dim",
            "scenario",
            "target_samples",
            "method",
            "subset_seed",
        ]
    )
    result_frame.to_csv(summary_path, index=False, encoding="utf-8-sig")
    aggregated = aggregate(result_frame)
    aggregated.to_csv(
        args.output_dir / "aggregate_summary.csv", index=False, encoding="utf-8-sig"
    )
    pca64_frame = prepare_pca64_rows(
        source,
        scenario_data,
        budgets,
        seeds,
        args.output_dir,
        args.force,
    )
    pca64_aggregated = aggregate(pca64_frame)
    pca64_aggregated.to_csv(
        args.output_dir / "pca64_aggregate_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    common32 = result_frame[
        (result_frame["representation_dim"] == 32)
        & (result_frame["target_samples"] > 0)
        & (result_frame["method"].isin(["AE-TL", "PCA", "PCA-Pooled"]))
    ]
    paired = common32.pivot_table(
        index=["scenario", "scenario_label", "target_samples", "subset_seed"],
        columns="method",
        values="auc",
    ).reset_index()
    paired["ae_minus_target_pca"] = paired["AE-TL"] - paired["PCA"]
    paired["ae_minus_pooled_pca"] = paired["AE-TL"] - paired["PCA-Pooled"]
    paired.to_csv(
        args.output_dir / "paired_auc_differences_common32.csv",
        index=False,
        encoding="utf-8-sig",
    )
    scenario_average = common32.groupby(
        ["method", "target_samples", "subset_seed"], as_index=False
    )["auc"].mean()
    scenario_average.to_csv(
        args.output_dir / "three_scenario_average_auc_common32.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if 32 in dims:
        plot_common32(aggregated, args.output_dir)
        plot_common32_with_pca64(aggregated, pca64_aggregated, args.output_dir)
        plot_pca_adaptation_modes(aggregated, pca64_aggregated, args.output_dir)
    if 192 in dims:
        plot_deployment192(aggregated, args.output_dir)

    config = {
        "experiment": "Target healthy sample budget comparison",
        "dimensions": list(dims),
        "budgets": list(budgets),
        "subset_seeds": list(seeds),
        "subset_rule": "For each seed, take the first N indices of one fixed permutation of the 400-sample target training pool.",
        "source_train_samples": base.SOURCE_TRAIN_SAMPLES,
        "target_training_pool": base.TARGET_TRAIN_SAMPLES,
        "target_validation_samples": base.TARGET_VAL_SAMPLES,
        "target_control_samples": base.TARGET_CONTROL_SAMPLES,
        "finetune_epochs": args.finetune_epochs,
        "batch_size": args.batch_size,
        "damage_labels_used_for_fitting_or_selection": False,
        "pca_feasibility_rule": "Target-refitted PCA is evaluated only when target sample count is at least the requested component count.",
        "pooled_pca_check": "PCA-Pooled is refitted on the fixed 1800-sample source training set plus exactly the same N target samples used by AE-TL.",
        "three_curve_figure": "AE-TL-32 and PCA-32 retain Source/50/100/200/400; PCA-64 retains the Source point, is undefined at N=50, and is target-refitted at N=100/200/400.",
        "five_curve_figure": "AE-TL uses the 32-dimensional experiment. Target-adapted PCA uses 32 and 64 components internally and is labelled as lower/higher rank in the figure; the 64-component target-adapted curve is undefined at N=50. Frozen source PCA at both ranks is repeated horizontally because it uses no target data.",
    }
    with (args.output_dir / "experiment_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    print(f"[done] {args.output_dir}")


if __name__ == "__main__":
    main()
