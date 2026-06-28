"""
AE_channel_ablation_auxiliary.py
================================
Per-channel trajectory analysis engine.
For each difficulty level, computes per-channel MAE across the training
progression (Pre-Shift -> Post-Shift -> N=50 -> ... -> N=400) and generates
a 2x3 subplot grid showing all 252 channels simultaneously.
No retraining required -- pure post-hoc evaluation.

Key design: per-difficulty adaptive channel highlighting.
At each difficulty, only channels whose Post-Shift delta exceeds a global
significance threshold are highlighted. This ensures that at low difficulty
(e.g. N_o=1 where only Ch.44 is offset) the actually affected channel is
visible, rather than globally-selected channels that happen to be inactive.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from AE_train_model_auxiliary import (
    Autoencoder, apply_style, FIG_DPI, PLOT_STYLE,
)
from AE_model_train_and_detect_auxiliary import compute_residuals
from AE_difficulty_ablation_auxiliary import (
    _MORANDI_6, _XLABEL_TO_DIFF_LABEL,
)

# Significance threshold: fraction of max delta across all difficulties
_SIG_FRAC = 0.03
# Maximum number of highlighted channels across the entire figure
_MAX_HIGHLIGHT = 5


# ========================================================================
# Per-channel MAE computation
# ========================================================================

def compute_all_channel_mae(
    V: np.ndarray,
    model: nn.Module,
    device: torch.device,
) -> np.ndarray:
    """
    Compute per-channel mean absolute residual.

    Returns:
        (D,) array of per-channel MAE
    """
    residuals = compute_residuals(V, model, device)  # (N, D)
    return np.mean(np.abs(residuals), axis=0)          # (D,)


# ========================================================================
# Channel selection engine
# ========================================================================

def _compute_channel_selection(
    trajectories: Dict, difficulty_values: list, D: int,
):
    """
    Compute per-difficulty active channels and unified color mapping.

    Uses delta trajectories (subtract Pre-Shift baseline) and a global
    significance threshold to identify channels that deviate meaningfully
    at each difficulty level.

    Returns:
        delta_trajectories, per_diff_active, channel_color_map,
        sig_threshold, all_active_sorted
    """
    # Delta trajectories: subtract Pre-Shift column
    delta_trajectories = {}
    for diff in difficulty_values:
        traj = trajectories[diff]
        delta_trajectories[diff] = traj - traj[:, 0:1]

    # Global threshold: _SIG_FRAC of max Post-Shift delta
    max_delta = max(
        np.nanmax(delta_trajectories[d][:, 1]) for d in difficulty_values
    )
    sig_threshold = max(max_delta * _SIG_FRAC, 1e-8)

    # Per-difficulty: channels exceeding threshold at Post-Shift
    per_diff_active: Dict[object, List[int]] = {}
    all_active: set = set()
    for diff in difficulty_values:
        delta_post = delta_trajectories[diff][:, 1]
        active = [ch for ch in range(D) if delta_post[ch] > sig_threshold]
        active.sort(key=lambda ch: -delta_post[ch])
        per_diff_active[diff] = active
        all_active.update(active)

    # Color map: sorted by max-difficulty delta (descending), capped
    max_diff = difficulty_values[-1]
    delta_max = delta_trajectories[max_diff][:, 1]
    all_active_sorted = sorted(all_active, key=lambda ch: -delta_max[ch])

    # Cap to _MAX_HIGHLIGHT most significant channels
    if len(all_active_sorted) > _MAX_HIGHLIGHT:
        all_active_sorted = all_active_sorted[:_MAX_HIGHLIGHT]
        kept = set(all_active_sorted)
        for diff in difficulty_values:
            per_diff_active[diff] = [
                ch for ch in per_diff_active[diff] if ch in kept
            ]

    channel_color_map = {
        ch: _MORANDI_6[i % len(_MORANDI_6)]
        for i, ch in enumerate(all_active_sorted)
    }

    return (delta_trajectories, per_diff_active, channel_color_map,
            sig_threshold, all_active_sorted)


# ========================================================================
# Summary output
# ========================================================================

def _save_summary(
    scenario_name: str,
    difficulty_values: list,
    delta_trajectories: Dict,
    per_diff_active: Dict,
    sig_threshold: float,
    all_active_sorted: List[int],
    output_dir: str,
) -> None:
    """Save channel selection summary to text file."""
    max_diff = difficulty_values[-1]
    delta_max = delta_trajectories[max_diff][:, 1]
    D = delta_max.shape[0]

    summary_path = os.path.join(output_dir, "channel_selection_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Scenario: {scenario_name}\n")
        f.write(f"Significance threshold: {sig_threshold:.6f}\n\n")

        f.write("Highlighted channels (by Post-Shift delta at max difficulty):\n")
        for ch in all_active_sorted:
            f.write(f"  Ch. {ch}: Post-Shift delta = {delta_max[ch]:.6f}\n")

        f.write(f"\nPer-difficulty active channels:\n")
        for diff in difficulty_values:
            active = per_diff_active[diff]
            delta_post = delta_trajectories[diff][:, 1]
            if active:
                labels = [f"Ch.{ch}({delta_post[ch]:.4f})" for ch in active]
                f.write(f"  {diff}: {', '.join(labels)}\n")
            else:
                f.write(f"  {diff}: (none)\n")

        f.write(f"\nAll channel Post-Shift delta (max difficulty = {max_diff}):\n")
        sorted_all = np.argsort(delta_max)[::-1]
        for rank, ch in enumerate(sorted_all[:20]):
            f.write(f"  Rank {rank+1}: Ch. {ch}, delta = {delta_max[ch]:.6f}\n")
    print(f"  Saved: {summary_path}")


# ========================================================================
# Main analysis engine
# ========================================================================

def run_channel_trajectory_analysis(
    scenario_name: str,
    difficulty_values: list,
    data_sizes: list,
    pretrain_model_path: str,
    model_dir: str,
    val_data_loader: Callable,
    output_dir: str,
    ae_config: dict,
    device: torch.device,
    xlabel: str,
) -> str:
    """
    Full-channel trajectory analysis for one scenario.

    For each difficulty d, computes per-channel MAE at 6 X-positions:
        [Pre-Shift, Post-Shift, N=50, N=100, N=200, N=400]

    Args:
        val_data_loader: callable(difficulty) -> V_val (N, D)
    """
    print(f"\n{'=' * 60}")
    print(f"[{scenario_name}] Full-Channel Trajectory Analysis")
    print(f"{'=' * 60}")

    os.makedirs(output_dir, exist_ok=True)

    # Load baseline data (difficulty=0)
    V_baseline = val_data_loader(difficulty_values[0])
    D = V_baseline.shape[1]

    # Load pretrain model
    pretrain_model = Autoencoder(
        D, ae_config["encoder_dims"], ae_config["latent_dim"],
        ae_config["decoder_dims"], ae_config["dropout"], ae_config["activation"],
    ).to(device)
    pretrain_model.load_state_dict(
        torch.load(pretrain_model_path, map_location=device, weights_only=True)
    )
    pretrain_model.eval()

    # Pre-Shift MAE: pretrain model on baseline data (same for all difficulties)
    mae_pre_shift = compute_all_channel_mae(V_baseline, pretrain_model, device)
    print(f"  Pre-Shift: max_ch_mae={mae_pre_shift.max():.6f}, "
          f"mean_ch_mae={mae_pre_shift.mean():.6f}")

    n_x = 2 + len(data_sizes)  # Pre-Shift + Post-Shift + N data sizes
    x_labels = ["Pre-\nShift", "Post-\nShift"] + [str(s) for s in data_sizes]

    # trajectories[diff] = (D, n_x) matrix of raw MAE
    trajectories: Dict = {}

    for j, diff in enumerate(difficulty_values):
        V_val = val_data_loader(diff)
        traj = np.zeros((D, n_x))

        # X=0: Pre-Shift (pretrain on baseline)
        traj[:, 0] = mae_pre_shift

        # X=1: Post-Shift (pretrain on shifted data)
        mae_post = compute_all_channel_mae(V_val, pretrain_model, device)
        traj[:, 1] = mae_post
        print(f"  diff={diff} Post-Shift: max={mae_post.max():.6f}, "
              f"mean={mae_post.mean():.6f}")

        # X=2..n_x-1: TL models
        for k, ds in enumerate(data_sizes):
            diff_str = f"{diff}".replace(".", "p")
            model_path = os.path.join(
                model_dir, f"diff_{diff_str}_n{ds}", "autoencoder.pth"
            )
            if not os.path.exists(model_path):
                print(f"  [skip] {model_path} not found")
                traj[:, 2 + k] = np.nan
                continue

            model = Autoencoder(
                D, ae_config["encoder_dims"], ae_config["latent_dim"],
                ae_config["decoder_dims"], ae_config["dropout"], ae_config["activation"],
            ).to(device)
            model.load_state_dict(
                torch.load(model_path, map_location=device, weights_only=True)
            )
            model.eval()

            mae_tl = compute_all_channel_mae(V_val, model, device)
            traj[:, 2 + k] = mae_tl
            print(f"  diff={diff} n={ds}: max={mae_tl.max():.6f}, "
                  f"mean={mae_tl.mean():.6f}")

        trajectories[diff] = traj

    # ---- Channel selection (per-difficulty adaptive) ----
    delta_trajs, per_diff_active, color_map, sig_thr, all_sorted = \
        _compute_channel_selection(trajectories, difficulty_values, D)

    # ---- Save CSV (raw MAE, same format as before) ----
    rows = []
    for diff in difficulty_values:
        traj = trajectories[diff]
        for ch in range(D):
            row = {"difficulty": diff, "channel": ch}
            for xi, lbl in enumerate(x_labels):
                row[lbl.replace("\n", "")] = traj[ch, xi]
            rows.append(row)
    csv_path = os.path.join(output_dir, "channel_ablation_results.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # ---- Save summary ----
    _save_summary(scenario_name, difficulty_values, delta_trajs,
                  per_diff_active, sig_thr, all_sorted, output_dir)

    # ---- Plot (raw MAE trajectories, highlighted by delta selection) ----
    fig_path = os.path.join(output_dir, "fig_channel_ablation.png")
    plot_channel_trajectories_grid(
        difficulty_values=difficulty_values,
        data_sizes=data_sizes,
        trajectories=trajectories,
        xlabel=xlabel,
        output_path=fig_path,
        per_diff_active=per_diff_active,
        channel_color_map=color_map,
        all_active_sorted=all_sorted,
    )

    print(f"\n[done] {scenario_name} channel trajectory analysis -> {output_dir}")
    return output_dir


# ========================================================================
# Replot from saved CSV
# ========================================================================

def replot_from_csv(
    scenario_name: str,
    difficulty_values: list,
    data_sizes: list,
    output_dir: str,
    xlabel: str,
) -> str:
    """Re-generate fig_channel_ablation.png from saved CSV data."""
    csv_path = os.path.join(output_dir, "channel_ablation_results.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"\n[replot] Loading {csv_path}")
    df = pd.read_csv(csv_path)
    D = int(df["channel"].max()) + 1
    n_x = 2 + len(data_sizes)
    col_names = ["Pre-Shift", "Post-Shift"] + [str(s) for s in data_sizes]

    # Reconstruct raw MAE trajectories
    trajectories = {}
    for diff in difficulty_values:
        sub = df[df["difficulty"] == diff].sort_values("channel")
        traj = np.zeros((D, n_x))
        for xi, col in enumerate(col_names):
            traj[:, xi] = sub[col].values
        trajectories[diff] = traj

    # Channel selection
    delta_trajs, per_diff_active, color_map, sig_thr, all_sorted = \
        _compute_channel_selection(trajectories, difficulty_values, D)

    # Plot (raw MAE trajectories)
    fig_path = os.path.join(output_dir, "fig_channel_ablation.png")
    plot_channel_trajectories_grid(
        difficulty_values=difficulty_values,
        data_sizes=data_sizes,
        trajectories=trajectories,
        xlabel=xlabel,
        output_path=fig_path,
        per_diff_active=per_diff_active,
        channel_color_map=color_map,
        all_active_sorted=all_sorted,
    )

    # Update summary
    _save_summary(scenario_name, difficulty_values, delta_trajs,
                  per_diff_active, sig_thr, all_sorted, output_dir)

    print(f"[replot] {scenario_name} -> {fig_path}")
    return fig_path


# ========================================================================
# Plotting
# ========================================================================

def plot_channel_trajectories_grid(
    difficulty_values: list,
    data_sizes: list,
    trajectories: Dict,
    xlabel: str,
    output_path: str,
    per_diff_active: Dict,
    channel_color_map: Dict[int, str],
    all_active_sorted: List[int],
) -> str:
    """
    2x3 line-plot grid: one subplot per difficulty level.
    Each subplot shows 252 channel trajectories (thin gray lines)
    with adaptively-selected channels highlighted in Morandi colours.

    X-axis: training stages (Pre-Shift, Post-Shift, N=50, 100, ...).
    Y-axis: raw per-channel MAE (per-row independent scaling).
    """

    apply_style(PLOT_STYLE)
    diff_label = _XLABEL_TO_DIFF_LABEL.get(xlabel, "Difficulty")
    n_diff = len(difficulty_values)

    n_rows = 2
    n_cols = (n_diff + 1) // 2
    # figsize 高度 9.6 给顶部外部图例留 0.6 in
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 9.6),
                             constrained_layout=True)
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    axes_flat = axes.flatten()

    n_x = 2 + len(data_sizes)
    x_positions = np.arange(n_x)
    x_labels = ["Pre-\nShift", "Post-\nShift"] + [str(s) for s in data_sizes]

    # Font sizes (figsize 18in, display ~6.3in -> scale ~0.35)
    # 字号梯度：label/title 26 > tick 20 > legend 14（legend < tick）
    fs_label = 26
    fs_tick = 20
    fs_title = 26
    fs_legend = 14

    # Global Y ceiling shared by all subplots (includes ALL columns so no
    # data exceeds the axis range). A single global vmax (rather than per-row)
    # ensures cross-difficulty visual comparability both along rows and columns.
    # trajectories[diff] shape: (D, n_x) where n_x = 2 + len(data_sizes)
    vmx_global = 0.0
    for diff in difficulty_values:
        traj = trajectories[diff]
        vmx_global = max(vmx_global, float(np.nanmax(traj)))
    global_vmax = max(vmx_global * 1.05, 1e-8)
    row_vmax = [global_vmax] * n_rows

    for idx, diff in enumerate(difficulty_values):
        ax = axes_flat[idx]
        traj = trajectories[diff]  # (D, n_x) raw MAE
        D = traj.shape[0]
        row = idx // n_cols

        active_set = set(per_diff_active.get(diff, []))

        # Draw all non-highlighted channels (thin gray)
        for ch in range(D):
            if ch in active_set:
                continue
            ax.plot(
                x_positions, traj[ch, :],
                color="#AAAAAA", linewidth=0.3, alpha=0.25,
            )

        # Highlight active channels
        for ch in per_diff_active.get(diff, []):
            color = channel_color_map.get(ch, "#333333")
            ax.plot(
                x_positions, traj[ch, :],
                color=color, linewidth=1.8, alpha=0.9,
                label=f"Ch. {ch}" if idx == 0 else None,
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=fs_tick - 2)
        ax.set_ylim(0, row_vmax[row])
        ax.tick_params(axis="y", labelsize=fs_tick)

        ax.text(
            0.03, 0.95, f"{diff_label} = {diff}",
            transform=ax.transAxes, fontsize=fs_title,
            va="top", fontstyle="italic",
        )

        if idx % n_cols == 0:
            ax.set_ylabel("Mean |Residual|", fontsize=fs_label)
        if idx >= n_cols:
            ax.set_xlabel(r"$N_{\mathrm{tgt}}$", fontsize=fs_label)

    # Hide extra subplots
    for idx in range(n_diff, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Legend placed outside upper center at figure level
    highlight_handles = [
        Line2D([0], [0], color=channel_color_map[ch], linewidth=1.8,
               label=f"Ch. {ch}")
        for ch in all_active_sorted if ch in channel_color_map
    ]
    highlight_handles.append(
        Line2D([0], [0], color="#AAAAAA", linewidth=0.8, alpha=0.5,
               label="Other channels")
    )
    fig.legend(
        handles=highlight_handles,
        loc="outside upper center",
        ncol=min(len(highlight_handles), 6),
        frameon=False, fontsize=fs_legend,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  [plot] Saved: {os.path.basename(output_path)}")
    return output_path
