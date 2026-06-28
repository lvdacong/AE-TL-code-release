"""
AD_gen_sensor_drift_figure.py
==============================
Generate a 2-row × 5-column panel figure showing non-uniform sensor drift.
Top row:    original vs drifted mean signal overlay per drift ratio.
Bottom row: effective per-channel drift ratio, exposing spatial non-uniformity.

Each column is wrapped by a black border, with the column label "r = X"
placed at the bottom — consistent visual style with Fig 3 and Fig 4.

Usage:
    cd script && python AD_gen_sensor_drift_figure.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from AE_train_model_auxiliary import apply_style, PLOT_STYLE
from AE_run_sd_difficulty_ablation import CHANNEL_DRIFT_FACTORS

FIGURE_OUTPUT = os.path.join(SCRIPT_DIR, "AE_model_train_and_detect_output", "Sensor_Drift")
AD_OUTPUT = os.path.join(SCRIPT_DIR, "AD_preprocess_datasets_output")

# Drift parameters
DRIFT_RATIOS = [0.05, 0.10, 0.15, 0.20, 0.25]

# Morandi palette
COLOR_ORIGINAL = "#7BA7BC"   # blue
COLOR_DRIFTED  = "#C97A6C"   # red
COLOR_FACTOR   = "#B8A9C9"   # muted purple for drift ratio bars

N_SAMPLES = 30


def main():
    print("=" * 60)
    print("Generating all-channel sensor drift visualization")
    print("=" * 60)

    # Load healthy data
    npz_path = os.path.join(AD_OUTPUT, "health_original_2000",
                            "preprocessed_data_raw.npz")
    V = np.load(npz_path)["V"].astype(np.float32)
    ch_mean_original = V[:N_SAMPLES].mean(axis=0)  # (252,)

    n_ch = len(ch_mean_original)
    drift_factors = CHANNEL_DRIFT_FACTORS[:n_ch]
    x = np.arange(n_ch)

    apply_style(PLOT_STYLE)

    # figsize 高度 5.6 + top=0.86 给顶部外部图例留约 0.78 in
    fig, all_axes = plt.subplots(
        2, 5, figsize=(19.5, 5.6),
        gridspec_kw={"height_ratios": [2.0, 0.9]},
        sharex="col",
    )
    fig.subplots_adjust(wspace=0.13, hspace=0.02,
                        left=0.04, right=0.99, bottom=0.12, top=0.86)

    top_axes = all_axes[0]
    bot_axes = all_axes[1]

    # Share y-axis within each row
    for ax in top_axes[1:]:
        ax.sharey(top_axes[0])
    for ax in bot_axes[1:]:
        ax.sharey(bot_axes[0])

    for i, r in enumerate(DRIFT_RATIOS):
        ax_top = top_axes[i]
        ax_bot = bot_axes[i]

        # --- Top row: signal overlay ---
        ch_mean_drifted = ch_mean_original * (1 + r * drift_factors)

        ax_top.plot(x, ch_mean_original, color=COLOR_ORIGINAL, linewidth=0.9,
                    alpha=0.90, label="Original")
        ax_top.plot(x, ch_mean_drifted, color=COLOR_DRIFTED, linewidth=0.9,
                    alpha=0.90, label="Drifted")
        ax_top.fill_between(x, ch_mean_original, ch_mean_drifted,
                            color=COLOR_DRIFTED, alpha=0.25)

        ax_top.set_xlim(0, n_ch - 1)
        ax_top.set_xticks([0, 125, 250])
        ax_top.tick_params(axis="both", direction="in", labelsize=18,
                           labelbottom=False)
        ax_top.grid(False)

        # --- Bottom row: per-channel effective drift ratio ---
        effective_drift = r * drift_factors  # shape (n_ch,)
        ax_bot.fill_between(x, 0, effective_drift, color=COLOR_FACTOR,
                            alpha=0.55, linewidth=0)
        ax_bot.plot(x, effective_drift, color=COLOR_FACTOR, linewidth=0.7,
                    alpha=0.8)
        ax_bot.axhline(r, color="#888888", linewidth=0.8, linestyle="--",
                       alpha=0.6)

        ax_bot.set_xlim(0, n_ch - 1)
        ax_bot.set_xticks([0, 125, 250])
        ax_bot.tick_params(axis="both", direction="in", labelsize=18)
        ax_bot.grid(False)

    # Simplified Y-axis labels
    top_axes[0].set_ylabel("Norm. signal", fontsize=22)
    bot_axes[0].set_ylabel("Drift ratio", fontsize=22)

    # Hide redundant y-tick labels
    for ax in top_axes[1:]:
        ax.tick_params(labelleft=False)
    for ax in bot_axes[1:]:
        ax.tick_params(labelleft=False)

    # Legend placed at figure-level outside top center
    handles, labels = top_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="upper center", bbox_to_anchor=(0.5, 0.98),
               ncol=len(labels), frameon=False, fontsize=16,
               handlelength=1.5)

    # ----- Bottom column labels -----
    pad_x = 0.004   # outward padding (figure coords)

    for i, r in enumerate(DRIFT_RATIOS):
        bbox_t = top_axes[i].get_position()
        bbox_b = bot_axes[i].get_position()
        x0 = min(bbox_t.x0, bbox_b.x0) - pad_x
        x1 = max(bbox_t.x1, bbox_b.x1) + pad_x

        # Column label "r = X" centered below the column
        cx = (x0 + x1) / 2
        fig.text(cx, 0.025, f"$r = {r:.2f}$",
                 ha="center", va="bottom", fontsize=24,
                 fontfamily="Times New Roman")

    os.makedirs(FIGURE_OUTPUT, exist_ok=True)
    output_path = os.path.join(FIGURE_OUTPUT, "fig_sensor_drift.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[saved] {output_path}")
    print("[done]")


if __name__ == "__main__":
    main()
