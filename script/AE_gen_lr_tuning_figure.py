"""
AE_gen_lr_tuning_figure.py
===========================
Generate the learning-rate tuning figure for the paper (Section 5.2).
Panel (a): val_loss convergence curves comparing lr=3e-4 vs lr=1e-3 vs data replay.
Panel (b): Ch.123 recovery bar chart across key experiments.

Usage:
    cd script && python AE_gen_lr_tuning_figure.py
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from AE_train_model_auxiliary import apply_style, PLOT_STYLE, FIG_DPI

# ========================================
# Paths
# ========================================
RECOVERY_DIR = os.path.join(
    SCRIPT_DIR, "AE_model_train_and_detect_output",
    "Damage_Repaired", "recovery_study"
)

CURVE_SOURCES = {
    "lr=3e-4": os.path.join(RECOVERY_DIR, "phase1", "1A_sgd_1000ep_cosT1000", "training_losses.csv"),
    "lr=1e-3 (adopted)": os.path.join(RECOVERY_DIR, "phase1", "1B_sgd_1000ep_lr1e3", "training_losses.csv"),
    "lr=1e-3 + replay": os.path.join(RECOVERY_DIR, "phase3", "3A_sgd_lr1e3_replay1x", "training_losses.csv"),
}

PRETRAIN_BASELINE = 0.001148

# Morandi palette
C_BLUE_GRAY = "#8EAABD"
C_DUSTY_ROSE = "#CB8E8E"
C_SAGE = "#A8B5A2"
C_MAUVE = "#B39DBC"
C_WARM_BEIGE = "#C9B99A"
C_SLATE = "#7A8B8B"

OUTPUT_PATH = os.path.join(RECOVERY_DIR, "fig_lr_tuning.png")


def _load_recovery_data() -> list[dict]:
    """Load key experiment results from phase comparison CSVs."""
    experiments = []

    # Phase 1
    p1 = pd.read_csv(os.path.join(RECOVERY_DIR, "phase1", "phase1_comparison.csv"))
    for _, row in p1.iterrows():
        experiments.append(dict(row))

    # Phase 3
    p3 = pd.read_csv(os.path.join(RECOVERY_DIR, "phase3", "phase3_comparison.csv"))
    for _, row in p3.iterrows():
        experiments.append(dict(row))

    # Phase 4
    p4 = pd.read_csv(os.path.join(RECOVERY_DIR, "phase4", "phase4_comparison.csv"))
    for _, row in p4.iterrows():
        experiments.append(dict(row))

    # Phase 5
    p5 = pd.read_csv(os.path.join(RECOVERY_DIR, "phase5", "phase5_comparison.csv"))
    for _, row in p5.iterrows():
        experiments.append(dict(row))

    return experiments


def generate_figure():
    apply_style(PLOT_STYLE)

    # figsize 高度 5.5 预留顶部外部图例空间（原 5 + 内部图例会遮数据区）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                   constrained_layout=True)

    # ── Panel (a): Val loss convergence curves ──
    curve_styles = {
        "lr=3e-4": {"color": C_BLUE_GRAY, "ls": "-", "lw": 1.5},
        "lr=1e-3 (adopted)": {"color": C_DUSTY_ROSE, "ls": "-", "lw": 1.5},
        "lr=1e-3 + replay": {"color": C_SAGE, "ls": "--", "lw": 1.2},
    }

    for label, path in CURVE_SOURCES.items():
        df = pd.read_csv(path)
        epochs = np.arange(1, len(df) + 1)
        style = curve_styles[label]
        ax1.plot(epochs, df["val_loss"].values,
                 color=style["color"], ls=style["ls"], lw=style["lw"],
                 label=label)

    ax1.axhline(PRETRAIN_BASELINE, color="#666666", ls=":", lw=0.8,
                label="Pre-train baseline")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Loss (MSE)")
    ax1.set_xlim(1, 1000)
    ax1.set_ylim(0.00110, 0.00200)
    ax1.text(0.02, 0.95, "(a)", transform=ax1.transAxes, fontsize=22,
             va="top")

    # ── Panel (b): Ch.123 recovery bar chart ──
    bar_data = [
        ("lr=3e-4\n1000ep", 86.3, C_BLUE_GRAY),
        ("lr=1e-3\n1000ep", 92.9, C_DUSTY_ROSE),
        ("Replay\n1\u00d7", 51.0, C_SAGE),
        ("Aug\n5\u00d7", 94.1, C_MAUVE),
        ("L2-SP", 92.9, C_WARM_BEIGE),
    ]

    labels = [d[0] for d in bar_data]
    values = [d[1] for d in bar_data]
    colors = [d[2] for d in bar_data]

    bars = ax2.bar(labels, values, color=colors, edgecolor="white", width=0.65)

    # Highlight the adopted config
    bars[1].set_edgecolor("#333333")
    bars[1].set_linewidth(2.0)

    # Mark the replay failure
    bars[2].set_hatch("//")

    ax2.set_ylabel("Ch. 123 Recovery (%)")
    ax2.set_ylim(0, 105)
    ax2.axhline(100, color="#666666", ls=":", lw=0.8)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                 f"{val:.0f}%", ha="center", va="bottom", fontsize=14)

    ax2.text(0.02, 0.95, "(b)", transform=ax2.transAxes, fontsize=22,
             va="top")

    # 图例从 ax1 抽取，放整图外部上方横排
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="outside upper center",
               ncol=len(labels), frameon=False, fontsize=12)

    fig.savefig(OUTPUT_PATH, dpi=FIG_DPI)
    plt.close(fig)
    print(f"[saved] {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_figure()
