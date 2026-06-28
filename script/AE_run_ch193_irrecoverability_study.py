"""
AE_run_ch193_irrecoverability_study.py
======================================
Two-part study of Ch.193's irrecoverability in the Sensor_Offset scenario:
  Part 1: Inter-channel correlation disruption diagnostic  (fast, ~10 sec)
  Part 3: Cross-scenario recovery landscape                (fast, ~5 sec)

Output:
    AE_model_train_and_detect_output/Sensor_Offset/ch193_study/

Usage:
    cd script && python AE_run_ch193_irrecoverability_study.py --part all
    cd script && python AE_run_ch193_irrecoverability_study.py --part part1
    cd script && python AE_run_ch193_irrecoverability_study.py --part part3
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from AE_train_model_auxiliary import apply_style, FIG_DPI, PLOT_STYLE
from AE_difficulty_ablation_auxiliary import _MORANDI_6

# ========================================
# Paths & constants
# ========================================
AC_OUTPUT = os.path.join(SCRIPT_DIR, "AC_convert_and_extract_output")
AD_OUTPUT = os.path.join(SCRIPT_DIR, "AD_preprocess_datasets_output")
AE_OUTPUT = os.path.join(SCRIPT_DIR, "AE_model_train_and_detect_output")
SO_ABLATION = os.path.join(AE_OUTPUT, "Sensor_Offset", "difficulty_ablation")
OUTPUT_DIR = os.path.join(AE_OUTPUT, "Sensor_Offset", "ch193_study")


def get_offset_channels_at_oc(oc: int) -> list:
    """Auto-detect which channel indices are offset at the given offset_count
    by diffing measures_ID_offset_count_X.csv against measures_ID_original.csv."""
    orig = pd.read_csv(
        os.path.join(AC_OUTPUT, "measures_ID_original.csv")
    ).iloc[:, 0].astype(int).tolist()
    off = pd.read_csv(
        os.path.join(AC_OUTPUT, f"measures_ID_offset_count_{oc}.csv")
    ).iloc[:, 0].astype(int).tolist()
    return [i for i in range(len(orig)) if orig[i] != off[i]]


def load_V(oc: int) -> np.ndarray:
    npz = os.path.join(AD_OUTPUT, f"health_offset_count_{oc}_2000",
                       "preprocessed_data_raw.npz")
    return np.load(npz)["V"].astype(np.float32)


# ========================================================================
# Part 1: Correlation Disruption Diagnostic
# ========================================================================
def run_part1():
    print("\n" + "=" * 60)
    print("Part 1: Inter-Channel Correlation Disruption Diagnostic")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Auto-detect which channels are offset at oc=4
    offset_chs_oc4 = get_offset_channels_at_oc(4)
    print(f"  Offset channels at oc=4 (auto-detected): {offset_chs_oc4}")

    V0 = load_V(0)   # (2000, 252) baseline
    V4 = load_V(4)   # (2000, 252) offset_count=4

    # Pearson correlation matrices
    C0 = np.corrcoef(V0.T)  # (252, 252)
    C4 = np.corrcoef(V4.T)

    D = C0.shape[0]
    corr_disrupt = np.array([
        np.linalg.norm(C4[ch, :] - C0[ch, :]) for ch in range(D)
    ])

    # Load MAE delta from existing CSV
    csv = pd.read_csv(os.path.join(SO_ABLATION, "channel_ablation_results.csv"))
    s4 = csv[csv["difficulty"] == 4].sort_values("channel")
    mae_delta = s4["Post-Shift"].values - s4["Pre-Shift"].values

    print("\n  Key metrics at offset_count=4:")
    for ch in offset_chs_oc4:
        print(f"    Ch.{ch:>3d}: corr_L2={corr_disrupt[ch]:.4f}, "
              f"MAE_delta={mae_delta[ch]:.5f}")

    # Save
    pd.DataFrame({
        "channel": range(D),
        "corr_disruption_L2": corr_disrupt,
        "mae_delta_oc4": mae_delta,
    }).to_csv(os.path.join(OUTPUT_DIR, "correlation_disruption_metrics.csv"),
              index=False)

    # ---- Plot ----
    apply_style(PLOT_STYLE)
    # No figure legend here (only in-axes annotations), so (6, 6) square is fine
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

    offset_set = set(offset_chs_oc4)
    mask = np.array([ch not in offset_set for ch in range(D)])

    ax.scatter(corr_disrupt[mask], mae_delta[mask],
               color="#BBBBBB", s=12, alpha=0.5, zorder=1)

    colors = _MORANDI_6[:len(offset_chs_oc4)]
    # Alternate annotation offsets to avoid label overlap
    offsets_xy = [(14, 10), (-14, -16), (14, -16), (-14, 10), (14, 18)]
    ha_list = ["left", "right", "left", "right", "left"]
    for i, ch in enumerate(offset_chs_oc4):
        ax.scatter(corr_disrupt[ch], mae_delta[ch],
                   color=colors[i], s=90, zorder=3,
                   edgecolors="black", linewidths=0.5)
        oxy = offsets_xy[i % len(offsets_xy)]
        ha = ha_list[i % len(ha_list)]
        ax.annotate(
            f"Ch. {ch}", (corr_disrupt[ch], mae_delta[ch]),
            textcoords="offset points", xytext=oxy, ha=ha,
            fontsize=16, color=colors[i],
        )

    ax.set_xlabel("Correlation-Row $L_2$ Disruption", fontsize=20)
    ax.set_ylabel(r"Post-Shift $\Delta$ MAE", fontsize=20)
    ax.tick_params(labelsize=16)

    path = os.path.join(OUTPUT_DIR, "fig_correlation_disruption.png")
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"\n  Saved: {path}")


# ========================================================================
# Part 3: Cross-Scenario Recovery Landscape
# ========================================================================
def run_part3():
    print("\n" + "=" * 60)
    print("Part 3: Cross-Scenario Recovery Landscape")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    scenarios = [
        ("Structural Repair", "Damage_Repaired", 20, "$N_r = 20$"),
        ("Sensor Relocation", "Sensor_Offset", 5, "$N_o = 5$"),
        ("Sensor Drift", "Sensor_Drift", 0.25, "$r = 0.25$"),
    ]

    all_data = []
    for label, folder, max_d, diff_lbl in scenarios:
        csv_path = os.path.join(AE_OUTPUT, folder, "difficulty_ablation",
                                "channel_ablation_results.csv")
        df = pd.read_csv(csv_path)
        sub = df[df["difficulty"] == max_d].sort_values("channel")

        pre = sub["Pre-Shift"].values
        post = sub["Post-Shift"].values
        n400 = sub["400"].values
        delta = post - pre
        threshold = max(delta.max() * 0.03, 1e-8)

        chs = []
        for idx in range(len(sub)):
            if delta[idx] > threshold:
                ch = int(sub.iloc[idx]["channel"])
                rec = (post[idx] - n400[idx]) / delta[idx] * 100
                chs.append({"ch": ch, "delta": delta[idx], "rec": min(rec, 115)})

        # Keep top 5 by delta magnitude, then sort by recovery descending
        chs.sort(key=lambda x: -x["delta"])
        chs = chs[:5]
        chs.sort(key=lambda x: -x["rec"])
        all_data.append({"label": label, "diff_lbl": diff_lbl, "chs": chs})

        print(f"\n  {label} ({diff_lbl}):")
        for c in chs:
            print(f"    Ch.{c['ch']:>3d}: delta={c['delta']:.4f}, rec={c['rec']:.1f}%")

    # ---- Plot ----
    apply_style(PLOT_STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    for ax, sd in zip(axes, all_data):
        chs = sd["chs"]
        labels = [f"Ch. {c['ch']}" for c in chs]
        recs = [c["rec"] for c in chs]
        bar_colors = [_MORANDI_6[1] if r < 30 else _MORANDI_6[0] for r in recs]

        ax.barh(range(len(chs)), recs, color=bar_colors, height=0.6)
        ax.set_yticks(range(len(chs)))
        ax.set_yticklabels(labels, fontsize=18)
        ax.set_xlim(0, 120)
        ax.axvline(100, color="#CCCCCC", lw=0.8, ls="--")
        ax.set_xlabel("Recovery (%)", fontsize=20)
        ax.tick_params(axis="x", labelsize=16)
        ax.invert_yaxis()

        ax.text(0.97, 0.95, f"{sd['label']}\n{sd['diff_lbl']}",
                transform=ax.transAxes, fontsize=18, va="top", ha="right",
                fontstyle="italic")

    path = os.path.join(OUTPUT_DIR, "fig_recovery_landscape.png")
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"\n  Saved: {path}")


# ========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["all", "part1", "part3"],
                        default="all")
    args = parser.parse_args()

    if args.part in ("all", "part1"):
        run_part1()
    if args.part in ("all", "part3"):
        run_part3()

    print(f"\n{'=' * 60}")
    print(f"[done] Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
