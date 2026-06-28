"""
AE_gen_scenario_figures.py
===========================
Generate 3 scenario-unique figures: DR channel bias, SO offset
profile, and SD drift convergence.

Usage:
    cd script && python AE_gen_scenario_figures.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from AE_train_model_auxiliary import Autoencoder, apply_style, PLOT_STYLE, FIG_DPI

# ── paths ───────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_OUT     = os.path.join(SCRIPT_DIR, "AE_model_train_and_detect_output")
BASE_PRE     = os.path.join(SCRIPT_DIR, "AD_preprocess_datasets_output")
BASE_FIGURE_OUT = os.path.join(SCRIPT_DIR, "AE_model_train_and_detect_output")
PRETRAIN_PTH = os.path.join(BASE_OUT, "Damage_Repaired", "pretrain", "autoencoder.pth")

# ── model config ────────────────────────────────────────────────────────
AE_CONFIG = dict(
    encoder_dims=[768, 384, 192],
    latent_dim=192,
    decoder_dims=[192, 384, 768],
    dropout=0.0,
    activation="relu",
)
VAL_SAMPLES = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Morandi palette ─────────────────────────────────────────────────────
MORANDI = {
    "muted_red":    "#C97A6C",
    "muted_blue":   "#7BA7BC",
    "muted_green":  "#8FAE7E",
    "muted_purple": "#A492B0",
    "muted_gold":   "#C8B870",
    "muted_gray":   "#999999",
}


# ── helpers ─────────────────────────────────────────────────────────────
def load_ae(pth: str, D: int) -> Autoencoder:
    m = Autoencoder(
        D,
        AE_CONFIG["encoder_dims"],
        AE_CONFIG["latent_dim"],
        AE_CONFIG["decoder_dims"],
        AE_CONFIG["dropout"],
        AE_CONFIG["activation"],
    ).to(device)
    m.load_state_dict(torch.load(pth, map_location=device))
    m.eval()
    return m


def load_npz_last(folder: str, n: int) -> np.ndarray:
    path = os.path.join(BASE_PRE, folder, "preprocessed_data_raw.npz")
    V = np.load(path)["V"].astype(np.float32)
    return V[-n:]


def channel_mean_residual(model: Autoencoder, data: np.ndarray) -> np.ndarray:
    """Return per-channel mean |residual| (shape [D])."""
    with torch.no_grad():
        x = torch.tensor(data, dtype=torch.float32, device=device)
        x_hat = model(x)
        residuals = (x_hat - x).abs().cpu().numpy()
    return residuals.mean(axis=0)


# =====================================================================
# Figure 1: Damage_Repaired -- Channel-wise Residual Bias Bar Chart
# =====================================================================
def fig_dr_channel_bias():
    print("[Fig 1] Damage_Repaired channel-wise residual bias ...")

    ctrl = load_npz_last("damage_repaired_12_original_500", VAL_SAMPLES)
    D = ctrl.shape[1]

    old_model = load_ae(PRETRAIN_PTH, D)
    tl_model  = load_ae(
        os.path.join(BASE_OUT, "Damage_Repaired", "tl_comparison", "TL", "autoencoder.pth"), D
    )

    res_old = channel_mean_residual(old_model, ctrl)
    res_tl  = channel_mean_residual(tl_model, ctrl)

    channels = np.arange(D)
    bar_w = 0.4

    apply_style(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(12, 3.8), constrained_layout=True)

    ax.bar(channels - bar_w / 2, res_old, bar_w,
           label="OldModel", color=MORANDI["muted_red"], edgecolor="none")
    ax.bar(channels + bar_w / 2, res_tl, bar_w,
           label="TL", color=MORANDI["muted_blue"], edgecolor="none")

    ax.set_xlabel("Channel Index")
    ax.set_ylabel("Mean |Residual|")
    ax.set_xlim(-1, D)
    # Cap y-axis to show detail; the single OldModel spike (~0.35) is annotated
    cap = np.percentile(np.concatenate([res_old, res_tl]), 99) * 1.3
    peak_val = res_old.max()
    peak_ch = res_old.argmax()
    if peak_val > cap:
        ax.set_ylim(0, cap)
        ax.annotate(f"{peak_val:.2f}", xy=(peak_ch, cap * 0.95),
                    fontsize=12, ha="center", color=MORANDI["muted_red"])
    ax.tick_params(direction="in")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="outside upper center",
               ncol=len(labels), frameon=False, fontsize=12)

    out = os.path.join(BASE_FIGURE_OUT, "Damage_Repaired", "fig_dr_channel_bias.png")
    fig.savefig(out, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  Saved: {out}")


# =====================================================================
# Figure 2: Sensor_Offset -- Per-channel Residual Profile Line Plot
# =====================================================================
def fig_so_offset_profile():
    """Per-channel residual profile under the max-stress-difference offset scheme.
    Compares OldModel on N_o=0 vs N_o=1, and TL on N_o=1.
    """
    print("[Fig 2] Sensor_Offset per-channel residual profile ...")

    ctrl_so0 = load_npz_last("health_offset_count_0_2000", VAL_SAMPLES)
    ctrl_so1 = load_npz_last("health_offset_count_1_2000", VAL_SAMPLES)
    D = ctrl_so0.shape[1]

    old_model = load_ae(PRETRAIN_PTH, D)
    tl_so1    = load_ae(
        os.path.join(BASE_OUT, "Sensor_Offset", "tl_comparison", "TL", "autoencoder.pth"), D
    )

    lines_cfg = [
        ("OldModel on $N_o=0$", old_model, ctrl_so0, MORANDI["muted_blue"],  "-",  1.2),
        ("OldModel on $N_o=1$", old_model, ctrl_so1, MORANDI["muted_red"],   "-",  1.4),
        ("TL on $N_o=1$",       tl_so1,    ctrl_so1, MORANDI["muted_green"], "-",  1.0),
    ]

    channels = np.arange(D)

    apply_style(PLOT_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2),
                             gridspec_kw={"width_ratios": [1, 3]},
                             constrained_layout=True)

    # Collect residuals
    all_res = {}
    for label, model, data, color, ls, lw in lines_cfg:
        all_res[label] = channel_mean_residual(model, data)

    # Auto-locate the affected channel (largest OldModel residual under N_o=1)
    diff = all_res["OldModel on $N_o=1$"] - all_res["OldModel on $N_o=0$"]
    affected_ch = int(np.argmax(diff))
    zoom_lo = max(0, affected_ch - 5)
    zoom_hi = min(D, affected_ch + 6)

    # Left panel: zoomed on the affected channel
    for label, _, _, color, ls, lw in lines_cfg:
        axes[0].plot(channels[zoom_lo:zoom_hi], all_res[label][zoom_lo:zoom_hi],
                     label=label, color=color, linestyle=ls, linewidth=lw,
                     marker='o', markersize=4)
    axes[0].set_xlabel("Channel Index")
    axes[0].set_ylabel("Mean |Residual|")
    axes[0].set_xlim(zoom_lo - 0.5, zoom_hi - 0.5)
    axes[0].tick_params(direction="in")
    axes[0].axvline(affected_ch, color="#999999", linestyle="--", linewidth=0.8, alpha=0.6)

    # Right panel: full range
    for label, _, _, color, ls, lw in lines_cfg:
        axes[1].plot(channels, all_res[label], label=label,
                     color=color, linestyle=ls, linewidth=lw, alpha=0.85)
    axes[1].set_xlabel("Channel Index")
    axes[1].set_ylabel("Mean |Residual|")
    axes[1].set_xlim(0, D - 1)
    axes[1].tick_params(direction="in")

    # Subplot labels
    axes[0].text(0.03, 0.95, "(a)", transform=axes[0].transAxes, fontsize=18, va="top")
    axes[1].text(0.03, 0.95, "(b)", transform=axes[1].transAxes, fontsize=18, va="top")

    # 图例从 axes[0] 抽取，放整图外部上方横排
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="outside upper center",
               ncol=len(labels), frameon=False, fontsize=12)

    out = os.path.join(BASE_FIGURE_OUT, "Sensor_Offset", "fig_so_offset_profile.png")
    fig.savefig(out, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  Saved: {out}")


# =====================================================================
# Figure 3: Sensor_Drift -- Drift Ratio Convergence Line Plot
# =====================================================================
def fig_sd_drift_convergence():
    print("[Fig 3] Sensor_Drift drift-ratio convergence ...")

    csv_path = os.path.join(
        BASE_OUT, "Sensor_Drift", "difficulty_ablation",
        "difficulty_ablation_Sensor_Drift.csv"
    )
    df = pd.read_csv(csv_path)

    # Extract sorted unique drift ratios and data sizes
    drift_ratios = sorted(df["difficulty"].unique())
    data_sizes = sorted(df["data_size"].unique())
    colors = [
        MORANDI["muted_blue"],
        MORANDI["muted_green"],
        MORANDI["muted_purple"],
        MORANDI["muted_gold"],
    ]

    apply_style(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(10, 3.6), constrained_layout=True)

    # Before TL reference (one value per drift ratio)
    before_tl = df.groupby("difficulty")["before_tl"].first().sort_index().values
    ax.plot(drift_ratios, before_tl, label="Before TL", color=MORANDI["muted_gray"],
            linestyle="--", linewidth=1.2, marker="s", markersize=5)

    for n, color in zip(data_sizes, colors):
        subset = df[df["data_size"] == n].sort_values("difficulty")
        ax.plot(subset["difficulty"].values, subset["val_loss"].values,
                label=f"N = {n}", color=color,
                linestyle="-", linewidth=1.2, marker="o", markersize=5)

    ax.set_yscale("log")
    ax.set_xlabel("Drift Ratio")
    ax.set_ylabel("Validation Loss")
    ax.set_xticks(drift_ratios)
    ax.tick_params(direction="in")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="outside upper center",
               ncol=min(len(labels), 5), frameon=False, fontsize=12)

    out = os.path.join(BASE_FIGURE_OUT, "Sensor_Drift", "fig_sd_drift_convergence.png")
    fig.savefig(out, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── main ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Output directories are already created by the experiment scripts
    fig_dr_channel_bias()
    fig_so_offset_profile()
    fig_sd_drift_convergence()
    print("\nAll 3 figures generated.")
