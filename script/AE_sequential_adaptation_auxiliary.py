"""
AE_sequential_adaptation_auxiliary.py
=====================================
Sequential multi-domain adaptation: fine-tune a pretrained AE through
consecutive domain shifts (structural repair → sensor relocation → sensor drift).

Outputs (to AE_model_train_and_detect_output/Sequential_Adaptation/):
  - fig_sequential_all_orders_overlay.png  : All 6 orderings overlaid (equal-width phases)
  - fig_sequential_heatmap.png             : Cross-domain reconstruction error matrix
  - sequential_results.csv / .json         : Full numerical results
"""

from __future__ import annotations

import os
import sys
import copy
import itertools
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from AE_train_model_auxiliary import Autoencoder
from AE_model_train_and_detect_auxiliary import (
    compute_anomaly_scores,
    compute_detection_metrics,
)

DATA_ROOT = os.path.join(SCRIPT_DIR, "AD_preprocess_datasets_output")
PRETRAIN_PATH = os.path.join(
    SCRIPT_DIR,
    "AE_model_train_and_detect_output/Damage_Repaired/pretrain/autoencoder.pth",
)
OUTPUT_DIR = os.path.join(
    SCRIPT_DIR, "AE_model_train_and_detect_output", "Sequential_Adaptation"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── model config ───────────────────────────────────────────────────
AE_CONFIG = dict(
    input_dim=252,
    encoder_dims=[768, 384, 192],
    latent_dim=192,
    decoder_dims=[192, 384, 768],
    dropout=0.0,
    activation="relu",
)

# ── training hyper-params (match isolated TL config from §5.3) ─────
EPOCHS_PER_STAGE = 200
BATCH_SIZE = 256
LR = 1e-3           # decoder learning rate
ENCODER_LR = 1e-5   # encoder learning rate (differential)
NOISE_STD = 0.005    # Gaussian input noise std
GRAD_CLIP = 1.0
VAL_SAMPLES = 200        # for Source/SO/SD (2000-sample datasets)
VAL_SAMPLES_DR = 100     # DR has only 500 samples => 400 train + 100 val
EVAL_INTERVAL = 5        # cross-domain eval every N epochs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── domain definitions (cumulative: each stage inherits previous shifts) ──
# Physical scenario: repair → offset on repaired hull → drift on repaired+offset hull
# Stage 1 (DR):       repair only
# Stage 2 (DR+SO):    repair + sensor offset
# Stage 3 (DR+SO+SD): repair + sensor offset + temperature drift
DOMAINS = {
    "Source": {
        "healthy": "health_original_2000",
        "damage": "first_damage_original_100",
        "val_samples": VAL_SAMPLES,
        "label": "Source Domain",
        "short": "Src",
    },
    "DR": {
        "healthy": "damage_repaired_12_original_500",
        "damage": "second_damage_12_original_100",
        "val_samples": VAL_SAMPLES_DR,
        "label": "Repair",
        "short": "DR",
    },
    "SO": {
        "healthy": "damage_repaired_12_offset_count_1_500",
        "damage": "second_damage_12_offset_count_1_100",
        "val_samples": VAL_SAMPLES_DR,
        "label": "Repair + Offset",
        "short": "DR+SO",
    },
    "SD": {
        "healthy": "damage_repaired_12_offset_count_1_drift_nonuniform_500",
        "damage": "second_damage_12_offset_count_1_drift_nonuniform_100",
        "val_samples": VAL_SAMPLES_DR,
        "label": "Repair + Offset + Drift",
        "short": "DR+SO+SD",
    },
}

# adapt-able domains (the 3 shift scenarios, excluding Source)
SHIFT_KEYS = ["DR", "SO", "SD"]

# ── Morandi palette ────────────────────────────────────────────────
COLORS = {
    "Source": "#8B8B8B",   # neutral gray
    "DR":     "#6B8EAE",   # steel blue
    "SO":     "#8EAE8B",   # sage green
    "SD":     "#C49A6C",   # warm sand
}
ORDER_COLORS = [
    "#6B8EAE", "#C49A6C", "#8EAE8B",
    "#B07AA1", "#CB8E8E", "#9B8EC4",
]

PLOT_STYLE = {
    "font.family": "Times New Roman",
    "font.size": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 12,
    "legend.frameon": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.grid": False,
    "axes.spines.top": True,
    "axes.spines.right": True,
}
FIG_DPI = 300


# ====================================================================
# Data loading
# ====================================================================

def _load_npz(name: str) -> np.ndarray:
    path = os.path.join(DATA_ROOT, name, "preprocessed_data_raw.npz")
    return np.load(path)["V"].astype(np.float32)


def load_domain_data():
    """Load train / val / damage arrays for every domain."""
    data = {}
    for key, cfg in DOMAINS.items():
        V = _load_npz(cfg["healthy"])
        vs = cfg["val_samples"]
        V_train = V[: len(V) - vs]
        V_val = V[len(V) - vs :]
        V_damage = _load_npz(cfg["damage"])
        data[key] = {"train": V_train, "val": V_val, "damage": V_damage}
        print(f"[{key:6s}] train {V_train.shape}, val {V_val.shape}, damage {V_damage.shape}")
    return data


# ====================================================================
# Model helpers
# ====================================================================

def _make_model():
    return Autoencoder(
        AE_CONFIG["input_dim"],
        AE_CONFIG["encoder_dims"],
        AE_CONFIG["latent_dim"],
        AE_CONFIG["decoder_dims"],
        AE_CONFIG["dropout"],
        AE_CONFIG["activation"],
    )


def load_pretrained():
    model = _make_model().to(DEVICE)
    ckpt = torch.load(PRETRAIN_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt)
    return model


def compute_val_loss(model, V_val):
    model.eval()
    X = torch.from_numpy(V_val).to(DEVICE)
    loader = DataLoader(TensorDataset(X), batch_size=BATCH_SIZE, shuffle=False)
    crit = nn.MSELoss()
    total, cnt = 0.0, 0
    with torch.no_grad():
        for (xb,) in loader:
            total += crit(model(xb), xb).item() * xb.size(0)
            cnt += xb.size(0)
    return total / max(cnt, 1)


# ====================================================================
# Sequential training with cross-domain monitoring
# ====================================================================

def run_sequential(
    sequence: list[str],
    domain_data: dict,
    epochs_per_stage: int = EPOCHS_PER_STAGE,
    eval_interval: int = EVAL_INTERVAL,
    label: str = "",
    warmup_epochs: int = 0,
) -> dict:
    """
    Run sequential fine-tuning through *sequence* of domain keys.

    Returns dict with:
      'history':  list of dicts, one per eval point
                  {epoch_global, stage, domain_losses: {key: loss}}
      'stage_boundaries': list of global-epoch ints
      'final_model': state_dict
      'detection':  {domain_key: metrics_dict} after all stages
    """
    print(f"\n{'='*70}")
    print(f"Sequential Adaptation: {' -> '.join(sequence)}  {label}")
    print(f"{'='*70}")

    model = load_pretrained()
    history = []
    train_loss_history = []          # per-epoch training MSE (for continuous curve)
    stage_boundaries = [0]
    global_epoch = 0

    # --- evaluate pretrained model on all domains (epoch 0) ---
    losses_0 = {k: compute_val_loss(model, domain_data[k]["val"]) for k in DOMAINS}
    history.append({"epoch": 0, "stage": "pretrain", "losses": losses_0})
    print(f"[pretrain] " + "  ".join(f"{k}={v:.5f}" for k, v in losses_0.items()))

    # --- sequential stages ---
    for stage_idx, domain_key in enumerate(sequence):
        print(f"\n--- Stage {stage_idx+1}: Fine-tune on {domain_key} ({DOMAINS[domain_key]['label']}) ---")
        V_train = domain_data[domain_key]["train"]
        # Cap to 400 samples for consistency
        if V_train.shape[0] > 400:
            V_train = V_train[:400]

        X_train = torch.from_numpy(V_train).to(DEVICE)
        train_loader = DataLoader(
            TensorDataset(X_train), batch_size=BATCH_SIZE, shuffle=True
        )
        param_groups = [
            {"params": list(model.encoder.parameters()), "lr": ENCODER_LR},
            {"params": list(model.decoder.parameters()), "lr": LR},
        ]
        optimizer = Adam(param_groups, lr=LR)
        criterion = nn.MSELoss()

        # CosineAnnealing per stage (matching single-stage fine-tuning config)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs_per_stage, eta_min=1e-6)

        for ep in range(1, epochs_per_stage + 1):
            # -- train one epoch --
            model.train()
            run_loss, run_cnt = 0.0, 0
            for (xb,) in train_loader:
                optimizer.zero_grad(set_to_none=True)
                xb_input = xb + torch.randn_like(xb) * NOISE_STD
                recon = model(xb_input)
                loss = criterion(recon, xb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                run_loss += loss.item() * xb.size(0)
                run_cnt += xb.size(0)

            scheduler.step()

            global_epoch += 1
            train_loss_history.append({
                "epoch": global_epoch,
                "stage": domain_key,
                "train_loss": run_loss / max(run_cnt, 1),
            })

            # -- cross-domain eval --
            if ep % eval_interval == 0 or ep == 1 or ep == epochs_per_stage:
                losses = {k: compute_val_loss(model, domain_data[k]["val"]) for k in DOMAINS}
                history.append({"epoch": global_epoch, "stage": domain_key, "losses": losses})
                if ep % 50 == 0 or ep == 1:
                    info = "  ".join(f"{k}={v:.5f}" for k, v in losses.items())
                    print(f"  Epoch {ep:4d} (global {global_epoch}) | {info}")

        stage_boundaries.append(global_epoch)

    # --- detection metrics on all domains after full sequential adaptation ---
    print("\n--- Detection metrics after full sequential adaptation ---")
    detection = {}
    for key in DOMAINS:
        V_ctrl = domain_data[key]["val"]
        V_dmg = domain_data[key]["damage"]
        scores_ctrl = compute_anomaly_scores(V_ctrl, model, DEVICE, method="mean")
        scores_dmg = compute_anomaly_scores(V_dmg, model, DEVICE, method="mean")
        metrics = compute_detection_metrics(scores_dmg, scores_ctrl)
        detection[key] = metrics
        print(f"  {key}: AUC={metrics['auc']:.4f} [{metrics['auc_ci_lo']:.3f}, {metrics['auc_ci_hi']:.3f}]"
              f"  TPR@5%={metrics['tpr_at_fpr']:.2f}  F1={metrics['best_f1']:.4f}")

    # max-channel detection too
    detection_max = {}
    for key in DOMAINS:
        V_ctrl = domain_data[key]["val"]
        V_dmg = domain_data[key]["damage"]
        scores_ctrl = compute_anomaly_scores(V_ctrl, model, DEVICE, method="max")
        scores_dmg = compute_anomaly_scores(V_dmg, model, DEVICE, method="max")
        metrics = compute_detection_metrics(scores_dmg, scores_ctrl)
        detection_max[key] = metrics
        print(f"  {key} (max): AUC={metrics['auc']:.4f}")

    result = {
        "history": history,
        "train_loss_history": train_loss_history,
        "stage_boundaries": stage_boundaries,
        "sequence": sequence,
        "detection_mean": {k: {kk: vv for kk, vv in v.items() if kk not in ("fprs", "tprs")}
                          for k, v in detection.items()},
        "detection_max": {k: {kk: vv for kk, vv in v.items() if kk not in ("fprs", "tprs")}
                         for k, v in detection_max.items()},
    }

    return result


# ====================================================================
# Figures
# ====================================================================

def apply_style():
    plt.rcParams.update(PLOT_STYLE)



def plot_cross_domain_heatmap(result: dict, output_path: str):
    """
    Heatmap of val loss at key checkpoints:
      rows = evaluation domain, cols = model checkpoint.
    Uses BEST val loss achieved during each stage, not endpoint.
    """
    apply_style()
    history = result["history"]
    sequence = result["sequence"]

    checkpoint_labels = ["Pre-trained", "Stage 1\n(+DR)", "Stage 2\n(+SO)", "Stage 3\n(+SD)"]

    domain_keys = list(DOMAINS.keys())
    domain_labels = [DOMAINS[k]["label"] for k in domain_keys]

    # Build matrix: for each stage, find the best val loss on each domain
    matrix = np.zeros((len(domain_keys), len(checkpoint_labels)))
    h0 = history[0]
    for di, dk in enumerate(domain_keys):
        matrix[di, 0] = h0["losses"][dk]
    for si, stage_key in enumerate(sequence):
        stage_hist = [h for h in history if h["stage"] == stage_key]
        if not stage_hist:
            continue
        best_entry = min(stage_hist, key=lambda h: h["losses"][stage_key])
        for di, dk in enumerate(domain_keys):
            matrix[di, si + 1] = best_entry["losses"][dk]

    log_matrix = np.log10(matrix)
    vmin, vmax = log_matrix.min(), log_matrix.max()

    # ── clean academic heatmap ──
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)

    cmap = LinearSegmentedColormap.from_list(
        "academic_seq",
        ["#F7FCF5", "#C7E9C0", "#74C476", "#238B45", "#00441B"],
    )
    im = ax.imshow(log_matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    # cell borders
    for edge in ["top", "bottom", "left", "right"]:
        ax.spines[edge].set_visible(False)
    ax.set_xticks(np.arange(len(checkpoint_labels) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(domain_keys) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # annotations
    mid = (vmin + vmax) / 2
    for di in range(len(domain_keys)):
        for ci in range(len(checkpoint_labels)):
            val = matrix[di, ci]
            txt_color = "white" if log_matrix[di, ci] > mid else "#333333"
            ax.text(ci, di, f"{val:.4f}", ha="center", va="center",
                    fontsize=13, color=txt_color)

    ax.set_xticks(range(len(checkpoint_labels)))
    ax.set_xticklabels(checkpoint_labels, fontsize=15)
    ax.set_yticks(range(len(domain_labels)))
    ax.set_yticklabels(domain_labels, fontsize=15)
    ax.tick_params(axis="both", length=0)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
    cbar.set_label("$\\log_{10}$(MSE)", fontsize=15)
    cbar.ax.tick_params(labelsize=13)

    fig.savefig(output_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_all_orders_overlay(
    all_results: dict,
    pretrain_csv_path: str,
    output_path: str,
    pretrain_epochs: int = 2000,
) -> None:
    """
    Overlay all 6 ordering permutations on one continuous training-loss plot.
    X-axis uses uniform real epoch scale (pretrain 0-2000, stages 2001-2600).
    """
    import pandas as pd
    import matplotlib.transforms as mtransforms
    from matplotlib.lines import Line2D

    # ── load pretrain training loss ──
    pt_df = pd.read_csv(pretrain_csv_path)
    pt_train = pt_df["train_loss"].values

    # ── real epoch coordinates (uniform scale) ──
    pt_x = np.arange(1, pretrain_epochs + 1)

    # ── high-contrast colours for 6 orderings ──
    HC_COLORS = [
        "#2B6DA1",   # strong blue
        "#CB4335",   # strong red
        "#28874F",   # strong green
        "#8E44AD",   # strong purple
        "#D68910",   # strong amber
        "#16A085",   # strong teal
    ]

    # ── style ── 整体字号放大：label 22 > tick 18 > legend 12
    curve_style = PLOT_STYLE.copy()
    curve_style.update({
        "font.size": 22,
        "axes.labelsize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 12,
    })
    plt.rcParams.update(curve_style)

    # figsize 高度 5.2 预留顶部外部图例空间（原 4.5 + 内部图例遮挡）
    fig, ax = plt.subplots(1, 1, figsize=(12, 5.2), constrained_layout=True)
    lw_pt, lw_seq = 0.9, 0.7

    # ── pretrain (shared) ──
    C_PT = "#C97A6C"
    ax.plot(pt_x, pt_train, color=C_PT, linewidth=lw_pt, linestyle="-")

    # ── 6 sequential curves (shift to real epoch = pretrain_epochs + global_epoch) ──
    for i, (order_label, res) in enumerate(all_results.items()):
        tlh = res["train_loss_history"]
        ep_global = np.array([h["epoch"] for h in tlh])
        losses = np.array([h["train_loss"] for h in tlh])
        sx = pretrain_epochs + ep_global
        ax.plot(sx, losses, color=HC_COLORS[i % len(HC_COLORS)],
                linewidth=lw_seq, linestyle="-", alpha=0.9)

    # ── phase boundary separators ──
    total_seq = 3 * EPOCHS_PER_STAGE
    for s in range(3):
        ax.axvline(pretrain_epochs + s * EPOCHS_PER_STAGE,
                   color="#BBBBBB", linewidth=0.8, linestyle="--")

    # ── phase labels ──
    blend = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(pretrain_epochs / 2, 0.93, "Pre-train",
            ha="center", fontsize=14, color="black", fontstyle="italic",
            transform=blend)
    for si, label in enumerate(["Stage 1", "Stage 2", "Stage 3"]):
        cx = pretrain_epochs + (si + 0.5) * EPOCHS_PER_STAGE
        ax.text(cx, 0.93, label,
                ha="center", fontsize=12, color="black", fontstyle="italic",
                transform=blend)

    # ── axes ──
    total_epochs = pretrain_epochs + total_seq
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss (MSE)")
    ax.set_yscale("log")
    ax.set_xlim(0, total_epochs)
    ax.set_xticks([0, 500, 1000, 1500, 2000,
                   pretrain_epochs + EPOCHS_PER_STAGE,
                   pretrain_epochs + 2 * EPOCHS_PER_STAGE,
                   total_epochs])

    # ── legend: figure-level outside upper center, single row ──
    legend_elements = [
        Line2D([0], [0], color=C_PT, lw=lw_pt, label="Pre-train"),
    ]
    for i, order_label in enumerate(all_results.keys()):
        legend_elements.append(
            Line2D([0], [0], color=HC_COLORS[i % len(HC_COLORS)],
                   lw=lw_seq + 0.3, alpha=0.9, label=order_label)
        )
    fig.legend(handles=legend_elements,
               loc="outside upper center",
               ncol=len(legend_elements),
               frameon=False, fontsize=12,
               columnspacing=1.0, handlelength=1.5)

    fig.savefig(output_path, dpi=FIG_DPI)
    plt.close(fig)
    plt.rcParams.update(PLOT_STYLE)
    print(f"Saved: {output_path}")


# ====================================================================
# Data export
# ====================================================================

def save_results_csv(all_results: dict, output_path: str):
    """Save summary results to CSV."""
    rows = []
    for order_label, res in all_results.items():
        final_losses = res["history"][-1]["losses"]
        det_mean = res["detection_mean"]
        det_max = res["detection_max"]
        for dk in DOMAINS:
            rows.append({
                "ordering": order_label,
                "domain": dk,
                "final_val_loss": final_losses[dk],
                "auc_mean": det_mean[dk]["auc"],
                "auc_mean_ci_lo": det_mean[dk]["auc_ci_lo"],
                "auc_mean_ci_hi": det_mean[dk]["auc_ci_hi"],
                "tpr_at_fpr5_mean": det_mean[dk]["tpr_at_fpr"],
                "f1_mean": det_mean[dk]["best_f1"],
                "auc_max": det_max[dk]["auc"],
            })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {output_path}")


def save_all_json(all_results: dict, output_path: str):
    """Save full results (including history) to JSON for replotting."""
    import json
    serializable = {}
    for label, res in all_results.items():
        serializable[label] = {
            "history": res["history"],
            "train_loss_history": res.get("train_loss_history", []),
            "stage_boundaries": res["stage_boundaries"],
            "sequence": res["sequence"],
            "detection_mean": res["detection_mean"],
            "detection_max": res["detection_max"],
        }
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved: {output_path}")


# ====================================================================
# Main
# ====================================================================

def main():
    print("Loading data...")
    domain_data = load_domain_data()

    # ── Experiment 1: Main sequence DR→SO→SD ──
    main_seq = ["DR", "SO", "SD"]
    main_result = run_sequential(main_seq, domain_data, label="(main)")

    # Plot figures for main sequence
    pretrain_csv = os.path.join(
        SCRIPT_DIR,
        "AE_model_train_and_detect_output/Damage_Repaired/pretrain/training_losses.csv",
    )
    plot_cross_domain_heatmap(
        main_result,
        os.path.join(OUTPUT_DIR, "fig_sequential_heatmap.png"),
    )

    # ── Experiment 2: All 6 orderings ──
    all_results = {}
    all_results["→".join(main_seq)] = main_result

    for perm in itertools.permutations(SHIFT_KEYS):
        label = "→".join(perm)
        if label in all_results:
            continue
        res = run_sequential(list(perm), domain_data, label=f"({label})")
        all_results[label] = res

    plot_all_orders_overlay(
        all_results,
        pretrain_csv,
        os.path.join(OUTPUT_DIR, "fig_sequential_all_orders_overlay.png"),
    )

    # ── Save all numerical results ──
    save_results_csv(
        all_results,
        os.path.join(OUTPUT_DIR, "sequential_results.csv"),
    )
    save_all_json(
        all_results,
        os.path.join(OUTPUT_DIR, "sequential_results.json"),
    )

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY: Sequential Adaptation Study Complete")
    print("=" * 70)
    for order_label, res in all_results.items():
        final = res["history"][-1]["losses"]
        print(f"\n  {order_label}:")
        for dk in DOMAINS:
            det = res["detection_mean"][dk]
            print(f"    {dk}: val_loss={final[dk]:.5f}  AUC={det['auc']:.4f}")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
