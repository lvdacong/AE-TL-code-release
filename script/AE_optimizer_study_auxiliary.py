"""
AE_optimizer_study_auxiliary.py
================================
Systematic optimizer study on DR scenario (N_r=12).
Four experiments to select the optimal fine-tuning configuration.

Uses the real train_model() from AE_train_model_auxiliary (not a custom loop),
so results are directly comparable with all production experiments.

Called via AE_run_optimizer_study.py.

Experiments:
    A: Optimizer type comparison (Adam / AdamW / SGD+momentum) at difficulty=12
    B: Warmup duration sweep (0 / 10 / 30 / 50 / 100) with Adam+Cosine
    C: Data size effect (50 / 100 / 200 / 400) with Adam+Warmup50+Cosine
    D: Final shootout at difficulty=0 AND difficulty=12
"""

from __future__ import annotations

import copy
import csv
import os
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from AE_train_model_auxiliary import (
    Autoencoder, train_model, load_data_from_path,
    FIG_DPI, PLOT_STYLE, apply_style,
)

# ========================================
# Configuration
# ========================================

AD_OUTPUT = os.path.join(SCRIPT_DIR, "AD_preprocess_datasets_output")
AE_OUTPUT = os.path.join(SCRIPT_DIR, "AE_model_train_and_detect_output")
PRETRAIN_PTH = os.path.join(AE_OUTPUT, "Damage_Repaired", "pretrain", "autoencoder.pth")
OUTPUT_DIR = os.path.join(AE_OUTPUT, "Damage_Repaired", "optimizer_study")

# Data paths
DATA_DIFF0 = os.path.join(AD_OUTPUT, "health_original_2000", "preprocessed_data_raw.npz")
DATA_DIFF12 = os.path.join(AD_OUTPUT, "damage_repaired_12_original_500", "preprocessed_data_raw.npz")

AE_CONFIG = {
    "encoder_dims": [768, 384, 192],
    "latent_dim": 192,
    "decoder_dims": [192, 384, 768],
    "dropout": 0.0,
    "activation": "relu",
}

DEFAULT_EPOCHS = 500
DEFAULT_LR = 3e-4
DEFAULT_BATCH_SIZE = 256
DEFAULT_N_TRAIN = 400
VAL_SAMPLES = 200

# Morandi color palette
COLORS = [
    "#8EAABD",  # blue-gray
    "#CB8E8E",  # dusty rose
    "#A8B5A2",  # sage green
    "#C4A882",  # warm sand
    "#9B8EC4",  # lavender
    "#D4A9A9",  # pink
    "#7FA3B0",  # steel blue
]


# ========================================
# Utilities
# ========================================

def _load_data(npz_path: str, n_train: int = DEFAULT_N_TRAIN) -> Tuple[np.ndarray, np.ndarray]:
    """Load data, split into train/val."""
    data = np.load(npz_path)
    V = data["V"].astype(np.float32)
    N = V.shape[0]
    V_val = V[N - VAL_SAMPLES:]
    V_train = V[:N - VAL_SAMPLES]
    if n_train < V_train.shape[0]:
        V_train = V_train[:n_train]
    return V_train, V_val


def _compute_pretrain_val(npz_path: str) -> float:
    """Compute pretrain model's val loss on given data."""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, V_val = _load_data(npz_path, n_train=1)
    model = Autoencoder(252, **{k: AE_CONFIG[k] for k in
        ["encoder_dims", "latent_dim", "decoder_dims", "dropout", "activation"]}).to(device)
    checkpoint = torch.load(PRETRAIN_PTH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    X_val = torch.from_numpy(V_val).to(device)
    with torch.no_grad():
        recon = model(X_val)
        loss = torch.nn.MSELoss()(recon, X_val).item()
    return loss


def _base_training_config(**overrides) -> dict:
    """Build training_config with defaults, applying overrides."""
    cfg = {
        "finetune": {
            "epochs": DEFAULT_EPOCHS,
            "lr": DEFAULT_LR,
            "batch_size": DEFAULT_BATCH_SIZE,
        },
        "val_samples": VAL_SAMPLES,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "use_lr_scheduler": True,
        "lr_scheduler_type": "cosine",
        "cosine_eta_min": 1e-6,
        "shuffle": True,
        "optimizer_type": "adam",
        "warmup_epochs": 0,
    }
    cfg.update(overrides)
    return cfg


def _run_one(label: str, data_path: str, training_config: dict,
             n_train: int = DEFAULT_N_TRAIN, run_dir: str = "") -> dict:
    """Run one training experiment. Returns result dict."""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    V_train, V_val = _load_data(data_path, n_train=n_train)

    os.makedirs(run_dir, exist_ok=True)
    ft = training_config["finetune"]
    model, train_losses, val_losses = train_model(
        V_train, V_val, device,
        epochs=ft["epochs"],
        lr=ft["lr"],
        batch_size=ft["batch_size"],
        output_dir=run_dir,
        ae_config=AE_CONFIG,
        training_config=training_config,
        plot_style=PLOT_STYLE,
        fig_dpi=FIG_DPI,
        pretrain_model_path=PRETRAIN_PTH,
        model_name=f"optimizer_study_{label}",
        max_train_samples=n_train,
        train_shuffle=training_config.get("shuffle", True),
    )

    return {
        "label": label,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "first_val": val_losses[0],
        "min_val": min(val_losses),
        "final_val": val_losses[-1],
        "best_epoch": int(np.argmin(val_losses)),
    }


def _plot_comparison(results: List[dict], title_suffix: str, output_path: str,
                     pretrain_val: float = None):
    """Plot val_loss curves for multiple configs."""
    apply_style()
    # figsize 高度 3.6 预留顶部外部图例空间（原 3.0 会压扁 Axes 到小于 ylabel）
    fig, ax = plt.subplots(figsize=(10, 3.6), constrained_layout=True)

    for i, res in enumerate(results):
        color = COLORS[i % len(COLORS)]
        ax.plot(range(1, len(res["val_losses"]) + 1), res["val_losses"],
                color=color, linewidth=1.2, label=res["label"])

    if pretrain_val is not None:
        ax.axhline(y=pretrain_val, color="#666666", linestyle="--", linewidth=0.8,
                   label="Pretrain baseline")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss (MSE)")
    ax.set_yscale("log")

    # 图框外部上方横排图例
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="outside upper center",
               ncol=min(len(labels), 4),
               frameon=False, fontsize=12)
    fig.savefig(output_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"[Saved] {os.path.basename(output_path)}")


# ========================================
# Experiment A: Optimizer Type Comparison
# ========================================

def experiment_A() -> Tuple[List[dict], float]:
    """Adam vs AdamW vs SGD+momentum, all at difficulty=12, N=400."""
    print("\n" + "=" * 70)
    print("EXPERIMENT A: Optimizer Type Comparison (difficulty=12, N=400)")
    print("=" * 70)

    pretrain_val = _compute_pretrain_val(DATA_DIFF12)
    print(f"Pretrain val_loss on DR-12 data: {pretrain_val:.6f}")

    configs = [
        ("Adam + Cosine", {"optimizer_type": "adam", "warmup_epochs": 0}),
        ("AdamW + Cosine", {"optimizer_type": "adamw", "warmup_epochs": 0}),
        ("SGD(m=0.9) + Cosine", {"optimizer_type": "sgd", "warmup_epochs": 0, "sgd_momentum": 0.9}),
    ]

    results = []
    for label, overrides in configs:
        print(f"\n--- {label} ---")
        cfg = _base_training_config(**overrides)
        run_dir = os.path.join(OUTPUT_DIR, "expA", label.replace(" ", "_").replace("(", "").replace(")", "").replace("=", ""))
        res = _run_one(label, DATA_DIFF12, cfg, run_dir=run_dir)
        spike = res["first_val"] / pretrain_val
        res["spike_ratio"] = spike
        results.append(res)
        print(f"  spike={spike:.1f}x  min={res['min_val']:.6f}  final={res['final_val']:.6f}")

    _plot_comparison(results, "Optimizer Type", os.path.join(OUTPUT_DIR, "fig_expA_optimizer_type.png"), pretrain_val)

    return results, pretrain_val


# ========================================
# Experiment B: Warmup Duration Sweep
# ========================================

def experiment_B() -> Tuple[List[dict], float]:
    """Adam + Warmup(0/10/30/50/100) + Cosine, difficulty=12, N=400."""
    print("\n" + "=" * 70)
    print("EXPERIMENT B: Warmup Duration Sweep (difficulty=12, N=400)")
    print("=" * 70)

    pretrain_val = _compute_pretrain_val(DATA_DIFF12)
    warmup_values = [0, 10, 30, 50, 100]

    results = []
    for wu in warmup_values:
        label = f"warmup={wu}"
        print(f"\n--- {label} ---")
        cfg = _base_training_config(optimizer_type="adam", warmup_epochs=wu)
        run_dir = os.path.join(OUTPUT_DIR, "expB", f"warmup_{wu}")
        res = _run_one(label, DATA_DIFF12, cfg, run_dir=run_dir)
        spike = res["first_val"] / pretrain_val
        res["spike_ratio"] = spike
        results.append(res)
        print(f"  spike={spike:.1f}x  min={res['min_val']:.6f}  final={res['final_val']:.6f}")

    _plot_comparison(results, "Warmup Duration", os.path.join(OUTPUT_DIR, "fig_expB_warmup_sweep.png"), pretrain_val)

    return results, pretrain_val


# ========================================
# Experiment C: Data Size Effect
# ========================================

def experiment_C() -> Tuple[List[dict], float]:
    """Adam + Warmup50 + Cosine, difficulty=12, N=50/100/200/400."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C: Data Size Effect with Warmup (difficulty=12)")
    print("=" * 70)

    pretrain_val = _compute_pretrain_val(DATA_DIFF12)
    data_sizes = [50, 100, 200, 400]

    results = []
    for n in data_sizes:
        label = f"N={n}"
        print(f"\n--- {label} ---")
        cfg = _base_training_config(optimizer_type="adam", warmup_epochs=50)
        run_dir = os.path.join(OUTPUT_DIR, "expC", f"n_{n}")
        res = _run_one(label, DATA_DIFF12, cfg, n_train=n, run_dir=run_dir)
        spike = res["first_val"] / pretrain_val
        res["spike_ratio"] = spike
        results.append(res)
        print(f"  spike={spike:.1f}x  min={res['min_val']:.6f}  final={res['final_val']:.6f}")

    _plot_comparison(results, "Data Size", os.path.join(OUTPUT_DIR, "fig_expC_datasize.png"), pretrain_val)

    return results, pretrain_val


# ========================================
# Experiment D: Final Shootout (diff=0 + diff=12)
# ========================================

def experiment_D() -> Tuple[List[dict], float, float]:
    """Three configs at BOTH difficulty=0 and difficulty=12."""
    print("\n" + "=" * 70)
    print("EXPERIMENT D: Final Shootout (difficulty=0 AND difficulty=12)")
    print("=" * 70)

    pretrain_val_d0 = _compute_pretrain_val(DATA_DIFF0)
    pretrain_val_d12 = _compute_pretrain_val(DATA_DIFF12)
    print(f"Pretrain val_loss: diff=0 → {pretrain_val_d0:.6f}, diff=12 → {pretrain_val_d12:.6f}")

    configs = [
        ("Adam + Cosine", {"optimizer_type": "adam", "warmup_epochs": 0}),
        ("Adam + Warmup50 + Cosine", {"optimizer_type": "adam", "warmup_epochs": 50}),
        ("SGD(m=0.9) + Cosine", {"optimizer_type": "sgd", "warmup_epochs": 0, "sgd_momentum": 0.9}),
    ]

    results_d0 = []
    results_d12 = []

    for label, overrides in configs:
        # difficulty=0
        print(f"\n--- {label} @ difficulty=0 ---")
        cfg = _base_training_config(**overrides)
        run_dir = os.path.join(OUTPUT_DIR, "expD", "diff0",
                               label.replace(" ", "_").replace("(", "").replace(")", "").replace("=", ""))
        res = _run_one(f"{label} [d=0]", DATA_DIFF0, cfg, run_dir=run_dir)
        res["spike_ratio"] = res["first_val"] / pretrain_val_d0
        results_d0.append(res)
        print(f"  spike={res['spike_ratio']:.1f}x  min={res['min_val']:.6f}  final={res['final_val']:.6f}")

        # difficulty=12
        print(f"\n--- {label} @ difficulty=12 ---")
        cfg = _base_training_config(**overrides)
        run_dir = os.path.join(OUTPUT_DIR, "expD", "diff12",
                               label.replace(" ", "_").replace("(", "").replace(")", "").replace("=", ""))
        res = _run_one(f"{label} [d=12]", DATA_DIFF12, cfg, run_dir=run_dir)
        res["spike_ratio"] = res["first_val"] / pretrain_val_d12
        results_d12.append(res)
        print(f"  spike={res['spike_ratio']:.1f}x  min={res['min_val']:.6f}  final={res['final_val']:.6f}")

    # Two-panel plot
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4.2),
                                   constrained_layout=True)

    for i, (label, _) in enumerate(configs):
        color = COLORS[i]
        ax1.plot(range(1, len(results_d0[i]["val_losses"]) + 1), results_d0[i]["val_losses"],
                 color=color, linewidth=1.2, label=label)
        ax2.plot(range(1, len(results_d12[i]["val_losses"]) + 1), results_d12[i]["val_losses"],
                 color=color, linewidth=1.2, label=label)

    ax1.axhline(y=pretrain_val_d0, color="#666666", linestyle="--", linewidth=0.8,
                label="Pretrain")
    ax2.axhline(y=pretrain_val_d12, color="#666666", linestyle="--", linewidth=0.8)

    for ax, title in [(ax1, "$N_r = 0$"), (ax2, "$N_r = 12$")]:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Loss (MSE)")
        ax.set_yscale("log")
        ax.text(0.02, 0.02, title, transform=ax.transAxes, fontsize=14,
                verticalalignment='bottom', fontstyle='italic')

    # 图例从 ax1 抽取放整图外部上方横排
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="outside upper center",
               ncol=len(labels), frameon=False, fontsize=12)

    fig.savefig(os.path.join(OUTPUT_DIR, "fig_expD_shootout.png"), dpi=FIG_DPI)
    plt.close(fig)
    print(f"[Saved] fig_expD_shootout.png")

    return results_d0, results_d12, pretrain_val_d0, pretrain_val_d12


# ========================================
# Summary
# ========================================

def generate_summary(all_results: dict):
    """Generate summary CSV and print table."""
    csv_path = os.path.join(OUTPUT_DIR, "optimizer_study_summary.csv")
    rows = []

    for exp_name, (results, *baselines) in all_results.items():
        pretrain_val = baselines[0] if baselines else None
        for res in results:
            rows.append({
                "experiment": exp_name,
                "config": res["label"],
                "pretrain_val_loss": f"{pretrain_val:.6f}" if pretrain_val else "",
                "first_epoch_val_loss": f"{res['first_val']:.6f}",
                "spike_ratio": f"{res.get('spike_ratio', 0):.1f}",
                "min_val_loss": f"{res['min_val']:.6f}",
                "final_val_loss": f"{res['final_val']:.6f}",
                "best_epoch": res["best_epoch"],
            })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[Saved] {csv_path}")

    # Print summary table
    print("\n" + "=" * 100)
    print("OPTIMIZER STUDY SUMMARY")
    print("=" * 100)
    print(f"{'Experiment':<12} {'Config':<35} {'Spike':>6} {'Min Val':>10} {'Final Val':>10} {'Best Ep':>8}")
    print("-" * 100)
    for row in rows:
        print(f"{row['experiment']:<12} {row['config']:<35} {row['spike_ratio']:>6}x "
              f"{row['min_val_loss']:>10} {row['final_val_loss']:>10} {row['best_epoch']:>8}")


# ========================================
# Main
# ========================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("OPTIMIZER SELECTION STUDY FOR AE-TL FINE-TUNING")
    print(f"Pretrain model: {PRETRAIN_PTH}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)

    all_results = {}

    # Experiment A
    results_A, baseline_A = experiment_A()
    all_results["ExpA"] = (results_A, baseline_A)

    # Experiment B
    results_B, baseline_B = experiment_B()
    all_results["ExpB"] = (results_B, baseline_B)

    # Experiment C
    results_C, baseline_C = experiment_C()
    all_results["ExpC"] = (results_C, baseline_C)

    # Experiment D
    results_d0, results_d12, bv_d0, bv_d12 = experiment_D()
    all_results["ExpD_d0"] = (results_d0, bv_d0)
    all_results["ExpD_d12"] = (results_d12, bv_d12)

    # Summary
    generate_summary(all_results)

    # Final recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    # Find best config at difficulty=0 (must match pretrain)
    print(f"\nDifficulty=0 (pretrain baseline = {bv_d0:.6f}):")
    for res in results_d0:
        ratio = res["final_val"] / bv_d0
        status = "PASS" if ratio < 1.05 else "FAIL"
        print(f"  {res['label']:<40} final={res['final_val']:.6f}  ratio={ratio:.3f}  [{status}]")

    print(f"\nDifficulty=12 (pretrain baseline = {bv_d12:.6f}):")
    for res in results_d12:
        print(f"  {res['label']:<40} final={res['final_val']:.6f}  min={res['min_val']:.6f}")

    print("\nStudy complete!")
