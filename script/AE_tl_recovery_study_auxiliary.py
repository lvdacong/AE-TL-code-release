"""
AE_tl_recovery_study_auxiliary.py
==================================
Systematic study on TL reconstruction error recovery.

Problem: After domain shift, TL fine-tuning doesn't recover val_loss to
pretrain baseline (0.001148). Current best: 0.001245 (8.5% gap).
Root cause: Cosine LR decays too fast → model plateaus after epoch ~300.

Phase 1: LR schedule fixes (config-only, no code changes)
Phase 2: Optimizer + schedule combos (minor code changes)
Phase 3: Advanced strategies (data replay, discriminative LR, etc.)

Called via AE_run_tl_recovery_study.py.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from AE_train_model_auxiliary import (
    Autoencoder, train_model,
    FIG_DPI, PLOT_STYLE, apply_style,
)
from AE_channel_ablation_auxiliary import compute_all_channel_mae

# ========================================
# Configuration
# ========================================

AD_OUTPUT = os.path.join(SCRIPT_DIR, "AD_preprocess_datasets_output")
AE_OUTPUT = os.path.join(SCRIPT_DIR, "AE_model_train_and_detect_output")
PRETRAIN_PTH = os.path.join(AE_OUTPUT, "Damage_Repaired", "pretrain", "autoencoder.pth")
OUTPUT_DIR = os.path.join(AE_OUTPUT, "Damage_Repaired", "recovery_study")

DATA_DIFF12 = os.path.join(AD_OUTPUT, "damage_repaired_12_original_500", "preprocessed_data_raw.npz")
DATA_HEALTH = os.path.join(AD_OUTPUT, "health_original_2000", "preprocessed_data_raw.npz")

AE_CONFIG = {
    "encoder_dims": [768, 384, 192],
    "latent_dim": 192,
    "decoder_dims": [192, 384, 768],
    "dropout": 0.0,
    "activation": "relu",
}

VAL_SAMPLES = 200
DEFAULT_N_TRAIN = 400
TARGET_CHANNEL = 123  # Most affected channel in DR scenario

# Morandi palette
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
    V = np.load(npz_path)["V"].astype(np.float32)
    N = V.shape[0]
    V_val = V[N - VAL_SAMPLES:]
    V_train = V[:N - VAL_SAMPLES]
    if n_train < V_train.shape[0]:
        V_train = V_train[:n_train]
    return V_train, V_val


def _compute_baselines(device: torch.device) -> Dict[str, float]:
    """Compute baseline metrics (once per study)."""
    _, V_val_shifted = _load_data(DATA_DIFF12)
    _, V_val_health = _load_data(DATA_HEALTH, n_train=1)

    model = Autoencoder(252, **{k: AE_CONFIG[k] for k in
        ["encoder_dims", "latent_dim", "decoder_dims", "dropout", "activation"]}).to(device)
    model.load_state_dict(torch.load(PRETRAIN_PTH, map_location=device, weights_only=True))
    model.eval()

    # Pretrain val_loss on health data (gold standard)
    X_h = torch.from_numpy(V_val_health).to(device)
    with torch.no_grad():
        pretrain_health = torch.nn.MSELoss()(model(X_h), X_h).item()

    # Pretrain val_loss on shifted data (before-TL)
    X_s = torch.from_numpy(V_val_shifted).to(device)
    with torch.no_grad():
        pretrain_shifted = torch.nn.MSELoss()(model(X_s), X_s).item()

    # Per-channel MAE baselines
    ch_mae_health = compute_all_channel_mae(V_val_health, model, device)
    ch_mae_shifted = compute_all_channel_mae(V_val_shifted, model, device)

    return {
        "pretrain_health_val": pretrain_health,
        "pretrain_shifted_val": pretrain_shifted,
        "ch_target_preshift_mae": float(ch_mae_health[TARGET_CHANNEL]),
        "ch_target_postshift_mae": float(ch_mae_shifted[TARGET_CHANNEL]),
    }


def _base_training_config(**overrides) -> dict:
    """Build training_config with defaults, applying overrides."""
    cfg = {
        "finetune": {"epochs": 500, "lr": 3e-4, "batch_size": 256},
        "val_samples": VAL_SAMPLES,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "use_lr_scheduler": True,
        "lr_scheduler_type": "cosine",
        "cosine_eta_min": 1e-6,
        "shuffle": True,
        "optimizer_type": "sgd",
        "sgd_momentum": 0.9,
        "warmup_epochs": 0,
    }
    cfg.update(overrides)
    return cfg


def _run_one(label: str, training_config: dict, run_dir: str,
             device: torch.device, baselines: dict) -> dict:
    """Run one training experiment. Returns result dict with all metrics."""
    os.makedirs(run_dir, exist_ok=True)
    V_train, V_val = _load_data(DATA_DIFF12)

    t0 = time.time()
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
        model_name=f"recovery_{label}",
        max_train_samples=DEFAULT_N_TRAIN,
        train_shuffle=training_config.get("shuffle", True),
    )
    wall_time = time.time() - t0

    # Per-channel MAE evaluation
    ch_mae = compute_all_channel_mae(V_val, model, device)
    ch_target_mae = float(ch_mae[TARGET_CHANNEL])

    pre = baselines["ch_target_preshift_mae"]
    post = baselines["ch_target_postshift_mae"]
    recovery_pct = (post - ch_target_mae) / (post - pre) * 100 if post > pre else 0.0

    min_val = min(val_losses)
    best_ep = int(np.argmin(val_losses))

    result = {
        "label": label,
        "final_val": val_losses[-1],
        "min_val": min_val,
        "best_epoch": best_ep,
        "final_train": train_losses[-1],
        "train_val_gap": val_losses[-1] - train_losses[-1],
        "ch_target_mae": ch_target_mae,
        "ch_recovery_pct": recovery_pct,
        "wall_time": wall_time,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }

    print(f"\n{'─'*50}")
    print(f"[{label}] final_val={result['final_val']:.6f}  min_val={min_val:.6f} (ep{best_ep})")
    print(f"  Ch.{TARGET_CHANNEL} MAE={ch_target_mae:.4f}  recovery={recovery_pct:.1f}%")
    print(f"  train/val gap={result['train_val_gap']:.6f}  time={wall_time:.1f}s")
    print(f"{'─'*50}")

    return result


def _plot_comparison(results: List[dict], phase_name: str, output_path: str,
                     baselines: dict):
    """Plot val_loss curves for all experiments in a phase."""
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 5.6), constrained_layout=True)

    fs_label = 18
    fs_tick = 14
    fs_legend = 11

    for i, res in enumerate(results):
        color = COLORS[i % len(COLORS)]
        epochs = range(1, len(res["val_losses"]) + 1)
        ax.plot(epochs, res["val_losses"], color=color, linewidth=1.2,
                label=f"{res['label']} ({res['min_val']:.6f})")

    # Baseline references
    ax.axhline(y=baselines["pretrain_health_val"], color="#333333",
               linestyle="--", linewidth=0.8,
               label=f"Pretrain baseline ({baselines['pretrain_health_val']:.6f})")
    ax.axhline(y=0.001245, color="#999999",
               linestyle=":", linewidth=0.8,
               label="Current best TL (0.001245)")

    ax.set_xlabel("Epoch", fontsize=fs_label)
    ax.set_ylabel("Validation Loss (MSE)", fontsize=fs_label)
    ax.tick_params(labelsize=fs_tick)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="outside upper center",
               ncol=min(len(labels), 3),
               frameon=False, fontsize=fs_legend)

    fig.savefig(output_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"[Saved] {os.path.basename(output_path)}")


def _save_phase_csv(results: List[dict], baselines: dict, csv_path: str):
    """Save phase results to CSV."""
    rows = []
    for r in results:
        rows.append({
            "experiment": r["label"],
            "final_val_loss": f"{r['final_val']:.6f}",
            "min_val_loss": f"{r['min_val']:.6f}",
            "best_epoch": r["best_epoch"],
            "final_train_loss": f"{r['final_train']:.6f}",
            "train_val_gap": f"{r['train_val_gap']:.6f}",
            "ch123_mae": f"{r['ch_target_mae']:.5f}",
            "ch123_recovery_pct": f"{r['ch_recovery_pct']:.1f}",
            "wall_time_s": f"{r['wall_time']:.1f}",
            "gap_vs_baseline_pct": f"{(r['min_val'] / baselines['pretrain_health_val'] - 1) * 100:.2f}",
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[Saved] {os.path.basename(csv_path)}")


# ========================================
# Phase 1: LR Schedule Fixes
# ========================================

PHASE1_CONFIGS = {
    "1A_sgd_1000ep_cosT1000": _base_training_config(
        finetune={"epochs": 1000, "lr": 3e-4, "batch_size": 256},
        cosine_t_max=1000,
    ),
    "1B_sgd_1000ep_lr1e3": _base_training_config(
        finetune={"epochs": 1000, "lr": 1e-3, "batch_size": 256},
        cosine_t_max=1000,
    ),
    "1C_sgd_500ep_etamin1e4": _base_training_config(
        finetune={"epochs": 500, "lr": 3e-4, "batch_size": 256},
        cosine_eta_min=1e-4,
    ),
    "1D_adam_warm20_1000ep": _base_training_config(
        finetune={"epochs": 1000, "lr": 3e-4, "batch_size": 256},
        optimizer_type="adam",
        warmup_epochs=20,
        cosine_t_max=1000,
    ),
    "1E_sgd_1000ep_noSched": _base_training_config(
        finetune={"epochs": 1000, "lr": 1e-4, "batch_size": 256},
        use_lr_scheduler=False,
    ),
}


def run_phase1(device: torch.device, baselines: dict) -> List[dict]:
    """Execute Phase 1 experiments."""
    print(f"\n{'='*60}")
    print("Phase 1: LR Schedule Fixes")
    print(f"{'='*60}")

    phase_dir = os.path.join(OUTPUT_DIR, "phase1")
    results = []

    for name, cfg in PHASE1_CONFIGS.items():
        run_dir = os.path.join(phase_dir, name)
        result = _run_one(name, cfg, run_dir, device, baselines)
        results.append(result)

    # Summary
    _save_phase_csv(results, baselines, os.path.join(phase_dir, "phase1_comparison.csv"))
    _plot_comparison(results, "Phase 1", os.path.join(phase_dir, "fig_phase1_comparison.png"), baselines)

    # Decision gate
    best = min(results, key=lambda r: r["min_val"])
    target = 0.001160
    print(f"\n{'='*60}")
    print(f"Phase 1 Summary")
    print(f"{'='*60}")
    print(f"  Best: {best['label']} -> min_val={best['min_val']:.6f}")
    print(f"  Target: {target:.6f}")
    print(f"  Baseline: {baselines['pretrain_health_val']:.6f}")
    if best["min_val"] <= target:
        print(f"  >>> SUCCESS: Target reached!")
    elif best["min_val"] < 0.001230:
        print(f"  >>> IMPROVED but not reached target. Proceed to Phase 2.")
    else:
        print(f"  >>> NO significant improvement. LR schedule may not be the issue.")
    print(f"{'='*60}")

    return results


# ========================================
# Phase 2: Optimizer + Schedule Combos
# ========================================

def get_phase2_configs(best_phase1_label: str = None) -> dict:
    """Build Phase 2 configs, optionally informed by Phase 1 best."""
    configs = {
        "2A_adam_warm50_2000ep": _base_training_config(
            finetune={"epochs": 2000, "lr": 3e-4, "batch_size": 256},
            optimizer_type="adam",
            warmup_epochs=50,
            cosine_t_max=2000,
        ),
        "2B_adam_warm50_etamin1e4": _base_training_config(
            finetune={"epochs": 2000, "lr": 3e-4, "batch_size": 256},
            optimizer_type="adam",
            warmup_epochs=50,
            cosine_eta_min=1e-4,
            cosine_t_max=2000,
        ),
        "2C_sgd_noScheduler_2000ep": _base_training_config(
            finetune={"epochs": 2000, "lr": 1e-4, "batch_size": 256},
            use_lr_scheduler=False,
        ),
    }
    return configs


def run_phase2(device: torch.device, baselines: dict) -> List[dict]:
    """Execute Phase 2 experiments."""
    print(f"\n{'='*60}")
    print("Phase 2: Optimizer + Schedule Combos")
    print(f"{'='*60}")

    phase_dir = os.path.join(OUTPUT_DIR, "phase2")
    configs = get_phase2_configs()
    results = []

    for name, cfg in configs.items():
        run_dir = os.path.join(phase_dir, name)
        result = _run_one(name, cfg, run_dir, device, baselines)
        results.append(result)

    _save_phase_csv(results, baselines, os.path.join(phase_dir, "phase2_comparison.csv"))
    _plot_comparison(results, "Phase 2", os.path.join(phase_dir, "fig_phase2_comparison.png"), baselines)

    best = min(results, key=lambda r: r["min_val"])
    print(f"\n{'='*60}")
    print(f"Phase 2 Summary")
    print(f"{'='*60}")
    print(f"  Best: {best['label']} -> min_val={best['min_val']:.6f}")
    if best["min_val"] <= 0.001160:
        print(f"  >>> SUCCESS!")
    print(f"{'='*60}")

    return results


# ========================================
# Phase 3: Data Replay + Regularization
# ========================================

def _run_one_with_replay(label: str, training_config: dict, run_dir: str,
                         device: torch.device, baselines: dict,
                         health_ratio: float = 1.0) -> dict:
    """Run experiment with data replay: mix shifted + health data."""
    os.makedirs(run_dir, exist_ok=True)

    # Load shifted domain data
    V_train_shift, V_val = _load_data(DATA_DIFF12)

    # Load health data for replay
    V_health = np.load(DATA_HEALTH)["V"].astype(np.float32)
    n_replay = int(V_train_shift.shape[0] * health_ratio)
    # Randomly sample from health data (exclude last 200 used as val in pretrain)
    rng = np.random.RandomState(42)
    idx = rng.choice(V_health.shape[0] - VAL_SAMPLES, n_replay, replace=False)
    V_replay = V_health[idx]

    # Concatenate
    V_train_mixed = np.concatenate([V_train_shift, V_replay], axis=0)
    print(f"  [data replay] {V_train_shift.shape[0]} shifted + {n_replay} health = {V_train_mixed.shape[0]} total")

    t0 = time.time()
    ft = training_config["finetune"]
    model, train_losses, val_losses = train_model(
        V_train_mixed, V_val, device,
        epochs=ft["epochs"],
        lr=ft["lr"],
        batch_size=ft["batch_size"],
        output_dir=run_dir,
        ae_config=AE_CONFIG,
        training_config=training_config,
        plot_style=PLOT_STYLE,
        fig_dpi=FIG_DPI,
        pretrain_model_path=PRETRAIN_PTH,
        model_name=f"recovery_{label}",
        train_shuffle=training_config.get("shuffle", True),
    )
    wall_time = time.time() - t0

    # Per-channel MAE evaluation
    ch_mae = compute_all_channel_mae(V_val, model, device)
    ch_target_mae = float(ch_mae[TARGET_CHANNEL])

    pre = baselines["ch_target_preshift_mae"]
    post = baselines["ch_target_postshift_mae"]
    recovery_pct = (post - ch_target_mae) / (post - pre) * 100 if post > pre else 0.0

    min_val = min(val_losses)
    best_ep = int(np.argmin(val_losses))

    result = {
        "label": label,
        "final_val": val_losses[-1],
        "min_val": min_val,
        "best_epoch": best_ep,
        "final_train": train_losses[-1],
        "train_val_gap": val_losses[-1] - train_losses[-1],
        "ch_target_mae": ch_target_mae,
        "ch_recovery_pct": recovery_pct,
        "wall_time": wall_time,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }

    print(f"\n{'─'*50}")
    print(f"[{label}] final_val={result['final_val']:.6f}  min_val={min_val:.6f} (ep{best_ep})")
    print(f"  Ch.{TARGET_CHANNEL} MAE={ch_target_mae:.4f}  recovery={recovery_pct:.1f}%")
    print(f"  train/val gap={result['train_val_gap']:.6f}  time={wall_time:.1f}s")
    print(f"{'─'*50}")

    return result


PHASE3_CONFIGS = {
    # Data replay experiments: SGD lr=1e-3 (Phase1 best) + health data
    "3A_sgd_lr1e3_replay1x": {
        "cfg": _base_training_config(
            finetune={"epochs": 1000, "lr": 1e-3, "batch_size": 256},
            cosine_t_max=1000,
        ),
        "health_ratio": 1.0,  # 400 shifted + 400 health
    },
    "3B_sgd_lr1e3_replay2x": {
        "cfg": _base_training_config(
            finetune={"epochs": 1000, "lr": 1e-3, "batch_size": 256},
            cosine_t_max=1000,
        ),
        "health_ratio": 2.0,  # 400 shifted + 800 health
    },
    # Weight decay experiments
    "3C_sgd_lr1e3_wd1e4": {
        "cfg": _base_training_config(
            finetune={"epochs": 1000, "lr": 1e-3, "batch_size": 256},
            cosine_t_max=1000,
            weight_decay=1e-4,
        ),
        "health_ratio": 0,
    },
    # Adam + replay (exploiting Adam's fast Ch.123 recovery + replay regularization)
    "3D_adam_w20_replay1x": {
        "cfg": _base_training_config(
            finetune={"epochs": 1000, "lr": 3e-4, "batch_size": 256},
            optimizer_type="adam",
            warmup_epochs=20,
            cosine_t_max=1000,
        ),
        "health_ratio": 1.0,
    },
    # Best combo: SGD lr=1e-3 + replay + small weight decay
    "3E_sgd_lr1e3_replay1x_wd1e5": {
        "cfg": _base_training_config(
            finetune={"epochs": 1000, "lr": 1e-3, "batch_size": 256},
            cosine_t_max=1000,
            weight_decay=1e-5,
        ),
        "health_ratio": 1.0,
    },
}


def run_phase3(device: torch.device, baselines: dict) -> List[dict]:
    """Execute Phase 3: Data replay + regularization experiments."""
    print(f"\n{'='*60}")
    print("Phase 3: Data Replay + Regularization")
    print(f"{'='*60}")

    phase_dir = os.path.join(OUTPUT_DIR, "phase3")
    results = []

    for name, spec in PHASE3_CONFIGS.items():
        run_dir = os.path.join(phase_dir, name)
        hr = spec["health_ratio"]
        if hr > 0:
            result = _run_one_with_replay(name, spec["cfg"], run_dir, device,
                                          baselines, health_ratio=hr)
        else:
            result = _run_one(name, spec["cfg"], run_dir, device, baselines)
        results.append(result)

    _save_phase_csv(results, baselines, os.path.join(phase_dir, "phase3_comparison.csv"))
    _plot_comparison(results, "Phase 3", os.path.join(phase_dir, "fig_phase3_comparison.png"), baselines)

    best = min(results, key=lambda r: r["min_val"])
    print(f"\n{'='*60}")
    print(f"Phase 3 Summary")
    print(f"{'='*60}")
    print(f"  Best: {best['label']} -> min_val={best['min_val']:.6f}")
    print(f"  Ch.{TARGET_CHANNEL} MAE={best['ch_target_mae']:.5f}  recovery={best['ch_recovery_pct']:.1f}%")
    if best["min_val"] <= 0.001160:
        print(f"  >>> SUCCESS: Target reached!")
    print(f"{'='*60}")

    return results


# ========================================
# Main entry
# ========================================

# ========================================
# Phase 4: Data Augmentation
# ========================================

def _run_one_augmented(label: str, training_config: dict, run_dir: str,
                       device: torch.device, baselines: dict,
                       noise_std: float = 0.01, n_copies: int = 3) -> dict:
    """Run experiment with Gaussian noise augmentation."""
    os.makedirs(run_dir, exist_ok=True)
    V_train_orig, V_val = _load_data(DATA_DIFF12)

    # Create augmented training data
    rng = np.random.RandomState(42)
    chunks = [V_train_orig]
    for _ in range(n_copies - 1):
        noise = rng.normal(0, noise_std, size=V_train_orig.shape).astype(np.float32)
        chunks.append(V_train_orig + noise)
    V_train_aug = np.concatenate(chunks, axis=0)
    print(f"  [augmentation] {V_train_orig.shape[0]} orig × {n_copies} copies "
          f"(σ={noise_std}) = {V_train_aug.shape[0]} total")

    t0 = time.time()
    ft = training_config["finetune"]
    model, train_losses, val_losses = train_model(
        V_train_aug, V_val, device,
        epochs=ft["epochs"],
        lr=ft["lr"],
        batch_size=ft["batch_size"],
        output_dir=run_dir,
        ae_config=AE_CONFIG,
        training_config=training_config,
        plot_style=PLOT_STYLE,
        fig_dpi=FIG_DPI,
        pretrain_model_path=PRETRAIN_PTH,
        model_name=f"recovery_{label}",
        train_shuffle=training_config.get("shuffle", True),
    )
    wall_time = time.time() - t0

    ch_mae = compute_all_channel_mae(V_val, model, device)
    ch_target_mae = float(ch_mae[TARGET_CHANNEL])
    pre = baselines["ch_target_preshift_mae"]
    post = baselines["ch_target_postshift_mae"]
    recovery_pct = (post - ch_target_mae) / (post - pre) * 100 if post > pre else 0.0

    min_val = min(val_losses)
    best_ep = int(np.argmin(val_losses))

    result = {
        "label": label,
        "final_val": val_losses[-1],
        "min_val": min_val,
        "best_epoch": best_ep,
        "final_train": train_losses[-1],
        "train_val_gap": val_losses[-1] - train_losses[-1],
        "ch_target_mae": ch_target_mae,
        "ch_recovery_pct": recovery_pct,
        "wall_time": wall_time,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }

    print(f"\n{'─'*50}")
    print(f"[{label}] final_val={result['final_val']:.6f}  min_val={min_val:.6f} (ep{best_ep})")
    print(f"  Ch.{TARGET_CHANNEL} MAE={ch_target_mae:.4f}  recovery={recovery_pct:.1f}%")
    print(f"  train/val gap={result['train_val_gap']:.6f}  time={wall_time:.1f}s")
    print(f"{'─'*50}")

    return result


def run_phase4(device: torch.device, baselines: dict) -> List[dict]:
    """Phase 4: Data augmentation experiments."""
    print(f"\n{'='*60}")
    print("Phase 4: Data Augmentation")
    print(f"{'='*60}")

    phase_dir = os.path.join(OUTPUT_DIR, "phase4")

    # Check data scale for noise calibration
    V_train, _ = _load_data(DATA_DIFF12)
    data_std = V_train.std()
    print(f"  Data std: {data_std:.4f}")

    experiments = [
        ("4A_sgd_lr1e3_aug5x_s001", 0.01 * data_std, 5),    # 0.1% noise, 5× data
        ("4B_sgd_lr1e3_aug5x_s005", 0.05 * data_std, 5),    # 0.5% noise, 5× data
        ("4C_sgd_lr1e3_aug10x_s001", 0.01 * data_std, 10),  # 0.1% noise, 10× data
        ("4D_sgd_lr1e3_aug5x_s01", 0.1 * data_std, 5),      # 1% noise, 5× data
    ]

    cfg = _base_training_config(
        finetune={"epochs": 1000, "lr": 1e-3, "batch_size": 256},
        cosine_t_max=1000,
    )

    results = []
    for name, noise_std, n_copies in experiments:
        run_dir = os.path.join(phase_dir, name)
        result = _run_one_augmented(name, cfg, run_dir, device, baselines,
                                    noise_std=noise_std, n_copies=n_copies)
        results.append(result)

    _save_phase_csv(results, baselines, os.path.join(phase_dir, "phase4_comparison.csv"))
    _plot_comparison(results, "Phase 4", os.path.join(phase_dir, "fig_phase4_comparison.png"), baselines)

    best = min(results, key=lambda r: r["min_val"])
    print(f"\n{'='*60}")
    print(f"Phase 4 Summary")
    print(f"{'='*60}")
    print(f"  Best: {best['label']} -> min_val={best['min_val']:.6f}")
    print(f"  Ch.{TARGET_CHANNEL} MAE={best['ch_target_mae']:.5f}  recovery={best['ch_recovery_pct']:.1f}%")
    if best["min_val"] <= 0.001160:
        print(f"  >>> SUCCESS!")
    print(f"{'='*60}")

    return results


# ========================================
# Phase 5: L2-SP Regularization
# ========================================

def _train_with_l2sp(
    V_train: np.ndarray, V_val: np.ndarray, device: torch.device,
    epochs: int, lr: float, batch_size: int, output_dir: str,
    alpha: float, pretrain_path: str,
) -> tuple:
    """Custom training loop with L2-SP: loss = MSE + α·||θ-θ₀||²."""
    import torch.nn as nn
    from torch.optim import SGD
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader, TensorDataset

    os.makedirs(output_dir, exist_ok=True)
    D = V_train.shape[1]

    # Build model & load pretrain
    model = Autoencoder(D, **{k: AE_CONFIG[k] for k in
        ["encoder_dims", "latent_dim", "decoder_dims", "dropout", "activation"]}).to(device)
    model.load_state_dict(torch.load(pretrain_path, map_location=device, weights_only=True))

    # Save pretrain params as reference (frozen)
    pretrain_params = {n: p.clone().detach() for n, p in model.named_parameters()}

    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    X_train = torch.from_numpy(V_train).to(device)
    X_val = torch.from_numpy(V_val).to(device)
    train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val), batch_size=batch_size, shuffle=False)

    best_val = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        running, count = 0.0, 0
        for (xb,) in train_loader:
            optimizer.zero_grad(set_to_none=True)
            recon = model(xb)
            mse_loss = criterion(recon, xb)

            # L2-SP penalty: α * Σ||θ_i - θ₀_i||²
            l2sp = torch.tensor(0.0, device=device)
            for name, param in model.named_parameters():
                l2sp = l2sp + torch.sum((param - pretrain_params[name]) ** 2)

            loss = mse_loss + alpha * l2sp
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += mse_loss.item() * xb.size(0)
            count += xb.size(0)
        train_epoch = running / max(1, count)

        model.eval()
        vrunning, vcount = 0.0, 0
        with torch.no_grad():
            for (xb,) in val_loader:
                recon = model(xb)
                vloss = criterion(recon, xb)
                vrunning += vloss.item() * xb.size(0)
                vcount += xb.size(0)
        val_epoch = vrunning / max(1, vcount)

        train_losses.append(train_epoch)
        val_losses.append(val_epoch)

        lr_now = optimizer.param_groups[0]['lr']
        scheduler.step()

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:4d}/{epochs} | Train: {train_epoch:.6f} | "
                  f"Val: {val_epoch:.6f} | LR: {lr_now:.2e}")

        if val_epoch + 1e-9 < best_val:
            best_val = val_epoch
            torch.save(model.state_dict(), os.path.join(output_dir, "autoencoder.pth"))

    # Reload best
    model.load_state_dict(torch.load(os.path.join(output_dir, "autoencoder.pth"),
                                     map_location=device, weights_only=True))
    pd.DataFrame({"train_loss": train_losses, "val_loss": val_losses}).to_csv(
        os.path.join(output_dir, "training_losses.csv"), index=False)

    return model, train_losses, val_losses


def run_phase5(device: torch.device, baselines: dict) -> List[dict]:
    """Phase 5: L2-SP regularization experiments."""
    print(f"\n{'='*60}")
    print("Phase 5: L2-SP Regularization")
    print(f"{'='*60}")

    phase_dir = os.path.join(OUTPUT_DIR, "phase5")
    V_train, V_val = _load_data(DATA_DIFF12)

    alphas = [1e-3, 1e-4, 1e-5, 1e-6]
    results = []

    for alpha in alphas:
        label = f"5_l2sp_a{alpha:.0e}"
        run_dir = os.path.join(phase_dir, label)
        print(f"\n--- {label}: α={alpha} ---")

        t0 = time.time()
        model, train_losses, val_losses = _train_with_l2sp(
            V_train, V_val, device,
            epochs=1000, lr=1e-3, batch_size=256,
            output_dir=run_dir, alpha=alpha,
            pretrain_path=PRETRAIN_PTH,
        )
        wall_time = time.time() - t0

        ch_mae = compute_all_channel_mae(V_val, model, device)
        ch_target_mae = float(ch_mae[TARGET_CHANNEL])
        pre = baselines["ch_target_preshift_mae"]
        post = baselines["ch_target_postshift_mae"]
        recovery_pct = (post - ch_target_mae) / (post - pre) * 100 if post > pre else 0.0

        min_val = min(val_losses)
        best_ep = int(np.argmin(val_losses))

        result = {
            "label": label,
            "final_val": val_losses[-1],
            "min_val": min_val,
            "best_epoch": best_ep,
            "final_train": train_losses[-1],
            "train_val_gap": val_losses[-1] - train_losses[-1],
            "ch_target_mae": ch_target_mae,
            "ch_recovery_pct": recovery_pct,
            "wall_time": wall_time,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }
        results.append(result)

        print(f"  min_val={min_val:.6f} (ep{best_ep})  Ch.123={ch_target_mae:.4f}  "
              f"recovery={recovery_pct:.1f}%  gap={result['train_val_gap']:.6f}")

    _save_phase_csv(results, baselines, os.path.join(phase_dir, "phase5_comparison.csv"))
    _plot_comparison(results, "Phase 5: L2-SP", os.path.join(phase_dir, "fig_phase5_comparison.png"), baselines)

    best = min(results, key=lambda r: r["min_val"])
    print(f"\n{'='*60}")
    print(f"Phase 5 Summary")
    print(f"{'='*60}")
    print(f"  Best: {best['label']} -> min_val={best['min_val']:.6f}")
    print(f"  Ch.{TARGET_CHANNEL} MAE={best['ch_target_mae']:.5f}  recovery={best['ch_recovery_pct']:.1f}%")
    if best["min_val"] <= 0.001160:
        print(f"  >>> SUCCESS!")
    print(f"{'='*60}")

    return results


def run_recovery_study(phase: str = "all"):
    """Run the recovery study.

    Args:
        phase: 'all' (full pipeline), 'phase1', 'phase2', 'phase3', or 'phase3_only'
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Compute baselines
    print("\n[Baselines] Computing reference values...")
    baselines = _compute_baselines(device)
    print(f"  pretrain_health_val = {baselines['pretrain_health_val']:.6f}")
    print(f"  pretrain_shifted_val = {baselines['pretrain_shifted_val']:.6f}")
    print(f"  Ch.{TARGET_CHANNEL} pre-shift MAE = {baselines['ch_target_preshift_mae']:.5f}")
    print(f"  Ch.{TARGET_CHANNEL} post-shift MAE = {baselines['ch_target_postshift_mae']:.5f}")

    all_results = []

    if phase in ("all", "phase1"):
        p1_results = run_phase1(device, baselines)
        all_results.extend(p1_results)

    if phase in ("all", "phase2"):
        p2_results = run_phase2(device, baselines)
        all_results.extend(p2_results)

    if phase in ("all", "phase3", "phase3_only"):
        p3_results = run_phase3(device, baselines)
        all_results.extend(p3_results)

    if phase in ("all", "phase4", "phase4_only"):
        p4_results = run_phase4(device, baselines)
        all_results.extend(p4_results)

    if phase in ("all", "phase5", "phase5_only"):
        p5_results = run_phase5(device, baselines)
        all_results.extend(p5_results)

    if not all_results:
        print("[Warning] No experiments run.")
        return

    # Global summary
    _save_phase_csv(all_results, baselines,
                    os.path.join(OUTPUT_DIR, f"recovery_study_summary_{phase}.csv"))
    _plot_comparison(all_results, phase,
                     os.path.join(OUTPUT_DIR, f"fig_recovery_{phase}.png"), baselines)

    best_overall = min(all_results, key=lambda r: r["min_val"])
    print(f"\n{'='*60}")
    print(f"OVERALL BEST: {best_overall['label']}")
    print(f"  min_val_loss = {best_overall['min_val']:.6f}")
    print(f"  Ch.{TARGET_CHANNEL} MAE = {best_overall['ch_target_mae']:.5f}")
    print(f"  Ch.{TARGET_CHANNEL} recovery = {best_overall['ch_recovery_pct']:.1f}%")
    print(f"  Gap vs baseline = {(best_overall['min_val']/baselines['pretrain_health_val'] - 1)*100:.2f}%")
    print(f"{'='*60}")
