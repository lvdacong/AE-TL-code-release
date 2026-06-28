"""
AE_run_sd_difficulty_ablation.py
=================================
Difficulty ablation for the Sensor Drift scenario.
6 drift ratios x 4 data sizes = 24 TL training runs.

For each drift_ratio, ALL channels of the original health data are scaled
by (1 + drift_ratio), simulating temperature-induced global sensor drift.

Usage:
    cd script && python AE_run_sd_difficulty_ablation.py
"""

from __future__ import annotations

import os
import shutil
import sys

import numpy as np
import torch

# ========================================
# Path setup
# ========================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

AD_OUTPUT = os.path.join(SCRIPT_DIR, "AD_preprocess_datasets_output")
AE_OUTPUT = os.path.join(SCRIPT_DIR, "AE_model_train_and_detect_output")

PRETRAIN_PTH = os.path.join(AE_OUTPUT, "Damage_Repaired", "pretrain", "autoencoder.pth")

DIFFICULTY_VALUES = [0, 0.05, 0.10, 0.15, 0.20, 0.25]
DATA_SIZES = [50, 100, 200, 400]

# Training config (consistent with existing experiments)
SETTINGS = {
    "model": {
        "encoder_dims": [768, 384, 192],
        "latent_dim": 192,
        "decoder_dims": [192, 384, 768],
        "dropout": 0.0,
        "activation": "relu",
    },
    "training": {
        "finetune": {
            "epochs": 1000,
            "lr": 1e-3,
            "batch_size": 256,
        },
        "val_samples": 100,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "use_lr_scheduler": True,
        "lr_scheduler_type": "cosine",
        "cosine_eta_min": 1e-6,
        "shuffle": True,
        "optimizer_type": "adam",
        "sgd_momentum": 0.9,
        "noise_std": 0.005,
        "encoder_lr": 1e-5,
    },
}


# ========================================
# Drift application (replicates AD_preprocess_datasets.apply_drift_to_data)
# ========================================

# Per-channel drift factors: simulate spatially non-uniform temperature field.
# Each sensor sits at a different position on the hull, so the local temperature
# (and hence the drift magnitude) varies.  A fixed seed ensures reproducibility.
_RNG = np.random.RandomState(42)
_N_CHANNELS = 252
CHANNEL_DRIFT_FACTORS = 1.0 + 0.3 * _RNG.randn(_N_CHANNELS)
CHANNEL_DRIFT_FACTORS = np.clip(CHANNEL_DRIFT_FACTORS, 0.4, 1.6).astype(np.float32)


def apply_drift(V: np.ndarray, ratio: float) -> np.ndarray:
    """
    Apply spatially non-uniform scale-type drift.

    Each channel d receives a different effective ratio:
        drifted[:, d] = original[:, d] * (1 + ratio * factor[d])
    where factor[d] ~ N(1, 0.3), clipped to [0.4, 1.6].

    At ratio=0.25 the per-channel ratios range from ~0.10 to ~0.40,
    modelling a realistic temperature gradient across the hull.

    Args:
        V:     data matrix, shape (n_samples, n_features)
        ratio: base drift ratio (0 = no drift)

    Returns:
        A copy of V with non-uniform drift applied.
    """
    if ratio == 0:
        return V.copy()
    D = V.shape[1]
    factors = CHANNEL_DRIFT_FACTORS[:D]
    return V * (1.0 + ratio * factors)


# ========================================
# Main
# ========================================

def main():
    print("=" * 60)
    print("Sensor_Drift Difficulty Ablation (all-channel temperature drift)")
    print(f"  difficulty (drift_ratio): {DIFFICULTY_VALUES}")
    print(f"  data_sizes: {DATA_SIZES}")
    print(f"  drift: all 252 channels")
    print(f"  total training runs: {len(DIFFICULTY_VALUES) * len(DATA_SIZES)}")
    print("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[device] {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Verify pretrain model
    if not os.path.exists(PRETRAIN_PTH):
        print(f"[error] Pretrain model not found: {PRETRAIN_PTH}")
        sys.exit(1)
    print(f"[pretrain] {PRETRAIN_PTH}")

    # ============================================
    # Step 1: Load base data
    # ============================================
    print(f"\n{'=' * 60}")
    print("[Step 1] Loading base (un-drifted) data")
    print(f"{'=' * 60}")

    health_npz = os.path.join(AD_OUTPUT, "health_original_2000", "preprocessed_data_raw.npz")
    if not os.path.exists(health_npz):
        print(f"[error] Health data not found: {health_npz}")
        sys.exit(1)

    V_health_base = np.load(health_npz)["V"].astype(np.float32)
    print(f"  health_original_2000: shape={V_health_base.shape}")

    # ============================================
    # Step 2: Build data_loader_func and run ablation engine
    # ============================================
    print(f"\n{'=' * 60}")
    print("[Step 2] Running ablation experiments")
    print(f"{'=' * 60}")

    from AE_difficulty_ablation_auxiliary import run_difficulty_ablation

    val_samples = SETTINGS["training"]["val_samples"]

    # Pre-compute drifted versions for each ratio
    drifted_data: dict[float, np.ndarray] = {}
    for ratio in DIFFICULTY_VALUES:
        drifted_data[ratio] = apply_drift(V_health_base, ratio)
        print(f"  [drift] ratio={ratio:.3f}: all-channel "
              f"mean {V_health_base.mean():.4f} -> "
              f"{drifted_data[ratio].mean():.4f}")

    def data_loader_func(difficulty: float, data_size: int):
        """
        Return (V_train, V_val) for the ablation engine.

        V_train: first data_size samples from the training pool (drifted)
        V_val:   last val_samples samples (drifted)
        """
        V_all = drifted_data[difficulty]
        N = V_all.shape[0]

        # Fixed validation set = last val_samples samples
        V_train_pool = V_all[:N - val_samples]
        V_val = V_all[N - val_samples:]

        # Truncate training set to data_size
        if data_size > V_train_pool.shape[0]:
            print(f"    [warning] requested {data_size} but only "
                  f"{V_train_pool.shape[0]} training samples available")
            V_train = V_train_pool
        else:
            V_train = V_train_pool[:data_size]

        return V_train, V_val

    output_dir = os.path.join(AE_OUTPUT, "Sensor_Drift", "difficulty_ablation")

    loss_matrix, before_tl_values = run_difficulty_ablation(
        scenario_name="Sensor_Drift",
        difficulty_values=DIFFICULTY_VALUES,
        data_sizes=DATA_SIZES,
        pretrain_model_path=PRETRAIN_PTH,
        data_loader_func=data_loader_func,
        output_dir=output_dir,
        settings=SETTINGS,
        device=device,
        xlabel="Drift Ratio",
        freeze_strategy="none",
    )

    # ============================================
    # Step 3: Summary
    # ============================================
    print(f"\n{'=' * 60}")
    print("[Step 3] Results summary")
    print(f"{'=' * 60}")

    print(f"\nBefore TL (pretrain direct inference):")
    for j, diff in enumerate(DIFFICULTY_VALUES):
        print(f"  drift_ratio={diff:.2f}: MSE = {before_tl_values[j]:.6f}")

    print(f"\nTL val_loss matrix (rows=data_size, cols=drift_ratio):")
    header = "data_size\t" + "\t".join([f"r={d:.2f}" for d in DIFFICULTY_VALUES])
    print(header)
    for i, ds in enumerate(DATA_SIZES):
        row = f"{ds}\t\t" + "\t".join(
            [f"{loss_matrix[i, j]:.6f}" for j in range(len(DIFFICULTY_VALUES))]
        )
        print(row)

    # Copy the generated figure to the paper-referenced filename
    engine_fig = os.path.join(output_dir, "difficulty_surface_Sensor_Drift.png")
    paper_fig = os.path.join(output_dir, "fig_sd_difficulty_surface.png")
    if os.path.exists(engine_fig):
        shutil.copy2(engine_fig, paper_fig)
        print(f"  [copy] {os.path.basename(engine_fig)} -> {os.path.basename(paper_fig)}")

    print(f"\n[done] Output directory: {output_dir}")
    print(f"  CSV: difficulty_ablation_Sensor_Drift.csv")
    print(f"  Fig: difficulty_surface_Sensor_Drift.png")
    print(f"  Fig: fig_sd_difficulty_surface.png (paper reference)")


if __name__ == "__main__":
    main()
