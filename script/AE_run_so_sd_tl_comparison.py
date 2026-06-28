"""
AE_run_so_sd_tl_comparison.py
==============================
TL comparison for the Sensor Relocation and Sensor Drift scenarios.
Runs full TL/OldModel/FromScratch training and visualization.

Usage:
    cd script && python AE_run_so_sd_tl_comparison.py
"""

from __future__ import annotations

import os
import sys
import shutil

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from AE_tl_comparison_auxiliary import run_tl_comparison
from AE_run_dr_tl_comparison import SETTINGS  # reuse DR settings template
from AE_run_sd_difficulty_ablation import apply_drift, CHANNEL_DRIFT_FACTORS

PRETRAIN_PTH = os.path.join(
    SCRIPT_DIR, "AE_model_train_and_detect_output",
    "Damage_Repaired", "pretrain", "autoencoder.pth"
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    if not os.path.exists(PRETRAIN_PTH):
        print(f"[error] Pretrain model not found: {PRETRAIN_PTH}")
        sys.exit(1)

    # SO/SD use first_damage.inp (not DR's second_damage_12.inp)
    # 并覆盖场景特定训练参数（DR 的 SETTINGS 含 DR 专用的 l2sp）
    settings = {**SETTINGS}
    settings["training"] = {**SETTINGS["training"]}
    settings["detection"] = {**SETTINGS["detection"]}
    settings["detection"]["render_3d"] = True
    settings["detection"]["inp_file_path"] = "C:/SHM_abaqus_models/first_damage.inp"

    # ========================================
    # Sensor_Offset: offset_count=1 (max-stress-diff scheme), N=400
    # ========================================
    print("\n" + "=" * 60)
    print("Sensor_Offset tl_comparison (offset_count=1, N=400)")
    print("=" * 60)

    SO_CONFIG = {
        "name": "Sensor_Offset",
        "description": "SO tl_comparison at offset_count=1 (max-stress-diff), N=400",
        "new_data_folder": "health_offset_count_1_2000",
        "test_damage_folder": "first_damage_offset_count_1_100",
        "tl_comparison": {"train_samples": 400, "freeze_strategy": "none"},
    }

    # SO 场景特定训练参数（§6.2 差异学习率 —— 仅 SO 场景启用）
    so_settings = {**settings}
    so_settings["training"] = {**settings["training"], "l2_sp_alpha": 0.0, "encoder_lr": 1e-5}

    so_results = run_tl_comparison(
        scenario_name="Sensor_Offset",
        scenario_config=SO_CONFIG,
        pretrain_pth=PRETRAIN_PTH,
        settings=so_settings,
        device=device,
    )

    print("\n[SO Results]")
    for key, result in so_results.items():
        vl = result.get("val_losses", [])
        final = vl[-1] if vl else float("nan")
        print(f"  {key}: final_val_loss = {final:.6f}")

    # ========================================
    # Sensor_Drift: all-channel temperature drift, ratio=0.01, N=400
    # ========================================
    print("\n" + "=" * 60)
    print("Sensor_Drift tl_comparison (drift=all, ratio=0.10, N=400)")
    print("=" * 60)

    # Pre-generate non-uniform drift data for SD tl_comparison (ratio=0.10)
    SD_DRIFT_RATIO = 0.10
    ad_output = os.path.join(SCRIPT_DIR, "AD_preprocess_datasets_output")
    V_health = np.load(os.path.join(ad_output, "health_original_2000", "preprocessed_data_raw.npz"))["V"].astype(np.float32)
    V_damage_orig = np.load(os.path.join(ad_output, "first_damage_drift_all_100", "preprocessed_data_raw.npz"))["V"].astype(np.float32)

    # Apply non-uniform drift (same factors as difficulty ablation)
    sd_health_dir = os.path.join(ad_output, "health_drift_nonuniform_2000")
    sd_damage_dir = os.path.join(ad_output, "first_damage_drift_nonuniform_100")
    os.makedirs(sd_health_dir, exist_ok=True)
    os.makedirs(sd_damage_dir, exist_ok=True)
    # health: non-uniform drift at SD_DRIFT_RATIO
    V_health_drift = apply_drift(V_health, SD_DRIFT_RATIO)
    np.savez(os.path.join(sd_health_dir, "preprocessed_data_raw.npz"), V=V_health_drift)
    # damage: apply same non-uniform drift to original (un-drifted) damage data
    # first_damage_drift_all_100 was ALREADY uniformly drifted; use original instead
    first_dmg_orig_path = os.path.join(ad_output, "first_damage_original_100", "preprocessed_data_raw.npz")
    if os.path.exists(first_dmg_orig_path):
        V_dmg_orig = np.load(first_dmg_orig_path)["V"].astype(np.float32)
        V_dmg_drift = apply_drift(V_dmg_orig, SD_DRIFT_RATIO)
    else:
        # Fallback: apply non-uniform drift to the already-drifted data
        # (undo old uniform drift first)
        V_dmg_undrift = V_damage_orig / (1 + SD_DRIFT_RATIO)
        V_dmg_drift = apply_drift(V_dmg_undrift, SD_DRIFT_RATIO)
    np.savez(os.path.join(sd_damage_dir, "preprocessed_data_raw.npz"), V=V_dmg_drift)
    print(f"  [SD] Non-uniform drift data generated (ratio={SD_DRIFT_RATIO})")

    SD_CONFIG = {
        "name": "Sensor_Drift",
        "description": "SD tl_comparison at drift_ratio=0.10 (non-uniform), N=400",
        "new_data_folder": "health_drift_nonuniform_2000",
        "test_damage_folder": "first_damage_drift_nonuniform_100",
        "tl_comparison": {"train_samples": 400, "freeze_strategy": "none"},
    }

    # SD 场景特定训练参数
    sd_settings = {**settings}
    sd_settings["training"] = {**settings["training"], "l2_sp_alpha": 0.0}

    sd_results = run_tl_comparison(
        scenario_name="Sensor_Drift",
        scenario_config=SD_CONFIG,
        pretrain_pth=PRETRAIN_PTH,
        settings=sd_settings,
        device=device,
    )

    print("\n[SD Results]")
    for key, result in sd_results.items():
        vl = result.get("val_losses", [])
        final = vl[-1] if vl else float("nan")
        print(f"  {key}: final_val_loss = {final:.6f}")

    print("\n" + "=" * 60)
    print("[Done] SO & SD tl_comparison complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
