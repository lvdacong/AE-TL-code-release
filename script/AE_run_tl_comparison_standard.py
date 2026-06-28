"""
AE_run_tl_comparison_standard.py
==================================
Run tl_comparison with STANDARD SGD config (no diffLR, no L2-SP).
Results saved to {scenario}/tl_comparison_standard/ for dual-mode comparison.

This serves as the "detection-priority" baseline against the "recovery-priority"
diffLR config in {scenario}/tl_comparison/.

Usage:
    cd script && python AE_run_tl_comparison_standard.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from AE_tl_comparison_auxiliary import run_tl_comparison
from AE_run_sd_difficulty_ablation import apply_drift

PRETRAIN_PTH = os.path.join(
    SCRIPT_DIR, "AE_model_train_and_detect_output",
    "Damage_Repaired", "pretrain", "autoencoder.pth",
)
WORKSPACE_DIR = os.path.dirname(SCRIPT_DIR)

# Standard SGD config (the original default, no enhancements)
SETTINGS = {
    "global": {
        "workspace_dir": WORKSPACE_DIR,
        "preprocess_output_base": "script/AD_preprocess_datasets_output",
        "output_dir": "script/AE_model_train_and_detect_output",
        "seed": 42,
        "device": "auto",
    },
    "model": {
        "encoder_dims": [768, 384, 192],
        "latent_dim": 192,
        "decoder_dims": [192, 384, 768],
        "dropout": 0.0,
        "activation": "relu",
    },
    "training": {
        "pretrain": {"epochs": 2000, "lr": 3e-4, "batch_size": 256},
        "finetune": {"epochs": 1000, "lr": 1e-3, "batch_size": 256},
        "val_samples": 100,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "use_lr_scheduler": True,
        "lr_scheduler_type": "cosine",
        "cosine_eta_min": 1e-6,
        "shuffle": True,
        "optimizer_type": "sgd",
        "sgd_momentum": 0.9,
        # NO encoder_lr, NO l2_sp_alpha, NO noise_std
    },
    "detection": {
        "enabled": True,
        "render_3d": False,
        "inp_file_path": "C:/SHM_abaqus_models/first_damage.inp",
        "middlewhole_elset": "middlewhole",
        "camera_config_path": "script/camera_position.json",
        "ac_output_dir": "script/AC_convert_and_extract_output",
        "vtu_file": "whole_from_inp.vtu",
        "id_mapping_file": "abaqus_id_to_vtu_index.csv",
        "measures_file": "measures_ID_original.csv",
    },
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    if not os.path.exists(PRETRAIN_PTH):
        print(f"[error] Pretrain model not found: {PRETRAIN_PTH}")
        sys.exit(1)

    ad_output = os.path.join(SCRIPT_DIR, "AD_preprocess_datasets_output")

    # ========================================
    # DR: standard SGD TL
    # ========================================
    print("\n" + "=" * 60)
    print("DR standard TL (SGD, no diffLR)")
    print("=" * 60)

    dr_settings = {**SETTINGS}
    dr_settings["detection"] = {**SETTINGS["detection"]}
    dr_settings["detection"]["inp_file_path"] = "C:/SHM_abaqus_models/second_damage_12.inp"

    DR_CONFIG = {
        "name": "Damage_Repaired",
        "description": "DR standard TL at N_r=12, N=400",
        "new_data_folder": "damage_repaired_12_original_500",
        "test_damage_folder": "second_damage_12_original_100",
        "tl_comparison": {"train_samples": 400, "freeze_strategy": "none"},
    }

    run_tl_comparison(
        scenario_name="Damage_Repaired",
        scenario_config=DR_CONFIG,
        pretrain_pth=PRETRAIN_PTH,
        settings=dr_settings,
        device=device,
        output_subdir="tl_comparison_standard",
    )

    # ========================================
    # SO: standard SGD TL
    # ========================================
    print("\n" + "=" * 60)
    print("SO standard TL (SGD, no diffLR)")
    print("=" * 60)

    SO_CONFIG = {
        "name": "Sensor_Offset",
        "description": "SO standard TL at offset_count=1 (max-stress-diff), N=400",
        "new_data_folder": "health_offset_count_1_2000",
        "test_damage_folder": "first_damage_offset_count_1_100",
        "tl_comparison": {"train_samples": 400, "freeze_strategy": "none"},
    }

    run_tl_comparison(
        scenario_name="Sensor_Offset",
        scenario_config=SO_CONFIG,
        pretrain_pth=PRETRAIN_PTH,
        settings=SETTINGS,
        device=device,
        output_subdir="tl_comparison_standard",
    )

    # ========================================
    # SD: standard SGD TL (non-uniform drift)
    # ========================================
    print("\n" + "=" * 60)
    print("SD standard TL (SGD, non-uniform drift)")
    print("=" * 60)

    # Generate non-uniform drift data
    SD_RATIO = 0.10
    V_health = np.load(os.path.join(ad_output, "health_original_2000", "preprocessed_data_raw.npz"))["V"].astype(np.float32)
    sd_health_dir = os.path.join(ad_output, "health_drift_nonuniform_2000")
    os.makedirs(sd_health_dir, exist_ok=True)
    np.savez(os.path.join(sd_health_dir, "preprocessed_data_raw.npz"), V=apply_drift(V_health, SD_RATIO))

    sd_damage_dir = os.path.join(ad_output, "first_damage_drift_nonuniform_100")
    os.makedirs(sd_damage_dir, exist_ok=True)
    first_dmg_path = os.path.join(ad_output, "first_damage_original_100", "preprocessed_data_raw.npz")
    if os.path.exists(first_dmg_path):
        V_dmg = np.load(first_dmg_path)["V"].astype(np.float32)
    else:
        V_dmg = np.load(os.path.join(ad_output, "first_damage_drift_all_100", "preprocessed_data_raw.npz"))["V"].astype(np.float32)
        V_dmg = V_dmg / (1 + SD_RATIO)  # undo old uniform drift
    np.savez(os.path.join(sd_damage_dir, "preprocessed_data_raw.npz"), V=apply_drift(V_dmg, SD_RATIO))

    SD_CONFIG = {
        "name": "Sensor_Drift",
        "description": "SD standard TL at drift=0.10 (non-uniform), N=400",
        "new_data_folder": "health_drift_nonuniform_2000",
        "test_damage_folder": "first_damage_drift_nonuniform_100",
        "tl_comparison": {"train_samples": 400, "freeze_strategy": "none"},
    }

    run_tl_comparison(
        scenario_name="Sensor_Drift",
        scenario_config=SD_CONFIG,
        pretrain_pth=PRETRAIN_PTH,
        settings=SETTINGS,
        device=device,
        output_subdir="tl_comparison_standard",
    )

    print("\n" + "=" * 60)
    print("[Done] All standard TL comparisons complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
