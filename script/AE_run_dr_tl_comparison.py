"""
AE_run_dr_tl_comparison.py
===========================
TL comparison for the Structural Repair scenario (N_r=12, N=400).
Compares TL, OldModel, and FromScratch on damage_repaired_12 data.

Usage:
    cd script && python AE_run_dr_tl_comparison.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import torch

# ========================================
# 路径设置
# ========================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

AD_OUTPUT = os.path.join(SCRIPT_DIR, "AD_preprocess_datasets_output")
AE_OUTPUT = os.path.join(SCRIPT_DIR, "AE_model_train_and_detect_output")
AC_OUTPUT = os.path.join(SCRIPT_DIR, "AC_convert_and_extract_output")

PRETRAIN_PTH = os.path.join(AE_OUTPUT, "Damage_Repaired", "pretrain", "autoencoder.pth")

# Typical difficulty = 12 units repaired, N = 400 training samples
DIFFICULTY = 12
TRAIN_SAMPLES = 400
FREEZE_STRATEGY = "none"

# Folder names for preprocessed data
NEW_DATA_FOLDER = f"damage_repaired_{DIFFICULTY}_original_500"
TEST_DAMAGE_FOLDER = f"second_damage_{DIFFICULTY}_original_100"

# ========================================
# 构建 settings 和 scenario_config
# ========================================

# 这些字段模拟 AE_settings.json 的结构, 供 run_tl_comparison 使用
WORKSPACE_DIR = os.path.dirname(SCRIPT_DIR)

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
        "pretrain": {
            "epochs": 2000,
            "lr": 3e-4,
            "batch_size": 256,
        },
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
        "l2_sp_alpha": 0.005,
    },
    "detection": {
        "enabled": True,
        "render_3d": False,
        "inp_file_path": f"C:/SHM_abaqus_models/second_damage_{DIFFICULTY}.inp",
        "middlewhole_elset": "middlewhole",
        "camera_config_path": "script/camera_position.json",
        "ac_output_dir": "script/AC_convert_and_extract_output",
        "vtu_file": "whole_from_inp.vtu",
        "id_mapping_file": "abaqus_id_to_vtu_index.csv",
        "measures_file": "measures_ID_original.csv",
    },
}

SCENARIO_CONFIG = {
    "name": "Damage_Repaired",
    "description": f"DR tl_comparison at difficulty={DIFFICULTY}, N={TRAIN_SAMPLES}",
    "new_data_folder": NEW_DATA_FOLDER,
    "test_damage_folder": TEST_DAMAGE_FOLDER,
    "tl_comparison": {
        "train_samples": TRAIN_SAMPLES,
        "freeze_strategy": FREEZE_STRATEGY,
    },
}


def main():
    print("=" * 60)
    print(f"Damage_Repaired TL Comparison (difficulty={DIFFICULTY}, N={TRAIN_SAMPLES})")
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

    # Verify preprocessed data
    new_data_path = os.path.join(AD_OUTPUT, NEW_DATA_FOLDER, "preprocessed_data_raw.npz")
    test_data_path = os.path.join(AD_OUTPUT, TEST_DAMAGE_FOLDER, "preprocessed_data_raw.npz")

    if not os.path.exists(new_data_path):
        print(f"[error] New data not found: {new_data_path}")
        print(f"[hint] Run AD preprocessing first")
        sys.exit(1)
    if not os.path.exists(test_data_path):
        print(f"[error] Test data not found: {test_data_path}")
        print(f"[hint] Run AD preprocessing first")
        sys.exit(1)

    d1 = np.load(new_data_path)
    d2 = np.load(test_data_path)
    print(f"[data] new_data: {d1['V'].shape}")
    print(f"[data] test_data: {d2['V'].shape}")

    # Run tl_comparison
    from AE_tl_comparison_auxiliary import run_tl_comparison

    results = run_tl_comparison(
        scenario_name="Damage_Repaired",
        scenario_config=SCENARIO_CONFIG,
        pretrain_pth=PRETRAIN_PTH,
        settings=SETTINGS,
        device=device,
    )

    # Summary
    print(f"\n{'=' * 60}")
    print("[Results Summary]")
    print(f"{'=' * 60}")
    for key, result in results.items():
        val_losses = result.get("val_losses", [])
        final_loss = val_losses[-1] if val_losses else float("nan")
        print(f"  {key}: final_val_loss = {final_loss:.6f}")

    output_dir = os.path.join(AE_OUTPUT, "Damage_Repaired", "tl_comparison")
    print(f"\n[done] Output: {output_dir}")


if __name__ == "__main__":
    main()
