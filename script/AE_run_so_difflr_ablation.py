"""
AE_run_so_difflr_ablation.py
==============================
Ablation: Adam with UNIFORM learning rate on the Sensor Relocation scenario.
This isolates the differential-LR effect by comparing:
  - Adam Uniform (lr=1e-3 for all params, noise=0.005)  ← this script
  - DiffLR Adam  (enc=1e-5, dec=1e-3, noise=0.005)      ← existing tl_comparison/
  - Standard SGD (lr=1e-3 uniform, noise=0)              ← existing tl_comparison_standard/

The only difference between Adam Uniform and DiffLR Adam is the encoder LR,
making this a single-variable ablation of the differential learning rate strategy.

Usage:
    cd script && python AE_run_so_difflr_ablation.py
"""
from __future__ import annotations

import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from AE_tl_comparison_auxiliary import run_tl_comparison

PRETRAIN_PTH = os.path.join(
    SCRIPT_DIR, "AE_model_train_and_detect_output",
    "Damage_Repaired", "pretrain", "autoencoder.pth",
)
WORKSPACE_DIR = os.path.dirname(SCRIPT_DIR)

# Adam Uniform config: same as DiffLR Adam EXCEPT encoder_lr is absent (uniform)
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
        "optimizer_type": "adam",       # same optimizer as DiffLR
        # NO encoder_lr → uniform lr=1e-3 for ALL params
        "noise_std": 0.005,            # match DiffLR config to control this variable
        "l2_sp_alpha": 0.0,            # match SO DiffLR config (already 0 for SO)
        "sgd_momentum": 0.9,
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

SO_CONFIG = {
    "name": "Sensor_Offset",
    "description": "SO Adam-Uniform ablation at offset_count=1 (max-stress-diff), N=400",
    "new_data_folder": "health_offset_count_1_2000",
    "test_damage_folder": "first_damage_offset_count_1_100",
    "tl_comparison": {"train_samples": 400, "freeze_strategy": "none"},
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    if not os.path.exists(PRETRAIN_PTH):
        print(f"[error] Pretrain model not found: {PRETRAIN_PTH}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("SO DiffLR Ablation: Adam Uniform (lr=1e-3 all params, noise=0.005)")
    print("=" * 60)

    run_tl_comparison(
        scenario_name="Sensor_Offset",
        scenario_config=SO_CONFIG,
        pretrain_pth=PRETRAIN_PTH,
        settings=SETTINGS,
        device=device,
        output_subdir="tl_comparison_adam_uniform",
    )

    print("\n" + "=" * 60)
    print("[Done] Adam Uniform ablation on SO complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
