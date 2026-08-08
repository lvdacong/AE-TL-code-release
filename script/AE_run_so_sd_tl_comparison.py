"""
AE_run_so_sd_tl_comparison.py
==============================
TL comparison for the Sensor Relocation and Temperature-conditioned Sensor
Response scenarios.
Runs full TL/OldModel/FromScratch training and visualization.

Usage:
    cd script && python AE_run_so_sd_tl_comparison.py
"""

from __future__ import annotations

import os
import sys
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from AE_tl_comparison_auxiliary import run_tl_comparison
from AE_run_dr_tl_comparison import SETTINGS  # reuse DR settings template
from AE_run_temperature_sd_tl_comparison import run_sensor_drift_comparison

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
    # Sensor_Drift: temperature-conditioned nonlinear response, N=400
    # ========================================
    print("\n" + "=" * 60)
    print("Sensor_Drift tl_comparison (temperature span=20 C, N=400)")
    print("=" * 60)
    sd_results = run_sensor_drift_comparison(device)

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
