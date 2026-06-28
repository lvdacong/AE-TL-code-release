"""
AE_run_channel_ablation_all.py
================================
Run per-channel trajectory analysis for all three scenarios.
Generates fig_channel_ablation.png bar charts showing 252-channel
domain shift (rose) vs TL recovery (blue-gray).

Uses the latest trained models from difficulty_ablation directories.

Usage:
    cd script && python AE_run_channel_ablation_all.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

AD_OUTPUT = os.path.join(SCRIPT_DIR, "AD_preprocess_datasets_output")
AE_OUTPUT = os.path.join(SCRIPT_DIR, "AE_model_train_and_detect_output")
PRETRAIN_PTH = os.path.join(AE_OUTPUT, "Damage_Repaired", "pretrain", "autoencoder.pth")

AE_CONFIG = {
    "encoder_dims": [768, 384, 192],
    "latent_dim": 192,
    "decoder_dims": [192, 384, 768],
    "dropout": 0.0,
    "activation": "relu",
}

from AE_channel_ablation_auxiliary import run_channel_trajectory_analysis


def load_data(folder_name: str) -> np.ndarray:
    path = os.path.join(AD_OUTPUT, folder_name, "preprocessed_data_raw.npz")
    return np.load(path)["V"].astype(np.float32)


def _apply_nonuniform_drift(V: np.ndarray, ratio: float) -> np.ndarray:
    """Non-uniform drift matching AE_run_sd_difficulty_ablation.apply_drift."""
    from AE_run_sd_difficulty_ablation import CHANNEL_DRIFT_FACTORS
    D = V.shape[1]
    return V * (1.0 + ratio * CHANNEL_DRIFT_FACTORS[:D])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # =============================================
    # Scenario configs
    # =============================================
    scenarios = {
        "Damage_Repaired": {
            "difficulty_values": [0, 4, 8, 12, 16, 20],
            "data_sizes": [50, 100, 200, 400],
            "xlabel": "Number of Repaired Elements",
            "val_loader": lambda diff: (
                load_data("health_original_2000")[-200:]
                if diff == 0
                else load_data(f"damage_repaired_{diff}_original_500")[-200:]
            ),
        },
        "Sensor_Offset": {
            "difficulty_values": [0, 1, 2, 3, 4, 5],
            "data_sizes": [50, 100, 200, 400],
            "xlabel": "Number of Offset Sensors",
            "val_loader": lambda diff: (
                load_data("health_original_2000")[-200:]
                if diff == 0
                else load_data(f"health_offset_count_{diff}_2000")[-200:]
            ),
        },
        "Sensor_Drift": {
            "difficulty_values": [0, 0.05, 0.10, 0.15, 0.20, 0.25],
            "data_sizes": [50, 100, 200, 400],
            "xlabel": "Drift Ratio",
            "val_loader": lambda diff, _health=load_data("health_original_2000"): (
                _health[-200:] if diff == 0
                else _apply_nonuniform_drift(_health, diff)[-200:]
            ),
        },
    }

    for scenario_name, cfg in scenarios.items():
        print(f"\n{'='*60}")
        print(f"Channel ablation: {scenario_name}")
        print(f"{'='*60}")

        ablation_dir = os.path.join(AE_OUTPUT, scenario_name, "difficulty_ablation")
        model_dir = ablation_dir  # models are in diff_X_nY/autoencoder.pth

        # Check if models exist for the expected data sizes
        diff_test = cfg["difficulty_values"][-1]
        diff_str = f"{diff_test}".replace(".", "p")
        max_n = cfg["data_sizes"][-1]
        model_check = os.path.join(model_dir, f"diff_{diff_str}_n{max_n}", "autoencoder.pth")
        if not os.path.exists(model_check):
            print(f"  [WARNING] Model not found: {model_check}")
            print(f"  Falling back to available data sizes...")
            # Find available data sizes
            available = []
            for ds in cfg["data_sizes"]:
                p = os.path.join(model_dir, f"diff_{diff_str}_n{ds}", "autoencoder.pth")
                if os.path.exists(p):
                    available.append(ds)
            if available:
                cfg["data_sizes"] = available
                print(f"  Using data_sizes: {available}")
            else:
                print(f"  [SKIP] No models found for {scenario_name}")
                continue

        run_channel_trajectory_analysis(
            scenario_name=scenario_name,
            difficulty_values=cfg["difficulty_values"],
            data_sizes=cfg["data_sizes"],
            pretrain_model_path=PRETRAIN_PTH,
            model_dir=model_dir,
            val_data_loader=cfg["val_loader"],
            output_dir=ablation_dir,
            ae_config=AE_CONFIG,
            device=device,
            xlabel=cfg["xlabel"],
        )

    print(f"\n{'='*60}")
    print("[Done] All channel ablation analyses complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
