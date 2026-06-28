"""
AE_run_pretrain.py
===================
Train the pretrain autoencoder on health_original_2000.
All AE_run_* experiment scripts depend on this model.

Usage:
    cd script && python AE_run_pretrain.py
"""

from __future__ import annotations

import os
import sys
import random

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from AE_train_model_auxiliary import (
    load_data_from_path, train_model,
    FIG_DPI, PLOT_STYLE,
)

AD_OUTPUT = os.path.join(SCRIPT_DIR, "AD_preprocess_datasets_output")
AE_OUTPUT = os.path.join(SCRIPT_DIR, "AE_model_train_and_detect_output")
PRETRAIN_DIR = os.path.join(AE_OUTPUT, "Damage_Repaired", "pretrain")
PRETRAIN_PTH = os.path.join(PRETRAIN_DIR, "autoencoder.pth")

# Training config (matches all experiments)
AE_CONFIG = {
    "encoder_dims": [768, 384, 192],
    "latent_dim": 192,
    "decoder_dims": [192, 384, 768],
    "dropout": 0.0,
    "activation": "relu",
}

TRAINING_CONFIG = {
    "pretrain": {"epochs": 2000, "lr": 3e-4, "batch_size": 256},
    "val_samples": 200,
    "weight_decay": 0.0,
    "grad_clip": 1.0,
    "use_lr_scheduler": True,
    "lr_scheduler_type": "cosine",
    "cosine_eta_min": 1e-6,
    "shuffle": True,
}


def main():
    print("=" * 60)
    print("Pretrain Autoencoder (health_original_2000)")
    print("=" * 60)

    # Seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Check if already exists
    if os.path.exists(PRETRAIN_PTH):
        print(f"[pretrain] Model already exists: {PRETRAIN_PTH}")
        print(f"[pretrain] Delete it first if you want to retrain.")
        return

    # Load data
    data_path = os.path.join(AD_OUTPUT, "health_original_2000")
    V_train, V_val = load_data_from_path(
        data_path, TRAINING_CONFIG["val_samples"], "pretrain data"
    )

    # Train
    os.makedirs(PRETRAIN_DIR, exist_ok=True)
    pt = TRAINING_CONFIG["pretrain"]

    model, train_losses, val_losses = train_model(
        V_train, V_val, device,
        epochs=pt["epochs"],
        lr=pt["lr"],
        batch_size=pt["batch_size"],
        output_dir=PRETRAIN_DIR,
        ae_config=AE_CONFIG,
        training_config=TRAINING_CONFIG,
        plot_style=PLOT_STYLE,
        fig_dpi=FIG_DPI,
        pretrain_model_path=None,
        model_name="Pre-train",
        train_shuffle=True,
    )

    print(f"\n[done] Pretrain model saved: {PRETRAIN_PTH}")
    print(f"  final val_loss: {val_losses[-1]:.6f}")


if __name__ == "__main__":
    main()
