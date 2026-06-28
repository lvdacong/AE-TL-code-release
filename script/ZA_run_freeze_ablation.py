"""
ZA_run_freeze_ablation.py
=========================
Standalone script to run freeze strategy ablation on the Structural Repair
scenario (N_r=12, 400 samples, SGD+momentum, 1000 epochs).
Outputs training curve comparison figure for the paper.

Usage:
    cd script && python ZA_run_freeze_ablation.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(__file__))
from AE_train_model_auxiliary import (
    Autoencoder, load_data_from_path, apply_freeze_strategy,
    plot_freeze_strategy_comparison, PLOT_STYLE, FIG_DPI, apply_style
)

# ── Configuration ──────────────────────────────────────────
PRETRAIN_PTH = "AE_model_train_and_detect_output/Damage_Repaired/pretrain/autoencoder.pth"
NEW_DATA_FOLDER = "AD_preprocess_datasets_output/damage_repaired_12_original_500"
OUTPUT_DIR = "AE_model_train_and_detect_output/Damage_Repaired/freeze_ablation"

AE_CONFIG = {
    "encoder_dims": [768, 384, 192],
    "latent_dim": 192,
    "decoder_dims": [192, 384, 768],
    "dropout": 0.0,
    "activation": "relu",
}

TRAIN_SAMPLES = 400
VAL_SAMPLES = 100
EPOCHS = 1000
LR = 1e-3
BATCH_SIZE = 256
MOMENTUM = 0.9
COSINE_ETA_MIN = 1e-6
SEED = 42
STRATEGIES = ["none", "bottom", "encoder"]

STRATEGY_DISPLAY = {
    "none": "Full Fine-tuning",
    "bottom": "Freeze Bottom",
    "encoder": "Freeze Encoder",
}


def train_with_freeze(V_train, V_val, pretrain_pth, strategy, device):
    """Train one model with the given freeze strategy, return (train_losses, val_losses)."""
    D = V_train.shape[1]

    # Create model and load pretrained weights
    model = Autoencoder(
        D,
        AE_CONFIG["encoder_dims"],
        AE_CONFIG["latent_dim"],
        AE_CONFIG["decoder_dims"],
        AE_CONFIG["dropout"],
        AE_CONFIG["activation"],
    ).to(device)
    ckpt = torch.load(pretrain_pth, map_location=device)
    model.load_state_dict(ckpt)

    # Apply freeze strategy BEFORE optimizer setup
    apply_freeze_strategy(model, strategy, verbose=True)

    # Build optimizer only on trainable params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(trainable_params, lr=LR, momentum=MOMENTUM)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=COSINE_ETA_MIN)

    X_train = torch.from_numpy(V_train).to(device)
    X_val = torch.from_numpy(V_val).to(device)
    n_train = X_train.shape[0]
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        # ── Training ──
        model.train()
        if True:  # shuffle
            perm = torch.randperm(n_train, device=device)
            X_train_shuffled = X_train[perm]
        else:
            X_train_shuffled = X_train

        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n_train, BATCH_SIZE):
            batch = X_train_shuffled[i : i + BATCH_SIZE]
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_losses.append(epoch_loss / n_batches)

        # ── Validation ──
        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_loss = criterion(val_out, X_val).item()
        val_losses.append(val_loss)

        scheduler.step()

        if epoch % 200 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{EPOCHS} | train={train_losses[-1]:.6f} val={val_loss:.6f}")

    return train_losses, val_losses


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    pretrain_pth = os.path.join(script_dir, PRETRAIN_PTH)
    new_data_path = os.path.join(script_dir, NEW_DATA_FOLDER)
    output_dir = os.path.join(script_dir, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    V_train, V_val = load_data_from_path(new_data_path, VAL_SAMPLES, "DR-freeze")
    V_train = V_train[:TRAIN_SAMPLES]
    print(f"[Data] train={V_train.shape[0]}, val={V_val.shape[0]}, D={V_train.shape[1]}")

    # Run all strategies
    strategy_curves = {}
    for strategy in STRATEGIES:
        display = STRATEGY_DISPLAY[strategy]
        print(f"\n{'='*60}")
        print(f"[Strategy] {display} ({strategy})")
        print(f"{'='*60}")

        # Fix seed per strategy for reproducibility
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        train_losses, val_losses = train_with_freeze(
            V_train, V_val, pretrain_pth, strategy, device
        )
        strategy_curves[display] = (train_losses, val_losses)
        print(f"  Final val_loss = {val_losses[-1]:.6f}")

        # Save individual CSV
        import pandas as pd
        strat_dir = os.path.join(output_dir, f"freeze_{strategy}")
        os.makedirs(strat_dir, exist_ok=True)
        pd.DataFrame({"train_loss": train_losses, "val_loss": val_losses}).to_csv(
            os.path.join(strat_dir, "training_losses.csv"), index=False
        )

    # Generate comparison figure
    print(f"\n[Plot] Generating freeze strategy comparison...")
    plot_freeze_strategy_comparison(strategy_curves, output_dir)
    print(f"[Done] Output: {output_dir}")


if __name__ == "__main__":
    main()
