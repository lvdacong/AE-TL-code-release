"""
AE_gen_difflr_roc.py
=====================
Generate ROC comparison figure for three optimizer configurations
in the Sensor Offset scenario: DiffLR Adam vs Standard SGD vs Adam Uniform.

Usage:
    cd script && python AE_gen_difflr_roc.py
"""

from __future__ import annotations
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from AE_train_model_auxiliary import (
    Autoencoder, apply_style, FIG_DPI, PLOT_STYLE
)
from AE_model_train_and_detect_auxiliary import (
    compute_anomaly_scores, compute_detection_metrics
)

# -- Paths --
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_OUT   = os.path.join(SCRIPT_DIR, 'AE_model_train_and_detect_output')
BASE_PRE   = os.path.join(SCRIPT_DIR, 'AD_preprocess_datasets_output')

# Three optimizer model paths (Sensor_Offset scenario)
MODEL_PATHS = {
    'DiffLR Adam':   os.path.join(BASE_OUT, 'Sensor_Offset', 'tl_comparison', 'TL', 'autoencoder.pth'),
    'Standard SGD':  os.path.join(BASE_OUT, 'Sensor_Offset', 'tl_comparison_standard', 'TL', 'autoencoder.pth'),
    'Adam Uniform':  os.path.join(BASE_OUT, 'Sensor_Offset', 'tl_comparison_adam_uniform', 'TL', 'autoencoder.pth'),
}

# Data paths
CONTROL_FOLDER = 'health_offset_count_1_2000'
TEST_FOLDER    = 'first_damage_offset_count_1_100'
TRAIN_SAMPLES  = 400
VAL_SAMPLES    = 100  # holdout: V[400:500]

# Model architecture (must match training)
AE_CONFIG = dict(encoder_dims=[768, 384, 192], latent_dim=192,
                 decoder_dims=[192, 384, 768], dropout=0.0, activation='relu')

# Morandi palette - three distinct colours
COLORS = {
    'DiffLR Adam':  '#7BA7BC',  # muted teal
    'Standard SGD': '#C97A6C',  # muted terracotta
    'Adam Uniform': '#8B9E77',  # muted sage green
}
LINESTYLES = {
    'DiffLR Adam':  '-',
    'Standard SGD': '--',
    'Adam Uniform': ':',
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_ae(pth: str, D: int = 768) -> Autoencoder:
    m = Autoencoder(D, AE_CONFIG['encoder_dims'], AE_CONFIG['latent_dim'],
                    AE_CONFIG['decoder_dims'], AE_CONFIG['dropout'],
                    AE_CONFIG['activation']).to(device)
    m.load_state_dict(torch.load(pth, map_location=device))
    m.eval()
    return m


def main():
    # Load test data (damaged samples)
    test_path = os.path.join(BASE_PRE, TEST_FOLDER, 'preprocessed_data_raw.npz')
    V_test = np.load(test_path)['V'].astype(np.float32)

    # Load healthy control: V[400:500] (holdout, consistent with AE_gen_roc_and_summary.py)
    ctrl_path = os.path.join(BASE_PRE, CONTROL_FOLDER, 'preprocessed_data_raw.npz')
    V_full = np.load(ctrl_path)['V'].astype(np.float32)
    V_ctrl = V_full[TRAIN_SAMPLES:TRAIN_SAMPLES + VAL_SAMPLES]

    D = V_ctrl.shape[1]
    print(f"Test: {V_test.shape}, Control: {V_ctrl.shape} (holdout[{TRAIN_SAMPLES}:{TRAIN_SAMPLES + VAL_SAMPLES}])")

    # Compute ROC for each optimizer
    roc_data = {}  # name -> (fprs, tprs, auc)
    model_order = ['DiffLR Adam', 'Standard SGD', 'Adam Uniform']

    for name in model_order:
        pth = MODEL_PATHS[name]
        if not os.path.exists(pth):
            print(f"  [skip] {name}: {pth} not found")
            continue
        model = load_ae(pth, D)
        sd = compute_anomaly_scores(V_test, model, device, method='max')
        sc = compute_anomaly_scores(V_ctrl, model, device, method='max')
        m  = compute_detection_metrics(sd, sc)
        roc_data[name] = (m['fprs'], m['tprs'], m['auc'])
        print(f"  {name}: AUC={m['auc']:.4f} [{m['auc_ci_lo']:.4f}, {m['auc_ci_hi']:.4f}]")

    # -- Plot: Type C square ROC --
    roc_style = PLOT_STYLE.copy()
    roc_style.update({
        "font.size": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 12,
    })
    apply_style(roc_style)

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.8), constrained_layout=True)

    for name in model_order:
        if name not in roc_data:
            continue
        fprs, tprs, auc = roc_data[name]
        ax.plot(fprs, tprs,
                label=name,
                color=COLORS[name], lw=1.5, linestyle=LINESTYLES[name])

    # Diagonal reference line (random classifier)
    ax.plot([0, 1], [0, 1], color='#cccccc', lw=0.8, linestyle='--')
    ax.axvline(x=0.05, color='#999999', lw=0.8, linestyle='--')

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(direction='in')
    ax.grid(False)

    # Legend outside upper center (2 rows to fit within figure width)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="outside upper center",
               ncol=2, frameon=False, fontsize=12)

    out_path = os.path.join(BASE_OUT, 'Sensor_Offset', 'fig_difflr_roc.png')
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"\n[saved] {os.path.relpath(out_path, SCRIPT_DIR)}")

    apply_style(PLOT_STYLE)


if __name__ == '__main__':
    main()
