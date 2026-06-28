"""
AE_gen_difflr_comparison.py
============================
Generate differential-LR comparison figures for the paper (Section 5).

Two outputs (Sensor Offset scenario, three optimizer configs):
  1. fig_difflr_training_curves.png  -- val-loss overlay (fine-tuning phase)
  2. fig_difflr_3d_comparison.png    -- 2x3 3D residual rendering

Usage:
    cd script && python AE_gen_difflr_comparison.py
"""
from __future__ import annotations

import os
import shutil
import sys

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from AE_train_model_auxiliary import Autoencoder, apply_style, PLOT_STYLE, FIG_DPI
from AE_model_train_and_detect_auxiliary import (
    compute_residuals, interpolate_to_nodes, render_damage_3d,
    merge_images_grid, HAS_PYVISTA,
    parse_elsets_from_inp, extract_middlewhole_submesh,
    load_camera_position
)

if HAS_PYVISTA:
    import pyvista as pv

# Custom camera for Sensor_Offset 3D figures (Fig 19 & 23 in paper)
# Interactively set by user on 2026-04-14; shared with AE_gen_comparison_figures.py
_SO_CUSTOM_CAMERA = [
    [-135048.78245139707, 46620.556215902245, 64682.97864052804],
    [13047.761038422219, 13099.113883212714, 42711.431058467],
    [0.21794191415760136, 0.9757631683238296, -0.019686579082266065],
]

# ── paths ──
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_OUT     = os.path.join(SCRIPT_DIR, 'AE_model_train_and_detect_output')
BASE_PRE     = os.path.join(SCRIPT_DIR, 'AD_preprocess_datasets_output')
SO_DIR       = os.path.join(BASE_OUT, 'Sensor_Offset')

# ── model config ──
AE_CONFIG = dict(encoder_dims=[768, 384, 192], latent_dim=192,
                 decoder_dims=[192, 384, 768], dropout=0.0, activation='relu')
TRAIN_SAMPLES = 400
VAL_SAMPLES   = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── three optimizer configurations ──
CONFIGS = [
    {
        'label': 'Standard SGD',
        'subdir': 'tl_comparison_standard',
    },
    {
        'label': 'Adam Uniform',
        'subdir': 'tl_comparison_adam_uniform',
    },
    {
        'label': 'DiffLR Adam',
        'subdir': 'tl_comparison',
    },
]

# ── Morandi palette ──
COLORS = {
    'Standard SGD':  '#A8B5A2',   # sage
    'Adam Uniform':  '#8EAABD',   # blue-gray
    'DiffLR Adam':   '#C97A6C',   # dusty rose (adopted)
}
LW = {
    'Standard SGD':  1.2,
    'Adam Uniform':  1.2,
    'DiffLR Adam':   1.8,
}

# ── SO data folders ──
CONTROL_FOLDER = 'health_offset_count_1_2000'
TEST_FOLDER    = 'first_damage_offset_count_1_100'


def load_ae(pth: str, D: int = 252) -> Autoencoder:
    m = Autoencoder(D, AE_CONFIG['encoder_dims'], AE_CONFIG['latent_dim'],
                    AE_CONFIG['decoder_dims'], AE_CONFIG['dropout'],
                    AE_CONFIG['activation']).to(device)
    m.load_state_dict(torch.load(pth, map_location=device))
    m.eval()
    return m


def load_npz_all(folder: str) -> np.ndarray:
    path = os.path.join(BASE_PRE, folder, 'preprocessed_data_raw.npz')
    return np.load(path)['V'].astype(np.float32)


# ═══════════════════════════════════════════
#  Figure 1: training curve overlay
# ═══════════════════════════════════════════
def generate_training_curves():
    """Val-loss overlay for three optimizer configs (fine-tuning phase only)."""
    apply_style(PLOT_STYLE)

    # figsize 高度 3.6 预留顶部外部图例空间
    fig, ax = plt.subplots(figsize=(10, 3.6), constrained_layout=True)

    for cfg in CONFIGS:
        csv_path = os.path.join(SO_DIR, cfg['subdir'], 'TL', 'training_losses.csv')
        if not os.path.exists(csv_path):
            print(f"  [SKIP] Missing: {csv_path}")
            return
        df = pd.read_csv(csv_path)
        val_loss = df['val_loss'].values
        epochs = np.arange(1, len(val_loss) + 1)
        label = cfg['label']
        ax.plot(epochs, val_loss, label=label,
                color=COLORS[label], linewidth=LW[label])

    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss (MSE)')
    ax.tick_params(direction='in', which='both')

    # 图框外部上方横排图例
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="outside upper center",
               ncol=len(labels),
               frameon=False, fontsize=12)

    out_path = os.path.join(SO_DIR, 'fig_difflr_training_curves.png')
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════
#  Figure 2: 3D residual rendering comparison
# ═══════════════════════════════════════════
def generate_3d_comparison():
    """2-row x 3-col 3D residual renders (damaged / healthy) x 3 configs."""
    # 3D rendering assets
    ac_out = os.path.join(SCRIPT_DIR, 'AC_convert_and_extract_output')
    vtu_path      = os.path.join(ac_out, 'whole_from_inp.vtu')
    id_map_path   = os.path.join(ac_out, 'abaqus_id_to_vtu_index.csv')
    measures_path = os.path.join(ac_out, 'measures_ID_original.csv')
    camera_path   = os.path.join(SCRIPT_DIR, 'camera_position.json')
    inp_path      = "C:/SHM_abaqus_models/health.inp"

    for p in [vtu_path, id_map_path, measures_path, camera_path, inp_path]:
        if not os.path.exists(p):
            print(f"  [SKIP] Missing: {p}")
            return

    base_mesh = pv.read(vtu_path)
    id_df = pd.read_csv(id_map_path)
    id_mapping = dict(zip(id_df['abaqus_id'], id_df['vtu_index']))
    measures_df = pd.read_csv(measures_path, header=0)
    measure_ids = measures_df.iloc[:, 0].astype(int).tolist()

    elsets = parse_elsets_from_inp(inp_path, 'middlewhole')
    middlewhole_ids = elsets.get('middlewhole', [])
    middlewhole_mesh, _ = extract_middlewhole_submesh(base_mesh, middlewhole_ids, id_mapping)
    camera_position = _SO_CUSTOM_CAMERA

    # Load data
    V_test = load_npz_all(TEST_FOLDER)
    V_ctrl_all = load_npz_all(CONTROL_FOLDER)
    n_control = V_test.shape[0]
    pure_start = TRAIN_SAMPLES
    V_control = V_ctrl_all[pure_start:pure_start + n_control]
    print(f"  V_test: {V_test.shape}, V_control: {V_control.shape}")

    # Pre-compute node values for all configs to determine shared color scales
    all_nv_d = []   # damage node values per config
    all_nv_c = []   # control node values per config
    models = []

    for cfg in CONFIGS:
        pth = os.path.join(SO_DIR, cfg['subdir'], 'TL', 'autoencoder.pth')
        if not os.path.exists(pth):
            print(f"  [SKIP] Missing: {pth}")
            return
        model = load_ae(pth)
        models.append(model)

        res_d = compute_residuals(V_test, model, device)
        res_c = compute_residuals(V_control, model, device)
        scores_d = np.abs(res_d).mean(axis=0)
        scores_c = np.abs(res_c).mean(axis=0)

        nv_d = interpolate_to_nodes(base_mesh, middlewhole_mesh, scores_d, measure_ids, id_mapping)
        nv_c = interpolate_to_nodes(base_mesh, middlewhole_mesh, scores_c, measure_ids, id_mapping)
        all_nv_d.append(nv_d)
        all_nv_c.append(nv_c)

    # Shared color scale per row
    clim_d = max(nv.max() for nv in all_nv_d)
    clim_c = max(nv.max() for nv in all_nv_c)
    clim_d = max(clim_d, 0.5)
    clim_c = max(clim_c, 0.5)

    # Render individual PNGs
    temp_dir = os.path.join(SO_DIR, '_temp_difflr_3d')
    os.makedirs(temp_dir, exist_ok=True)

    damage_images = []
    control_images = []
    col_titles = []

    for i, cfg in enumerate(CONFIGS):
        label = cfg['label']
        col_titles.append(label)

        d_path = os.path.join(temp_dir, f"{i}_damage.png")
        render_damage_3d(base_mesh, middlewhole_mesh, all_nv_d[i],
                         d_path, camera_position, override_clim_max=clim_d)
        damage_images.append(d_path)

        c_path = os.path.join(temp_dir, f"{i}_control.png")
        render_damage_3d(base_mesh, middlewhole_mesh, all_nv_c[i],
                         c_path, camera_position, override_clim_max=clim_c)
        control_images.append(c_path)

    # Merge into 2x3 grid
    out_path = os.path.join(SO_DIR, 'fig_difflr_3d_comparison.png')
    merge_images_grid(
        [damage_images, control_images],
        out_path,
        col_titles,
        row_titles=["Test (Damaged)", "Control (Healthy)"]
    )
    print(f"  Saved: {out_path}")

    # Cleanup temp
    shutil.rmtree(temp_dir, ignore_errors=True)


# ═══════════════════════════════════════════
#  main
# ═══════════════════════════════════════════
def main():
    print("=" * 60)
    print("  Generating DiffLR comparison figures (SO scenario)")
    print("=" * 60)

    print("\n[1/2] Training curves ...")
    generate_training_curves()

    if HAS_PYVISTA:
        print("\n[2/2] 3D residual comparison ...")
        generate_3d_comparison()
    else:
        print("\n[2/2] [SKIP] PyVista not available for 3D rendering")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == '__main__':
    main()
