"""
AE_gen_comparison_figures.py
=============================
Regenerate comparison figures (residuals, training curves, 3D) for all
scenarios from saved weights and CSVs. No retraining needed.

Usage:
    cd script && python AE_gen_comparison_figures.py
"""

from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from AE_train_model_auxiliary import (
    Autoencoder, apply_style, PLOT_STYLE, FIG_DPI,
    plot_dual_comparison_curves
)
from AE_model_train_and_detect_auxiliary import (
    compute_residuals, compute_normal_params, merge_images_grid,
    HAS_PYVISTA, render_damage_3d, load_camera_position,
    parse_elsets_from_inp, extract_middlewhole_submesh,
    interpolate_to_nodes, VIS_DPI
)

# Conditionally import from AE_tl_comparison_auxiliary
from AE_tl_comparison_auxiliary import _plot_comparison_residuals_grid

if HAS_PYVISTA:
    import pyvista as pv

# -- paths --
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_OUT     = os.path.join(SCRIPT_DIR, 'AE_model_train_and_detect_output')
BASE_PRE     = os.path.join(SCRIPT_DIR, 'AD_preprocess_datasets_output')
PRETRAIN_PTH = os.path.join(BASE_OUT, 'Damage_Repaired', 'pretrain', 'autoencoder.pth')

# -- model config --
AE_CONFIG = dict(encoder_dims=[768, 384, 192], latent_dim=192,
                 decoder_dims=[192, 384, 768], dropout=0.0, activation='relu')
VAL_SAMPLES = 100  # must match tl_comparison run-script training config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -- scenario configuration --
# IMPORTANT: control_folder/test_folder must match the actual TL run scripts
# (AE_run_dr_tl_comparison.py / AE_run_so_sd_tl_comparison.py). Earlier versions
# of this file referenced stale folder names (health_offset_2_2000,
# health_drift_all_2000) from a previous experiment iteration, which produced
# residuals computed on data the models had never seen.
SCENARIOS = [
    dict(
        name='Damage_Repaired',
        figure_label='Local Stiffening',
        control_folder='damage_repaired_12_original_500',
        test_folder='second_damage_12_original_100',
        train_samples=400,
        paper_prefix='fig_dr',
    ),
    dict(
        name='Sensor_Offset',
        figure_label='Relocation',
        control_folder='health_offset_count_1_2000',
        test_folder='first_damage_offset_count_1_100',
        train_samples=400,
        paper_prefix='fig_so',
    ),
    dict(
        name='Sensor_Drift',
        figure_label='Temp. response',
        control_folder='health_temperature_response_span20_2000',
        test_folder='first_damage_temperature_response_span20_100',
        train_samples=400,
        paper_prefix='fig_sd',
    ),
]


def load_ae(pth: str, D: int = 252) -> Autoencoder:
    m = Autoencoder(D, AE_CONFIG['encoder_dims'], AE_CONFIG['latent_dim'],
                    AE_CONFIG['decoder_dims'], AE_CONFIG['dropout'],
                    AE_CONFIG['activation']).to(device)
    m.load_state_dict(torch.load(pth, map_location=device))
    m.eval()
    return m


def load_npz_last(folder: str, n: int) -> np.ndarray:
    path = os.path.join(BASE_PRE, folder, 'preprocessed_data_raw.npz')
    V = np.load(path)['V'].astype(np.float32)
    return V[-n:]


def load_npz_all(folder: str) -> np.ndarray:
    path = os.path.join(BASE_PRE, folder, 'preprocessed_data_raw.npz')
    return np.load(path)['V'].astype(np.float32)


def regen_scenario(scenario: dict) -> None:
    name = scenario['name']
    tl_dir = os.path.join(BASE_OUT, name, 'tl_comparison')

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # Load models
    tl_pth = os.path.join(tl_dir, 'TL', 'autoencoder.pth')
    fs_pth = os.path.join(tl_dir, 'fromscratch', 'autoencoder.pth')

    V_ctrl_all = load_npz_all(scenario['control_folder'])
    V_test = load_npz_all(scenario['test_folder'])
    D = V_ctrl_all.shape[1]

    # Unify V_control sample count with V_test so that the 2x3 heatmap grid
    # is visually comparable across scenarios (previously DR had 100 control
    # rows while SO/SD had 1500, making the "Control (Healthy)" row look
    # structurally different for reasons unrelated to the physics).
    train_samples = scenario['train_samples']
    N = V_ctrl_all.shape[0]
    n_control = V_test.shape[0]
    pure_start = train_samples
    pure_end = N - VAL_SAMPLES
    if pure_end - pure_start >= n_control:
        # Prefer the pure holdout region (disjoint from train and val).
        V_control = V_ctrl_all[pure_start:pure_start + n_control]
        ctrl_src = f"holdout[{pure_start}:{pure_start + n_control}]"
    else:
        # Data-limited case (e.g. DR: N=500, train=400, val=100). Fall back
        # to V_val, which is held-out healthy data that never entered any
        # gradient step and is therefore a valid control source.
        V_control = V_ctrl_all[N - VAL_SAMPLES:N - VAL_SAMPLES + n_control]
        ctrl_src = f"V_val[{N - VAL_SAMPLES}:{N - VAL_SAMPLES + n_control}]"

    print(f"  Test: {V_test.shape}, Control: {V_control.shape} ({ctrl_src})")

    pretrain_model = load_ae(PRETRAIN_PTH, D)
    tl_model = load_ae(tl_pth, D)
    fs_model = load_ae(fs_pth, D)

    models = {
        'pretrain': pretrain_model,
        'tl': tl_model,
        'fromscratch': fs_model,
    }
    model_names = {
        'pretrain': 'Unadapted source',
        'tl': 'Transfer learning',
        'fromscratch': 'Scratch-trained',
    }

    # =========================================
    # 1. Residuals comparison (2x3 grid)
    # =========================================
    print("\n  [1/3] Regenerating comparison_residuals.png ...")
    all_residuals = {}
    column_titles = []
    for key in ['pretrain', 'tl', 'fromscratch']:
        column_titles.append(model_names[key])
        res_damage = compute_residuals(V_test, models[key], device)
        res_control = compute_residuals(V_control, models[key], device)
        all_residuals[key] = {'damage': res_damage, 'control': res_control}

    residuals_out = os.path.join(tl_dir, 'comparison_residuals.png')
    _plot_comparison_residuals_grid(all_residuals, column_titles, residuals_out)
    print(f"    Saved: {residuals_out}")

    # =========================================
    # 2. Training curve comparison
    # =========================================
    print("  [2/3] Regenerating training_curve_comparison.png ...")

    pretrain_csv = os.path.join(BASE_OUT, 'Damage_Repaired', 'pretrain', 'training_losses.csv')
    tl_csv = os.path.join(tl_dir, 'TL', 'training_losses.csv')
    fs_csv = os.path.join(tl_dir, 'fromscratch', 'training_losses.csv')

    if all(os.path.exists(c) for c in [pretrain_csv, tl_csv, fs_csv]):
        pt_df = pd.read_csv(pretrain_csv)
        tl_df = pd.read_csv(tl_csv)
        fs_df = pd.read_csv(fs_csv)

        plot_dual_comparison_curves(
            old_val_losses=pt_df['val_loss'].tolist(),
            transfer_val_losses=tl_df['val_loss'].tolist(),
            scratch_val_losses=fs_df['val_loss'].tolist(),
            output_dir=tl_dir,
            plot_style=PLOT_STYLE,
            fig_dpi=FIG_DPI,
            filename="training_curve_comparison.png",
            old_train_losses=pt_df['train_loss'].tolist(),
            transfer_train_losses=tl_df['train_loss'].tolist(),
            scratch_train_losses=fs_df['train_loss'].tolist(),
        )
        print(f"    Saved: {os.path.join(tl_dir, 'training_curve_comparison.png')}")
    else:
        print("    [SKIP] Missing CSV files for training curves")

    # =========================================
    # 3. 3D rendering comparison
    # =========================================
    if HAS_PYVISTA:
        print("  [3/3] Regenerating comparison_3d.png ...")
        regen_3d(scenario, models, model_names, V_test, V_control, tl_dir)
    else:
        print("  [3/3] [SKIP] PyVista not available for 3D rendering")

# Fixed camera for Sensor Relocation 3D figures; other scenarios use
# camera_position.json.
_SO_CUSTOM_CAMERA = [
    [-135048.78245139707, 46620.556215902245, 64682.97864052804],
    [13047.761038422219, 13099.113883212714, 42711.431058467],
    [0.21794191415760136, 0.9757631683238296, -0.019686579082266065],
]


def regen_3d(scenario, models, model_names, V_test, V_control, tl_dir):
    """Regenerate comparison_3d.png with shared row-wise colour scales."""
    # Load 3D rendering assets
    ac_out = os.path.join(SCRIPT_DIR, 'AC_convert_and_extract_output')
    vtu_path = os.path.join(ac_out, 'whole_from_inp.vtu')
    id_map_path = os.path.join(ac_out, 'abaqus_id_to_vtu_index.csv')
    measures_path = os.path.join(ac_out, 'measures_ID_original.csv')
    camera_path = os.path.join(SCRIPT_DIR, 'camera_position.json')
    inp_path = "C:/SHM_abaqus_models/health.inp"

    for p in [vtu_path, id_map_path, measures_path, camera_path, inp_path]:
        if not os.path.exists(p):
            print(f"    [SKIP] Missing: {p}")
            return

    base_mesh = pv.read(vtu_path)
    id_df = pd.read_csv(id_map_path)
    id_mapping = dict(zip(id_df['abaqus_id'], id_df['vtu_index']))
    measures_df = pd.read_csv(measures_path, header=0)
    measure_ids = measures_df.iloc[:, 0].astype(int).tolist()

    elsets = parse_elsets_from_inp(inp_path, 'middlewhole')
    middlewhole_ids = elsets.get('middlewhole', [])
    middlewhole_mesh, _ = extract_middlewhole_submesh(base_mesh, middlewhole_ids, id_mapping)

    if scenario['name'] == 'Sensor_Offset':
        camera_position = _SO_CUSTOM_CAMERA
    else:
        camera_position = load_camera_position(camera_path)

    temp_dir = os.path.join(tl_dir, '_temp_3d')
    os.makedirs(temp_dir, exist_ok=True)

    node_values = {}

    for key in ['pretrain', 'tl', 'fromscratch']:
        model = models[key]

        res_d = compute_residuals(V_test, model, device)
        res_c = compute_residuals(V_control, model, device)
        scores_d = np.abs(res_d).mean(axis=0)
        scores_c = np.abs(res_c).mean(axis=0)

        try:
            nv_d = interpolate_to_nodes(base_mesh, middlewhole_mesh, scores_d, measure_ids, id_mapping)
        except Exception as e:
            print(f"    [WARN] 3D interpolation failed ({key} damage): {e}")
            return

        try:
            nv_c = interpolate_to_nodes(base_mesh, middlewhole_mesh, scores_c, measure_ids, id_mapping)
        except Exception as e:
            print(f"    [WARN] 3D interpolation failed ({key} control): {e}")
            return

        node_values[key] = {'damage': nv_d, 'control': nv_c}

    # Use one numerical colour range across the three models in each row.
    # This preserves visibility of the healthy-control residuals while making
    # within-row source/TL/scratch colour comparisons quantitatively valid.
    damage_clim_max = max(
        0.5,
        max(float(np.nanmax(node_values[key]['damage'])) for key in node_values),
    )
    control_clim_max = max(
        0.5,
        max(float(np.nanmax(node_values[key]['control'])) for key in node_values),
    )

    damage_3d_images = []
    control_3d_images = []
    col_titles = []
    for key in ['pretrain', 'tl', 'fromscratch']:
        col_titles.append(model_names[key])
        d_path = os.path.join(temp_dir, f"{key}_damage_3d.png")
        c_path = os.path.join(temp_dir, f"{key}_control_3d.png")
        try:
            render_damage_3d(
                base_mesh, middlewhole_mesh, node_values[key]['damage'],
                d_path, camera_position, override_clim_max=damage_clim_max,
            )
            render_damage_3d(
                base_mesh, middlewhole_mesh, node_values[key]['control'],
                c_path, camera_position, override_clim_max=control_clim_max,
            )
        except Exception as e:
            print(f"    [WARN] 3D render failed ({key}): {e}")
            return
        damage_3d_images.append(d_path)
        control_3d_images.append(c_path)

    output_path = os.path.join(tl_dir, 'comparison_3d.png')
    merge_images_grid(
        [damage_3d_images, control_3d_images],
        output_path,
        col_titles,
        row_titles=["Test (Damaged)", "Control (Healthy)"]
    )
    print(f"    Saved: {output_path}")


def merge_all_scenario_3d() -> None:
    """Merge the three scenario comparisons into one 6x3 paper figure."""
    image_grid = []
    row_titles = []
    model_keys = ['pretrain', 'tl', 'fromscratch']
    column_titles = ['Unadapted source', 'AE-TL', 'Scratch-trained']

    for scenario in SCENARIOS:
        temp_dir = os.path.join(
            BASE_OUT, scenario['name'], 'tl_comparison', '_temp_3d'
        )
        for state, state_label in [('damage', 'Damaged'), ('control', 'Healthy')]:
            image_grid.append([
                os.path.join(temp_dir, f"{key}_{state}_3d.png")
                for key in model_keys
            ])
            row_titles.append(f"{scenario['figure_label']} - {state_label}")

    output_path = os.path.join(BASE_OUT, 'comparison_3d_all_scenarios.png')
    merge_images_grid(
        image_grid,
        output_path,
        column_titles=column_titles,
        row_titles=row_titles,
    )
    print(f"  Unified 3D comparison saved: {output_path}")


if __name__ == '__main__':
    print("=" * 60)
    print("  Regenerating comparison figures (residuals + training curves + 3D)")
    print("  with enlarged font sizes for paper readability")
    print("=" * 60)

    for s in SCENARIOS:
        regen_scenario(s)

    if HAS_PYVISTA:
        merge_all_scenario_3d()

    print("\n" + "=" * 60)
    print("  All comparison figures regenerated.")
    print("=" * 60)
