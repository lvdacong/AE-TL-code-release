"""
AE_gen_baseline_3d_figures.py
==============================
Generate 3D structural visualization for the baseline AE validation
chapter (Section 4). Produces one figure:

fig_baseline_ae_detection.png — 2x3 grid (Global/Zoomed rows):
  (a) Damage location on the FEM mesh (highlighted elements)
  (b) AE residual on healthy inputs (uniformly low)
  (c) AE residual on damaged inputs (hotspot at damage zone)

Usage:
    cd script && python AE_gen_baseline_3d_figures.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import torch

# PyVista
import pyvista as pv

# PIL for panel assembly
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

OUTPUT_DIR = os.path.join(
    SCRIPT_DIR, "AE_model_train_and_detect_output", "baseline_validation"
)
VTU_PATH = os.path.join(SCRIPT_DIR, "AC_convert_and_extract_output", "whole_from_inp.vtu")
ID_MAPPING_PATH = os.path.join(SCRIPT_DIR, "AC_convert_and_extract_output", "abaqus_id_to_vtu_index.csv")
MEASURES_PATH = os.path.join(SCRIPT_DIR, "AC_convert_and_extract_output", "measures_ID_original.csv")
CAMERA_PATH = os.path.join(SCRIPT_DIR, "camera_position.json")
INP_PATH = "C:/SHM_abaqus_models/health.inp"

# AE preprocessed data
PREPROCESS_BASE = os.path.join(SCRIPT_DIR, "AD_preprocess_datasets_output")
HEALTH_NPZ = os.path.join(PREPROCESS_BASE, "health_original_2000", "preprocessed_data_raw.npz")
DAMAGE_NPZ = os.path.join(PREPROCESS_BASE, "two_circle_damage_original_100", "preprocessed_data_raw.npz")

# Pretrained model
PRETRAIN_PTH = os.path.join(
    SCRIPT_DIR, "AE_model_train_and_detect_output", "Damage_Repaired", "pretrain", "autoencoder.pth"
)

# Two circular damage regions (BFS seed→8 elements on Y=4000 plane)
DAMAGE_CIRCLE_A = [126920, 126921, 23228, 126916, 124121, 23227, 126917, 126922]
DAMAGE_CIRCLE_B = [128846, 23896, 128842, 123095, 128847, 23963, 23895, 23892]
DAMAGE_REGION_IDS = DAMAGE_CIRCLE_A + DAMAGE_CIRCLE_B

# AE model config
AE_CONFIG = dict(
    encoder_dims=[768, 384, 192],
    latent_dim=192,
    decoder_dims=[192, 384, 768],
    dropout=0.0,
    activation="relu",
)

# ---------------------------------------------------------------------------
# Render settings
# ---------------------------------------------------------------------------
WINDOW_W, WINDOW_H = 1920, 1080
BACKGROUND = "white"
FEATURE_EDGE_ANGLE = 30
FEATURE_EDGE_COLOR = "black"
FEATURE_EDGE_WIDTH = 1.0

COLORMAP_NAME = "turbo"
COLORMAP_RANGE = (0.05, 1.0)

DAMAGE_HIGHLIGHT_COLOR = "#B85450"
MESH_COLOR = "#D6D6D6"


# =========================================================================
# Helper functions
# =========================================================================

def load_camera_position(path: str) -> list:
    with open(path, "r") as f:
        data = json.load(f)
    return [
        tuple(data["camera_position"]),
        tuple(data["focal_point"]),
        tuple(data["view_up"]),
    ]


def load_id_mapping(path: str) -> dict:
    df = pd.read_csv(path)
    return dict(zip(df["abaqus_id"], df["vtu_index"]))


def load_measure_ids(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    return df["all_measures"].dropna().astype(int).values


def compute_damage_centroid(base_mesh, damage_ids, id_mapping):
    centers = base_mesh.cell_centers().points
    coords = []
    for eid in damage_ids:
        if eid in id_mapping:
            idx = id_mapping[eid]
            if 0 <= idx < len(centers):
                coords.append(centers[idx])
    if coords:
        return np.mean(coords, axis=0)
    return None


def get_colormap_with_range(cmap_name, vmin=0.0, vmax=1.0):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    if vmin == 0.0 and vmax == 1.0:
        return cmap_name
    original = plt.get_cmap(cmap_name)
    colors = original(np.linspace(vmin, vmax, 256))
    return LinearSegmentedColormap.from_list(f"{cmap_name}_range", colors)


def load_font(size=80):
    paths = [
        "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/Times New Roman.ttf",
        "C:/Windows/Fonts/timesbd.ttf",
        "times.ttf",
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_damage_location_panel(
    middlewhole_mesh,
    feature_edges,
    damage_mesh,
    camera_position,
    output_path,
):
    """Render the structural mesh with damage region highlighted (no colorbar)."""
    plotter = pv.Plotter(window_size=(WINDOW_W, WINDOW_H), off_screen=True)
    plotter.set_background(BACKGROUND)

    # Grey base mesh with element edges
    plotter.add_mesh(
        middlewhole_mesh,
        color=MESH_COLOR,
        show_edges=True,
        edge_color="#BBBBBB",
        line_width=0.3,
        opacity=0.85,
    )

    # Highlight damage elements
    if damage_mesh.n_cells > 0:
        plotter.add_mesh(
            damage_mesh,
            color=DAMAGE_HIGHLIGHT_COLOR,
            show_edges=True,
            edge_color="#600000",
            line_width=2.0,
            opacity=1.0,
        )

    # Feature edges
    plotter.add_mesh(
        feature_edges,
        color=FEATURE_EDGE_COLOR,
        line_width=FEATURE_EDGE_WIDTH,
        render_lines_as_tubes=False,
    )

    plotter.camera_position = camera_position
    plotter.screenshot(output_path)
    plotter.close()


def render_residual_panel(
    middlewhole_mesh,
    feature_edges,
    node_values,
    camera_position,
    output_path,
    clim,
    colorbar_title="Mean |Residual|",
    show_colorbar=True,
    damage_overlay_mesh=None,
):
    """Render one panel of the AE residual field (node-based data)."""
    plotter = pv.Plotter(window_size=(WINDOW_W, WINDOW_H), off_screen=True)
    plotter.set_background(BACKGROUND)

    mesh = middlewhole_mesh.copy()
    mesh.point_data["residual"] = node_values

    colormap = get_colormap_with_range(COLORMAP_NAME, COLORMAP_RANGE[0], COLORMAP_RANGE[1])

    sbar_title_fontsize = 56
    sbar_label_fontsize = 48

    plotter.add_mesh(
        mesh,
        scalars="residual",
        cmap=colormap,
        clim=clim,
        show_edges=False,
        opacity=1.0,
        show_scalar_bar=show_colorbar,
        scalar_bar_args={
            "title": colorbar_title,
            "vertical": True,
            "position_x": 0.88,
            "position_y": 0.15,
            "width": 0.05,
            "height": 0.7,
            "title_font_size": sbar_title_fontsize,
            "label_font_size": sbar_label_fontsize,
            "font_family": "times",
            "fmt": "%.5f",
        } if show_colorbar else {},
    )

    if damage_overlay_mesh is not None and damage_overlay_mesh.n_cells > 0:
        plotter.add_mesh(
            damage_overlay_mesh,
            color=DAMAGE_HIGHLIGHT_COLOR,
            show_edges=True,
            edge_color="#600000",
            line_width=3.0,
            opacity=0.7,
        )

    plotter.add_mesh(
        feature_edges,
        color=FEATURE_EDGE_COLOR,
        line_width=FEATURE_EDGE_WIDTH,
        render_lines_as_tubes=False,
    )

    plotter.camera_position = camera_position
    plotter.screenshot(output_path)
    plotter.close()


def compute_zoomed_camera(zoom_center, zoom_radius=3000.0):
    """Compute zoomed camera focusing on the damage region."""
    focal = tuple(zoom_center)
    cam_pos = (
        zoom_center[0] - zoom_radius * 1.8,
        zoom_center[1] + zoom_radius * 3.0,
        zoom_center[2] + zoom_radius * 0.8,
    )
    view_up = (0.35, 0.90, -0.15)
    return [cam_pos, focal, view_up]


def extract_damage_submesh(base_mesh, damage_ids, id_mapping):
    """Extract the submesh corresponding to damage region elements."""
    n_cells = base_mesh.n_cells
    is_damage = np.zeros(n_cells, dtype=bool)
    for eid in damage_ids:
        if eid in id_mapping:
            idx = id_mapping[eid]
            if 0 <= idx < n_cells:
                is_damage[idx] = True

    mesh_copy = base_mesh.copy()
    mesh_copy.cell_data["is_damage"] = is_damage.astype(float)
    damage_mesh = mesh_copy.threshold(0.5, scalars="is_damage")
    return damage_mesh


# =========================================================================
# Generate the combined figure: damage location + AE residuals (2x3)
# =========================================================================

def gen_ae_detection_figure(base_mesh, middlewhole_mesh, feature_edges,
                            id_mapping, camera_position, damage_centroid):
    print("\n" + "=" * 60)
    print("Generating AE detection figure (2x3)")
    print("=" * 60)

    # Load model
    from AE_train_model_auxiliary import Autoencoder
    from AE_model_train_and_detect_auxiliary import (
        compute_residuals, interpolate_to_nodes, merge_images_grid,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    health_data = np.load(HEALTH_NPZ)
    V_health_all = health_data["V"].astype(np.float32)
    input_dim = V_health_all.shape[1]

    model = Autoencoder(
        input_dim,
        AE_CONFIG["encoder_dims"],
        AE_CONFIG["latent_dim"],
        AE_CONFIG["decoder_dims"],
        AE_CONFIG["dropout"],
        AE_CONFIG["activation"],
    ).to(device)
    model.load_state_dict(torch.load(PRETRAIN_PTH, map_location=device, weights_only=True))
    model.eval()
    print(f"[model] Loaded from {PRETRAIN_PTH}")

    V_healthy = V_health_all[-200:]
    damage_data = np.load(DAMAGE_NPZ)
    V_damaged = damage_data["V"].astype(np.float32)
    print(f"[data] Healthy: {V_healthy.shape}, Damaged: {V_damaged.shape}")

    # Compute residuals
    print("[inference] Computing residuals...")
    res_healthy = compute_residuals(V_healthy, model, device)
    res_damaged = compute_residuals(V_damaged, model, device)

    abs_res_h = np.abs(res_healthy)
    abs_res_d = np.abs(res_damaged)

    # Use mean across all samples for both healthy and damaged.
    # With the two-circle damage scenario, the per-channel mean is naturally
    # clean: the two damage seed channels dominate by ~10x over all others.
    channel_res_healthy = abs_res_h.mean(axis=0)  # (252,) over 200 samples
    channel_res_damaged = abs_res_d.mean(axis=0)  # (252,) over 100 samples

    print(f"  Healthy residual (mean): mean={channel_res_healthy.mean():.6f}, max={channel_res_healthy.max():.6f}")
    print(f"  Damaged residual (mean): mean={channel_res_damaged.mean():.6f}, max={channel_res_damaged.max():.6f}")
    top3 = np.argsort(channel_res_damaged)[-3:][::-1]
    print(f"    Top 3 channels: ch{top3[0]}={channel_res_damaged[top3[0]]:.4f}, "
          f"ch{top3[1]}={channel_res_damaged[top3[1]]:.4f}, "
          f"ch{top3[2]}={channel_res_damaged[top3[2]]:.4f}")

    # Interpolate to nodes
    measure_ids = load_measure_ids(MEASURES_PATH)
    print(f"[measures] {len(measure_ids)} measurement points")

    print("[interpolate] Healthy residual to nodes...")
    node_healthy = interpolate_to_nodes(
        base_mesh, middlewhole_mesh, channel_res_healthy, measure_ids, id_mapping
    )
    print("[interpolate] Damaged residual to nodes...")
    node_damaged = interpolate_to_nodes(
        base_mesh, middlewhole_mesh, channel_res_damaged, measure_ids, id_mapping
    )

    clim_max = max(node_damaged.max(), node_healthy.max())
    clim = (0, clim_max)
    print(f"  Residual clim: [{clim[0]:.6f}, {clim[1]:.6f}]")

    # Damage submesh for highlight
    damage_mesh = extract_damage_submesh(base_mesh, DAMAGE_REGION_IDS, id_mapping)
    print(f"  Damage mesh: {damage_mesh.n_cells} cells")

    # Cameras — zoom radius large enough to cover both damage regions
    zoomed_camera = compute_zoomed_camera(damage_centroid, zoom_radius=10000.0)

    # Temp directory
    temp_dir = os.path.join(SCRIPT_DIR, "_temp_detection_panels")
    os.makedirs(temp_dir, exist_ok=True)

    # ---- Column 1: Damage location (global + zoomed) ----
    panel_paths = [[], []]  # [row_global, row_zoomed]

    gpath = os.path.join(temp_dir, "loc_global.png")
    print("  [render] global damage location...")
    render_damage_location_panel(
        middlewhole_mesh, feature_edges, damage_mesh,
        camera_position, gpath,
    )
    panel_paths[0].append(gpath)

    zpath = os.path.join(temp_dir, "loc_zoomed.png")
    print("  [render] zoomed damage location...")
    render_damage_location_panel(
        middlewhole_mesh, feature_edges, damage_mesh,
        zoomed_camera, zpath,
    )
    panel_paths[1].append(zpath)

    # ---- Columns 2-3: Healthy and Damaged residuals ----
    for name, node_vals, cbar_title in [
        ("healthy", node_healthy, "Mean |Residual|"),
        ("damaged", node_damaged, "|Residual|"),
    ]:
        gpath = os.path.join(temp_dir, f"res_global_{name}.png")
        print(f"  [render] global {name} residual...")
        render_residual_panel(
            middlewhole_mesh, feature_edges, node_vals,
            camera_position, gpath, clim,
            colorbar_title=cbar_title,
            show_colorbar=True,
        )
        panel_paths[0].append(gpath)

        zpath = os.path.join(temp_dir, f"res_zoomed_{name}.png")
        print(f"  [render] zoomed {name} residual...")
        render_residual_panel(
            middlewhole_mesh, feature_edges, node_vals,
            zoomed_camera, zpath, clim,
            colorbar_title=cbar_title,
            show_colorbar=True,
            damage_overlay_mesh=damage_mesh,
        )
        panel_paths[1].append(zpath)

    # Assemble 2x3 grid
    print("\n[combine] Assembling 2x3 grid...")
    output_path = os.path.join(OUTPUT_DIR, "fig_baseline_ae_detection.png")
    merge_images_grid(
        panel_paths,
        output_path,
        column_titles=["(a)", "(b)", "(c)"],
        row_titles=["Global", "Zoomed"],
        border_width=15,
    )
    print(f"[saved] {output_path}")

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

    return output_path


# =========================================================================
# Residual heatmap + mean curve figure (single panel for paper)
# =========================================================================

def gen_residual_heatmap_figure():
    """Generate a single heatmap + mean residual curve figure (matches the
    style of one cell from comparison_residuals.png)."""
    print("\n" + "=" * 60)
    print("Generating residual heatmap + mean curve figure")
    print("=" * 60)

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    from AE_train_model_auxiliary import Autoencoder, apply_style, PLOT_STYLE
    from AE_model_train_and_detect_auxiliary import compute_residuals

    apply_style(PLOT_STYLE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    health_data = np.load(HEALTH_NPZ)
    V_health_all = health_data["V"].astype(np.float32)
    input_dim = V_health_all.shape[1]

    model = Autoencoder(
        input_dim,
        AE_CONFIG["encoder_dims"],
        AE_CONFIG["latent_dim"],
        AE_CONFIG["decoder_dims"],
        AE_CONFIG["dropout"],
        AE_CONFIG["activation"],
    ).to(device)
    model.load_state_dict(torch.load(PRETRAIN_PTH, map_location=device, weights_only=True))
    model.eval()

    # Compute residuals on damaged inputs
    damage_data = np.load(DAMAGE_NPZ)
    V_damaged = damage_data["V"].astype(np.float32)
    print(f"[data] Damaged: {V_damaged.shape}")

    residuals = compute_residuals(V_damaged, model, device)
    abs_res = np.abs(residuals)
    mean_abs = abs_res.mean(axis=0)
    N, D = abs_res.shape
    print(f"  Heatmap: {N} samples × {D} channels")

    # Layout: 6 rows × 2 cols
    #   col 0 (wide) holds the plots
    #   col 1 (narrow) holds the colorbar, only spanning the heatmap rows
    # Both heatmap and mean curve use col 0 with identical width → x-axes align.
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(
        6, 2, figure=fig,
        width_ratios=[40, 1],
        hspace=0.08, wspace=0.02,
        left=0.07, right=0.94, top=0.96, bottom=0.12,
    )
    ax_heat = fig.add_subplot(gs[0:4, 0])
    ax_mean = fig.add_subplot(gs[4:6, 0], sharex=ax_heat)
    cax = fig.add_subplot(gs[0:4, 1])

    # ---------- Heatmap (matches _plot_comparison_residuals_grid) ----------
    vmax = np.percentile(abs_res, 99.8)
    im = ax_heat.imshow(
        abs_res,
        aspect="auto",
        cmap="YlOrRd",
        interpolation="nearest",
        vmin=0,
        vmax=vmax,
    )
    ystep = 20 if N <= 100 else (50 if N <= 300 else (100 if N <= 600 else 200))
    ax_heat.set_yticks(np.arange(0, N, ystep))
    ax_heat.set_xlim([0, D - 1])
    ax_heat.tick_params(labelsize=16)
    plt.setp(ax_heat.get_xticklabels(), visible=False)
    ax_heat.set_ylabel("Sample Index", fontsize=20)

    # Colorbar in its own axis (does not shrink the heatmap)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("|Residual|", fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    # ---------- Mean residual curve ----------
    # Single-curve cell; the y-axis label "Mean |Res.|" already identifies the
    # quantity, so no per-subplot legend is drawn.
    ax_mean.plot(range(D), mean_abs, color="#7BA7BC", linewidth=1.0)
    ax_mean.fill_between(range(D), 0, mean_abs, color="#7BA7BC", alpha=0.2)
    ax_mean.set_xlim([0, D - 1])
    ax_mean.set_xticks(np.arange(0, D, 50))
    ax_mean.tick_params(labelsize=16)
    ax_mean.set_xlabel("Channel Index", fontsize=20)
    ax_mean.set_ylabel("Mean |Res.|", fontsize=20)

    output_path = os.path.join(OUTPUT_DIR, "fig_baseline_residual_heatmap.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[saved] {output_path}")
    return output_path


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 60)
    print("Baseline Validation: 3D Visualization Figures")
    print("=" * 60)

    # Load shared resources
    print("[load] VTU mesh...")
    base_mesh = pv.read(VTU_PATH)
    print(f"  cells: {base_mesh.n_cells}, points: {base_mesh.n_points}")

    print("[load] ID mapping...")
    id_mapping = load_id_mapping(ID_MAPPING_PATH)

    print("[load] Camera position...")
    camera_position = load_camera_position(CAMERA_PATH)

    print("[load] Parsing middlewhole from INP...")
    from AE_model_train_and_detect_auxiliary import (
        parse_elsets_from_inp, extract_middlewhole_submesh,
    )
    elsets = parse_elsets_from_inp(INP_PATH, "middlewhole")
    middlewhole_ids = elsets.get("middlewhole", [])
    middlewhole_mesh, _ = extract_middlewhole_submesh(base_mesh, middlewhole_ids, id_mapping)
    print(f"  middlewhole: {middlewhole_mesh.n_cells} cells")

    # Feature edges
    print("[compute] Feature edges...")
    feature_edges = base_mesh.extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=True,
        feature_edges=True,
        manifold_edges=False,
        feature_angle=FEATURE_EDGE_ANGLE,
    )

    # Damage centroid
    damage_centroid = compute_damage_centroid(base_mesh, DAMAGE_REGION_IDS, id_mapping)
    print(f"[damage] Centroid: {damage_centroid}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Residual heatmap + mean curve (single panel)
    fig_residuals = gen_residual_heatmap_figure()

    # 2. 3D detection figure (uses central interpolate_to_nodes,
    #    now configured for inverse_quadratic kernel with epsilon=5e-4)
    fig_3d = gen_ae_detection_figure(
        base_mesh, middlewhole_mesh, feature_edges,
        id_mapping, camera_position, damage_centroid,
    )

    print("\n" + "=" * 60)
    print(f"Heatmap: {fig_residuals}")
    print(f"3D figure: {fig_3d}")
    print("=" * 60)


if __name__ == "__main__":
    main()
