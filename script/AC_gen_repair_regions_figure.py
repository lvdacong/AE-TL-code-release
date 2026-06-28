"""
AC_gen_repair_regions_figure.py
================================
Generate a 1x5 panel showing BFS repair regions on the FEM mesh.
Highlights repaired elements for N_r in {4, 8, 12, 16, 20}.

Usage:
    cd script && python AC_gen_repair_regions_figure.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_OUTPUT = os.path.join(SCRIPT_DIR, "AE_model_train_and_detect_output", "Damage_Repaired")

# ========================================
# Paths
# ========================================

REGIONS_JSON = os.path.join(SCRIPT_DIR, "A0_repair_regions_output.json")
VTU_PATH = os.path.join(SCRIPT_DIR, "AC_convert_and_extract_output", "whole_from_inp.vtu")
ID_MAPPING_PATH = os.path.join(SCRIPT_DIR, "AC_convert_and_extract_output", "abaqus_id_to_vtu_index.csv")
CAMERA_PATH = os.path.join(SCRIPT_DIR, "camera_position.json")
INP_PATH = "C:/SHM_abaqus_models/health.inp"

# ========================================
# Render settings
# ========================================

WINDOW_W, WINDOW_H = 1200, 1200
BACKGROUND = "white"
FEATURE_EDGE_ANGLE = 30
FEATURE_EDGE_COLOR = "black"
FEATURE_EDGE_WIDTH = 1.5
HIGHLIGHT_COLOR = "#B85450"  # Morandi red-brown
MESH_COLOR = "#D6D6D6"       # Light grey for base mesh
REPAIR_OPACITY = 1.0

NR_VALUES = [4, 8, 12, 16, 20]


def load_camera_position(path: str) -> list:
    with open(path, "r") as f:
        data = json.load(f)
    return [
        tuple(data["camera_position"]),
        tuple(data["focal_point"]),
        tuple(data["view_up"]),
    ]


def load_regions(path: str) -> dict:
    with open(path, "r") as f:
        data = json.load(f)
    return data["regions"]


def load_id_mapping(path: str) -> dict:
    df = pd.read_csv(path)
    return dict(zip(df["abaqus_id"], df["vtu_index"]))


def parse_middlewhole_ids(inp_path: str) -> list:
    """Parse the middlewhole elset from the INP file."""
    sys.path.insert(0, SCRIPT_DIR)
    from AE_model_train_and_detect_auxiliary import parse_elsets_from_inp
    elsets = parse_elsets_from_inp(inp_path, "middlewhole")
    return elsets.get("middlewhole", [])


def compute_repair_centroid(base_mesh, repair_ids, id_mapping):
    """Compute the centroid of the repair region from cell centers."""
    centers = base_mesh.cell_centers().points
    coords = []
    for eid in repair_ids:
        if eid in id_mapping:
            idx = id_mapping[eid]
            if 0 <= idx < len(centers):
                coords.append(centers[idx])
    if coords:
        return np.mean(coords, axis=0)
    return None


def render_single_panel(
    base_mesh: pv.DataSet,
    middlewhole_mesh: pv.DataSet,
    feature_edges: pv.DataSet,
    repair_element_ids: list,
    id_mapping: dict,
    camera_position: list,
    output_path: str,
    zoom_center: np.ndarray = None,
    zoom_radius: float = 5000.0,
) -> None:
    """Render one panel with highlighted repair elements, zoomed in."""

    # Build a cell mask on the base mesh
    n_cells = base_mesh.n_cells
    is_repair = np.zeros(n_cells, dtype=bool)
    for eid in repair_element_ids:
        if eid in id_mapping:
            vtu_idx = id_mapping[eid]
            if 0 <= vtu_idx < n_cells:
                is_repair[vtu_idx] = True

    # Extract the repair submesh
    mesh_copy = base_mesh.copy()
    mesh_copy.cell_data["is_repair"] = is_repair.astype(float)
    repair_mesh = mesh_copy.threshold(0.5, scalars="is_repair")

    plotter = pv.Plotter(
        window_size=(WINDOW_W, WINDOW_H),
        off_screen=True,
    )
    plotter.set_background(BACKGROUND)

    # Base middlewhole mesh (grey) with edges to show element structure
    plotter.add_mesh(
        middlewhole_mesh,
        color=MESH_COLOR,
        show_edges=True,
        edge_color="#BBBBBB",
        line_width=0.3,
        opacity=0.85,
    )

    # Highlight repair elements (vivid red)
    if repair_mesh.n_cells > 0:
        plotter.add_mesh(
            repair_mesh,
            color=HIGHLIGHT_COLOR,
            show_edges=True,
            edge_color="#600000",
            line_width=1.5,
            opacity=REPAIR_OPACITY,
        )

    # Feature edges
    plotter.add_mesh(
        feature_edges,
        color=FEATURE_EDGE_COLOR,
        line_width=FEATURE_EDGE_WIDTH,
        render_lines_as_tubes=False,
    )

    # Zoom camera toward repair region
    if zoom_center is not None:
        focal = tuple(zoom_center)
        # Camera offset: above and to the side, close enough to see elements
        cam_pos = (
            zoom_center[0] - zoom_radius * 1.8,
            zoom_center[1] + zoom_radius * 3.0,
            zoom_center[2] + zoom_radius * 0.8,
        )
        view_up = (0.35, 0.90, -0.15)
        plotter.camera_position = [cam_pos, focal, view_up]
    else:
        plotter.camera_position = camera_position

    plotter.screenshot(output_path)
    plotter.close()
    print(f"  [saved] {os.path.basename(output_path)}")


def combine_panels(image_paths: list, labels: list, output_path: str) -> None:
    """Combine N panels into one horizontal figure with labels."""
    images = [Image.open(p) for p in image_paths]
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    max_h = max(heights)

    # Label area height (matches Fig 4/5 aspect ratio)
    label_h = 260
    total_w = sum(widths)
    total_h = max_h + label_h

    combined = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(combined)

    # Try to load a good font; fall back to default
    try:
        font = ImageFont.truetype("times.ttf", 100)
    except OSError:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/times.ttf", 100)
        except OSError:
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/timesbd.ttf", 100)
            except OSError:
                font = ImageFont.load_default()

    x_offset = 0
    border_w = 6
    for img, label in zip(images, labels):
        combined.paste(img, (x_offset, 0))
        # Black border around each panel (consistent with Fig 4/5)
        draw.rectangle(
            [x_offset, 0, x_offset + img.width - 1, max_h - 1],
            outline="black", width=border_w,
        )
        # Center label below each panel
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        tx = x_offset + (img.width - tw) // 2
        ty = max_h + (label_h - (bbox[3] - bbox[1])) // 2
        draw.text((tx, ty), label, fill="black", font=font)
        x_offset += img.width

    combined.save(output_path, dpi=(300, 300))
    print(f"[saved] {output_path}")


def main():
    print("=" * 60)
    print("Generating repair region visualization")
    print("=" * 60)

    # Load resources
    print("[load] VTU mesh...")
    base_mesh = pv.read(VTU_PATH)
    print(f"  cells: {base_mesh.n_cells}, points: {base_mesh.n_points}")

    print("[load] ID mapping...")
    id_mapping = load_id_mapping(ID_MAPPING_PATH)

    print("[load] Camera position...")
    camera_position = load_camera_position(CAMERA_PATH)

    print("[load] Repair regions...")
    regions = load_regions(REGIONS_JSON)

    print("[load] Parsing middlewhole from INP...")
    middlewhole_ids = parse_middlewhole_ids(INP_PATH)
    print(f"  middlewhole elements: {len(middlewhole_ids)}")

    # Extract middlewhole submesh
    from AE_model_train_and_detect_auxiliary import extract_middlewhole_submesh
    middlewhole_mesh, _ = extract_middlewhole_submesh(base_mesh, middlewhole_ids, id_mapping)
    print(f"  middlewhole mesh: {middlewhole_mesh.n_cells} cells")

    # Pre-compute feature edges
    print("[compute] Feature edges...")
    feature_edges = base_mesh.extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=True,
        feature_edges=True,
        manifold_edges=False,
        feature_angle=FEATURE_EDGE_ANGLE,
    )

    # Temporary directory for individual panels
    temp_dir = os.path.join(SCRIPT_DIR, "_temp_repair_panels")
    os.makedirs(temp_dir, exist_ok=True)

    # Compute zoom center from the largest region (N20) to keep camera consistent
    all_N20_ids = regions["N20"]["element_ids"]
    zoom_center = compute_repair_centroid(base_mesh, all_N20_ids, id_mapping)
    print(f"  Repair centroid: {zoom_center}")

    panel_paths = []
    labels = []

    for nr in NR_VALUES:
        key = f"N{nr}"
        elem_ids = regions[key]["element_ids"]
        print(f"\n[render] N_r = {nr} ({len(elem_ids)} elements)")

        panel_path = os.path.join(temp_dir, f"repair_N{nr}.png")
        render_single_panel(
            base_mesh=base_mesh,
            middlewhole_mesh=middlewhole_mesh,
            feature_edges=feature_edges,
            repair_element_ids=elem_ids,
            id_mapping=id_mapping,
            camera_position=camera_position,
            output_path=panel_path,
            zoom_center=zoom_center,
        )
        panel_paths.append(panel_path)
        labels.append(f"Nr = {nr}")

    # Combine panels
    print("\n[combine] Creating 1x5 panel figure...")
    output_path = os.path.join(FIGURE_OUTPUT, "fig_repair_regions.png")
    combine_panels(panel_paths, labels, output_path)

    # Clean up temp
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("[done]")


if __name__ == "__main__":
    main()
