"""
ZA_interactive_camera.py
========================
Interactive camera position adjustment tool.
Opens a PyVista window with the current camera_position.json as starting view.
After closing the window, prints the final camera position.
"""
import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyvista as pv
from AE_model_train_and_detect_auxiliary import (
    parse_elsets_from_inp, extract_middlewhole_submesh,
    load_camera_position, RENDER_BACKGROUND,
    FEATURE_EDGE_ANGLE, FEATURE_EDGE_COLOR, FEATURE_EDGE_WIDTH
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AC_OUT     = os.path.join(SCRIPT_DIR, 'AC_convert_and_extract_output')

def main():
    # Load assets
    vtu_path      = os.path.join(AC_OUT, 'whole_from_inp.vtu')
    id_map_path   = os.path.join(AC_OUT, 'abaqus_id_to_vtu_index.csv')
    camera_path   = os.path.join(SCRIPT_DIR, 'camera_position.json')
    inp_path      = "C:/SHM_abaqus_models/health.inp"

    base_mesh = pv.read(vtu_path)
    id_df = pd.read_csv(id_map_path)
    id_mapping = dict(zip(id_df['abaqus_id'], id_df['vtu_index']))

    elsets = parse_elsets_from_inp(inp_path, 'middlewhole')
    middlewhole_ids = elsets.get('middlewhole', [])
    middlewhole_mesh, _ = extract_middlewhole_submesh(base_mesh, middlewhole_ids, id_mapping)

    # Load current camera
    cam = load_camera_position(camera_path)
    print("=" * 60)
    print("  Interactive Camera Adjustment")
    print("=" * 60)
    print(f"\nStarting camera position:")
    print(f"  position:   {list(cam[0])}")
    print(f"  focal_point: {list(cam[1])}")
    print(f"  view_up:     {list(cam[2])}")
    print("\nAdjust the view in the window, then CLOSE the window.")
    print("The final camera position will be printed below.\n")

    # Open interactive window
    plotter = pv.Plotter(window_size=(1200, 900))
    plotter.set_background(RENDER_BACKGROUND)

    plotter.add_mesh(middlewhole_mesh, color='steelblue', opacity=1.0, show_edges=False)

    feature_edges = base_mesh.extract_feature_edges(
        boundary_edges=True, non_manifold_edges=True,
        feature_edges=True, manifold_edges=False,
        feature_angle=FEATURE_EDGE_ANGLE
    )
    plotter.add_mesh(feature_edges, color=FEATURE_EDGE_COLOR,
                     line_width=FEATURE_EDGE_WIDTH, render_lines_as_tubes=False)

    plotter.camera_position = cam
    plotter.show()

    # After window closes, read final camera position
    final_cam = plotter.camera_position
    pos   = list(final_cam[0])
    focal = list(final_cam[1])
    vup   = list(final_cam[2])

    print("\n" + "=" * 60)
    print("  FINAL CAMERA POSITION (copy this)")
    print("=" * 60)
    result = {
        "camera_position": pos,
        "focal_point": focal,
        "view_up": vup
    }
    print(json.dumps(result, indent=2))
    print("=" * 60)

if __name__ == '__main__':
    main()
