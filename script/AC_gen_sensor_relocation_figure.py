"""
AC_gen_sensor_relocation_figure.py
====================================
1x5 panel figure: exact relocated sensor cells highlighted on semi-transparent
FEM mesh. Original cells in red, new cells in blue.

Usage:
    cd script && python AC_gen_sensor_relocation_figure.py
"""

from __future__ import annotations

import json, os, sys, tempfile, shutil
import numpy as np
import pandas as pd
import pyvista as pv
import vtk
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

FIGURE_OUTPUT = os.path.join(
    SCRIPT_DIR, "AE_model_train_and_detect_output", "Sensor_Offset"
)

AC_OUT       = os.path.join(SCRIPT_DIR, "AC_convert_and_extract_output")
VTU_PATH     = os.path.join(AC_OUT, "whole_from_inp.vtu")
ID_MAP_PATH  = os.path.join(AC_OUT, "abaqus_id_to_vtu_index.csv")
MEASURES_PATH = os.path.join(AC_OUT, "measures_ID_original.csv")
CAMERA_PATH  = os.path.join(SCRIPT_DIR, "camera_position.json")
INP_PATH     = "C:/SHM_abaqus_models/health.inp"

WINDOW_W, WINDOW_H = 1200, 1200
ORIGINAL_COLOR = "#C97A6C"
NEW_COLOR      = "#7BA7BC"
MESH_COLOR     = "#D6D6D6"
NO_VALUES      = [1, 2, 3, 4, 5]


def load_camera(path):
    with open(path) as f:
        d = json.load(f)
    return [tuple(d["camera_position"]), tuple(d["focal_point"]),
            tuple(d["view_up"])]


def make_camera(cam_pos, new_focal, factor, tilt_down=0.0):
    cam = np.array(cam_pos[0]); foc = np.array(cam_pos[1])
    d = cam - foc; d_n = d / np.linalg.norm(d)
    nc = np.array(new_focal) + d_n * np.linalg.norm(d) * factor
    nc[1] -= tilt_down
    return [tuple(nc.tolist()), tuple(new_focal), cam_pos[2]]


def cells_by_ids(base_mesh, eids, id_map):
    n = base_mesh.n_cells
    mask = np.zeros(n, dtype=float)
    for eid in eids:
        if eid in id_map:
            i = id_map[eid]
            if 0 <= i < n:
                mask[i] = 1.0
    if mask.max() < 0.5:
        return None
    mc = base_mesh.copy()
    mc.cell_data["_s"] = mask
    return mc.threshold(0.5, scalars="_s")


def load_all():
    print("[load] mesh + data ...")
    base = pv.read(VTU_PATH)
    cell_centers = base.cell_centers().points
    id_df = pd.read_csv(ID_MAP_PATH)
    id_map = dict(zip(id_df["abaqus_id"], id_df["vtu_index"]))
    cam = load_camera(CAMERA_PATH)
    orig_ids = pd.read_csv(MEASURES_PATH).iloc[:, 0].astype(int).tolist()

    from AE_model_train_and_detect_auxiliary import (
        parse_elsets_from_inp, extract_middlewhole_submesh)
    mw_ids = parse_elsets_from_inp(INP_PATH, "middlewhole").get("middlewhole", [])
    mw_mesh, _ = extract_middlewhole_submesh(base, mw_ids, id_map)

    fe = base.extract_feature_edges(
        boundary_edges=True, non_manifold_edges=True,
        feature_edges=True, manifold_edges=False, feature_angle=30)

    offsets = {}
    prev_changed_set = set()
    for n_o in NO_VALUES:
        off = pd.read_csv(
            os.path.join(AC_OUT, f"measures_ID_offset_count_{n_o}.csv")
        ).iloc[:, 0].astype(int).tolist()
        o_e, n_e, ch_idx = [], [], []
        for ch, (oid, nid) in enumerate(zip(orig_ids, off)):
            if oid != nid:
                o_e.append(oid); n_e.append(nid); ch_idx.append(ch)

        cur_changed_set = set(ch_idx)
        new_chs = cur_changed_set - prev_changed_set
        # Pick a representative new channel for the focal point
        new_focal = None
        if new_chs:
            ch_pick = sorted(new_chs)[0]
            # find position of ch_pick within ch_idx to get the orig eid
            pos = ch_idx.index(ch_pick)
            orig_eid = o_e[pos]
            if orig_eid in id_map:
                vtu_idx = id_map[orig_eid]
                if 0 <= vtu_idx < len(cell_centers):
                    new_focal = tuple(cell_centers[vtu_idx].tolist())
        if new_focal is None:
            # Fallback: middlewhole center
            new_focal = tuple(np.array(mw_mesh.center).tolist())

        offsets[n_o] = {"orig": o_e, "new": n_e, "new_focal": new_focal}
        prev_changed_set = cur_changed_set

    return dict(base=base, mw=mw_mesh, fe=fe, cam=cam,
                id_map=id_map, offsets=offsets, cell_centers=cell_centers)


def render_panel(data, orig_eids, new_eids, new_focal, cam, out):
    p = pv.Plotter(window_size=(WINDOW_W, WINDOW_H), off_screen=True)
    p.set_background("white")

    p.add_mesh(data["mw"], color=MESH_COLOR,
               show_edges=True, edge_color="#CCCCCC", line_width=0.3,
               opacity=0.20)
    p.add_mesh(data["fe"], color="#AAAAAA", line_width=1.0,
               render_lines_as_tubes=False, opacity=0.30)

    # Highlight cells by extruding the shell along its normal into a
    # thin 3D slab, ensuring visibility from any view angle.
    def add_extruded_cells(sub, fill_color):
        for i in range(sub.n_cells):
            cell_ug = sub.extract_cells(i)
            pts = cell_ug.points
            if len(pts) < 3:
                continue
            v1 = pts[1] - pts[0]
            v2 = pts[2] - pts[0]
            nrm = np.cross(v1, v2)
            n_norm = np.linalg.norm(nrm)
            if n_norm < 1e-9:
                continue
            nrm = nrm / n_norm
            cell_poly = cell_ug.extract_surface()
            offset = nrm * 200
            slab = cell_poly.extrude(offset.tolist(), capping=True)
            slab.translate((-offset).tolist(), inplace=True)
            p.add_mesh(slab, color=fill_color, opacity=1.0,
                       show_edges=True, edge_color="black", line_width=2.0)

    if orig_eids:
        sub = cells_by_ids(data["base"], orig_eids, data["id_map"])
        if sub and sub.n_cells > 0:
            add_extruded_cells(sub, ORIGINAL_COLOR)

    if new_eids:
        sub = cells_by_ids(data["base"], new_eids, data["id_map"])
        if sub and sub.n_cells > 0:
            add_extruded_cells(sub, NEW_COLOR)

    p.camera_position = cam
    p.render()

    # Project new_focal world coords to 2D pixel position
    px, py = None, None
    if new_focal is not None:
        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToWorld()
        coord.SetValue(*new_focal)
        dp = coord.GetComputedDisplayValue(p.renderer)
        px = int(dp[0])
        py = int(WINDOW_H - dp[1])

    p.screenshot(out)
    p.close()

    # Draw a red highlight ring around the newly-added cell location
    if px is not None and 0 <= px < WINDOW_W and 0 <= py < WINDOW_H:
        img = Image.open(out)
        draw = ImageDraw.Draw(img)
        r = 110
        draw.ellipse([px - r, py - r, px + r, py + r],
                     outline="#E63E2C", width=10)
        img.save(out)


def combine(paths, labels, out):
    imgs = [Image.open(p) for p in paths]
    ws = [im.width for im in imgs]
    mh = max(im.height for im in imgs)
    th_legend = 100   # top legend strip
    lh = 200          # bottom panel labels
    tw = sum(ws)
    th = th_legend + mh + lh

    c = Image.new("RGB", (tw, th), "white")
    d = ImageDraw.Draw(c)
    try: f = ImageFont.truetype("C:/Windows/Fonts/times.ttf", 120)
    except: f = ImageFont.load_default()
    try: lf = ImageFont.truetype("C:/Windows/Fonts/times.ttf", 70)
    except: lf = f

    # Legend at top — centered horizontally
    items = [(ORIGINAL_COLOR, "Original"), (NEW_COLOR, "Relocated")]
    tot = sum(60 + (d.textbbox((0,0),t,font=lf)[2]-d.textbbox((0,0),t,font=lf)[0]) + 70
              for _,t in items)
    lx = (tw - tot) // 2
    ly = (th_legend - 36) // 2
    for col, txt in items:
        r = 18
        d.rectangle([lx, ly, lx+2*r, ly+2*r], fill=col, outline="#444", width=2)
        d.text((lx+2*r+14, ly-12), txt, fill="black", font=lf)
        bb = d.textbbox((0,0), txt, font=lf)
        lx += 2*r + 14 + (bb[2]-bb[0]) + 70

    # Paste panels below legend strip, with black borders
    border_w = 6
    xo = 0
    for im, lb in zip(imgs, labels):
        c.paste(im, (xo, th_legend))
        # Draw black border around this panel
        d.rectangle(
            [xo, th_legend, xo + im.width - 1, th_legend + im.height - 1],
            outline="black", width=border_w,
        )
        bb = d.textbbox((0,0), lb, font=f)
        tx = xo + (im.width - (bb[2]-bb[0])) // 2
        ty = th_legend + mh + (lh - (bb[3]-bb[1])) // 2
        d.text((tx, ty), lb, fill="black", font=f)
        xo += im.width

    c.save(out, dpi=(300,300))
    print(f"[saved] {out}")


def main():
    print("="*60)
    print("Generating sensor relocation figure")
    print("="*60)

    data = load_all()
    os.makedirs(FIGURE_OUTPUT, exist_ok=True)

    td = tempfile.mkdtemp(prefix="reloc_")
    pp, ll = [], []
    for n_o in NO_VALUES:
        info = data["offsets"][n_o]
        cam = make_camera(data["cam"], info["new_focal"],
                          factor=0.45, tilt_down=10000)
        fp = os.path.join(td, f"p{n_o}.png")
        print(f"[render] N_o={n_o} ({len(info['orig'])} relocated)"
              f" focal={info['new_focal']}")
        render_panel(data, info["orig"], info["new"],
                     info["new_focal"], cam, fp)
        pp.append(fp); ll.append(f"N\u2092 = {n_o}")

    combine(pp, ll, os.path.join(FIGURE_OUTPUT, "fig_sensor_relocation.png"))
    shutil.rmtree(td, ignore_errors=True)
    print("[done]")


if __name__ == "__main__":
    main()
