"""
AE_gen_sequential_cumulative_data.py
=====================================
Generate cumulative domain-shift datasets for the sequential adaptation experiment.

Physical scenario: repair → sensor offset on repaired structure → drift on repaired+offset structure.
Each stage's data inherits ALL previous domain shifts.

Datasets generated (all under AD_preprocess_datasets_output/):
  Stage 2 (DR+SO):
    - damage_repaired_12_offset_count_1_500      (healthy: repair + offset)
    - second_damage_12_offset_count_1_100         (damage: repair+crack + offset)
  Stage 3 (DR+SO+SD):
    - damage_repaired_12_offset_count_1_drift_nonuniform_500   (healthy: repair + offset + drift)
    - second_damage_12_offset_count_1_drift_nonuniform_100     (damage: repair+crack + offset + drift)

Usage:
    cd script && python AE_gen_sequential_cumulative_data.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

FEM_DATA_ROOT = "C:/SHM_abaqus_data"
AC_OUTPUT = os.path.join(SCRIPT_DIR, "AC_convert_and_extract_output")
AD_OUTPUT = os.path.join(SCRIPT_DIR, "AD_preprocess_datasets_output")

# Import the non-uniform drift function (same factors as SD scenario)
from AE_run_sd_difficulty_ablation import apply_drift

SD_RATIO = 0.10


def collect_stress(fem_dir: str, max_samples: int, measure_ids: list[int]) -> np.ndarray:
    """Extract stress at specified measurement point IDs from FEM iteration.npy files."""
    folders = sorted(
        [f for f in os.listdir(fem_dir) if f.isdigit()],
        key=int
    )[:max_samples]

    V = np.zeros((len(folders), len(measure_ids)), dtype=np.float32)
    ids_set = set(measure_ids)

    for i, folder in enumerate(tqdm(folders, desc=f"  {os.path.basename(fem_dir)}")):
        npy_path = os.path.join(fem_dir, folder, "iteration.npy")
        arr = np.load(npy_path)
        df = pd.DataFrame(arr, columns=["Element Label", "S-Mises"]).set_index("Element Label")
        V[i] = df.loc[measure_ids]["S-Mises"].values

    print(f"  collected {V.shape[0]} samples x {V.shape[1]} channels")
    return V


def save_dataset(name: str, V: np.ndarray):
    out_dir = os.path.join(AD_OUTPUT, name)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "preprocessed_data_raw.npz"), V=V)
    print(f"  saved: {name}/ ({V.shape})")


def main():
    print("=" * 60)
    print("Generating cumulative datasets for sequential adaptation")
    print("=" * 60)

    # Load offset measurement point IDs
    csv_path = os.path.join(AC_OUTPUT, "measures_ID_offset_count_1.csv")
    offset_ids = pd.read_csv(csv_path)["offset_measures"].values.tolist()
    print(f"Offset measurement IDs loaded: {len(offset_ids)} channels")

    # ── Stage 2: Repair + Offset ──────────────────────────────────
    print("\n--- Stage 2: Repair + Offset ---")

    # Healthy: repaired structure, offset sensors
    print("[healthy] damage_repaired_12 + offset_count_1")
    V_healthy_dr_so = collect_stress(
        os.path.join(FEM_DATA_ROOT, "damage_repaired_12"),
        max_samples=500,
        measure_ids=offset_ids,
    )
    save_dataset("damage_repaired_12_offset_count_1_500", V_healthy_dr_so)

    # Damage: repaired+crack structure, offset sensors
    print("[damage] second_damage_12 + offset_count_1")
    V_damage_dr_so = collect_stress(
        os.path.join(FEM_DATA_ROOT, "second_damage_12"),
        max_samples=100,
        measure_ids=offset_ids,
    )
    save_dataset("second_damage_12_offset_count_1_100", V_damage_dr_so)

    # ── Stage 3: Repair + Offset + Drift ──────────────────────────
    print("\n--- Stage 3: Repair + Offset + Drift ---")

    # Apply non-uniform drift to Stage 2 data
    print(f"[healthy] applying non-uniform drift (r={SD_RATIO})")
    V_healthy_dr_so_sd = apply_drift(V_healthy_dr_so, SD_RATIO)
    save_dataset("damage_repaired_12_offset_count_1_drift_nonuniform_500", V_healthy_dr_so_sd)

    print(f"[damage] applying non-uniform drift (r={SD_RATIO})")
    V_damage_dr_so_sd = apply_drift(V_damage_dr_so, SD_RATIO)
    save_dataset("second_damage_12_offset_count_1_drift_nonuniform_100", V_damage_dr_so_sd)

    print("\n" + "=" * 60)
    print("[Done] All cumulative datasets generated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
