"""
AC_search_offset_targets.py
================================
Search for optimal sensor-relocation targets that maximize stress-difference
contrast for the Sensor Offset scenario.

Strategy:
  1. Identify high-gradient candidate origin channels (std > threshold).
  2. For each candidate origin, search N-hop element neighbors in the FEM
     adjacency graph.
  3. For each (origin, neighbor) pair, compute the time-series stress
     difference (MAE) using the full health stress data.
  4. Pick the neighbor with maximum stress difference as the best target.
  5. Output a search report CSV for human review.

The output report is then used by the user to manually select 5 nested
(origin, target) pairs for the new offset scheme.
"""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
AC_OUT_DIR = os.path.join(SCRIPT_DIR, "AC_convert_and_extract_output")
AD_OUT_DIR = os.path.join(SCRIPT_DIR, "AD_preprocess_datasets_output")
AE_OUT_DIR = os.path.join(SCRIPT_DIR, "AE_model_train_and_detect_output")

ABAQUS_HEALTH_DIR = "C:/SHM_abaqus_data/health"

ADJACENCY_PKL = os.path.join(AC_OUT_DIR, "adjacency_graph.pkl")
MEASURES_ID_CSV = os.path.join(AC_OUT_DIR, "measures_ID_original.csv")
HEALTH_NPZ = os.path.join(AD_OUT_DIR, "health_original_2000", "preprocessed_data_raw.npz")

OUTPUT_REPORT = os.path.join(
    AE_OUT_DIR, "Sensor_Offset", "offset_search_report.csv"
)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
N_HOP = 3                  # N-hop neighborhood radius (~1.5m at FEM scale)
STD_THRESHOLD = 2.0        # Lower threshold to expand candidate set (was 2.6)
N_SAMPLES_LOAD = 2000      # Number of health iterations to load
TOP_K_TARGETS = 5          # Top-K best targets to record per origin
DIFF_METRIC = "mae"        # "mae" | "rel_mae" | "1_minus_pearson"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def load_measures_ids(csv_path: str) -> list[int]:
    """Load 252 channel element IDs from measures_ID_original.csv (header row)."""
    df = pd.read_csv(csv_path, header=0)
    return [int(x) for x in df.iloc[:, 0].tolist()]


def load_adjacency(pkl_path: str) -> dict[int, set[int]]:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def n_hop_neighbors(
    adj: dict[int, set[int]], seed_node: int, n_hop: int
) -> set[int]:
    """BFS up to n_hop steps. Returns neighbors excluding seed itself."""
    if seed_node not in adj:
        return set()
    visited = {seed_node}
    frontier = {seed_node}
    for _ in range(n_hop):
        next_frontier: set[int] = set()
        for node in frontier:
            next_frontier.update(adj.get(node, set()))
        next_frontier -= visited
        visited.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    return visited - {seed_node}


def load_full_stress(
    abaqus_dir: str,
    elements_of_interest: set[int],
    n_samples: int,
) -> tuple[np.ndarray, list[int]]:
    """
    Load S-Mises stress for the given set of elements across n_samples iterations.

    Returns:
        V_subset: (n_samples_actual, n_elements) array of stress values
        element_order: list of element IDs in the same order as columns
    """
    # Sort folders numerically (1, 2, 3, ..., 2000)
    all_folders = os.listdir(abaqus_dir)
    folders = sorted(
        (f for f in all_folders if f.isdigit()),
        key=lambda x: int(x),
    )[:n_samples]

    element_order = sorted(elements_of_interest)
    n_elements = len(element_order)
    V_subset = np.full((n_samples, n_elements), np.nan, dtype=np.float64)

    sample_idx = 0
    skipped = 0
    for folder in tqdm(folders, desc="  Loading iterations"):
        npy_path = os.path.join(abaqus_dir, folder, "iteration.npy")
        if not os.path.exists(npy_path):
            skipped += 1
            continue
        try:
            arr = np.load(npy_path)  # (n_total_elements, 2): [Element Label, S-Mises]
            ids = arr[:, 0].astype(np.int64)
            stress = arr[:, 1]
            id_to_stress = dict(zip(ids, stress))
            for col, eid in enumerate(element_order):
                if eid in id_to_stress:
                    V_subset[sample_idx, col] = id_to_stress[eid]
            sample_idx += 1
        except Exception as e:
            print(f"  [skip] {folder}: {e}")
            skipped += 1
            continue

    V_subset = V_subset[:sample_idx]
    print(f"  Loaded {sample_idx} samples, {n_elements} elements; skipped {skipped}")
    return V_subset, element_order


def compute_stress_diff(
    V_origin: np.ndarray, V_target: np.ndarray, metric: str = "mae"
) -> float:
    """Time-series stress difference between two element columns."""
    if metric == "mae":
        return float(np.mean(np.abs(V_origin - V_target)))
    elif metric == "rel_mae":
        denom = float(np.mean(np.abs(V_origin)))
        if denom < 1e-9:
            return 0.0
        return float(np.mean(np.abs(V_origin - V_target)) / denom)
    elif metric == "1_minus_pearson":
        if V_origin.std() < 1e-9 or V_target.std() < 1e-9:
            return 0.0
        return 1.0 - float(np.corrcoef(V_origin, V_target)[0, 1])
    else:
        raise ValueError(f"Unknown metric: {metric}")


# ----------------------------------------------------------------------
# Main search routine
# ----------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Sensor Offset Target Search")
    print("=" * 70)
    print(f"  N-hop:        {N_HOP}")
    print(f"  std threshold: {STD_THRESHOLD}")
    print(f"  metric:       {DIFF_METRIC}")
    print()

    # ---- Load measures and existing 252-channel health data ----
    print("[1/5] Loading 252-channel measures and health stress...")
    measures_ids = load_measures_ids(MEASURES_ID_CSV)
    print(f"  {len(measures_ids)} measurement points")

    health_data = np.load(HEALTH_NPZ)
    V_health_252 = health_data["V"].astype(np.float64)  # (2000, 252)
    print(f"  health_original_2000 shape: {V_health_252.shape}")

    # ---- Identify high-gradient candidate origin channels ----
    ch_means = V_health_252.mean(axis=0)
    ch_stds = V_health_252.std(axis=0)
    high_grad_idx = np.where(ch_stds > STD_THRESHOLD)[0]
    print(f"  high-gradient candidates (std > {STD_THRESHOLD}): "
          f"{len(high_grad_idx)} channels")

    # ---- Load adjacency and compute N-hop neighbors ----
    print("\n[2/5] Loading adjacency graph...")
    adj = load_adjacency(ADJACENCY_PKL)
    print(f"  adjacency: {len(adj)} elements, "
          f"avg degree {sum(len(v) for v in adj.values()) / len(adj):.2f}")

    print(f"\n[3/5] Computing {N_HOP}-hop neighbors for {len(high_grad_idx)} candidates...")
    candidate_neighbors: dict[int, set[int]] = {}
    all_target_elements: set[int] = set()
    for ch in high_grad_idx:
        eid = measures_ids[ch]
        neighbors = n_hop_neighbors(adj, eid, N_HOP)
        candidate_neighbors[ch] = neighbors
        all_target_elements.update(neighbors)
    print(f"  total unique candidate target elements: {len(all_target_elements)}")

    # ---- Load full stress for elements of interest ----
    elements_of_interest = set(measures_ids[ch] for ch in high_grad_idx)
    elements_of_interest.update(all_target_elements)
    print(f"\n[4/5] Loading stress for {len(elements_of_interest)} elements "
          f"x {N_SAMPLES_LOAD} samples...")
    V_subset, elem_order = load_full_stress(
        ABAQUS_HEALTH_DIR, elements_of_interest, N_SAMPLES_LOAD,
    )
    elem_to_col = {eid: i for i, eid in enumerate(elem_order)}

    # ---- Search best target for each candidate ----
    print(f"\n[5/5] Searching best targets for {len(high_grad_idx)} candidates...")
    rows = []
    for ch in tqdm(high_grad_idx, desc="  Candidates"):
        origin_eid = measures_ids[ch]
        if origin_eid not in elem_to_col:
            continue
        V_origin = V_subset[:, elem_to_col[origin_eid]]
        if np.any(np.isnan(V_origin)):
            continue

        neighbors = candidate_neighbors[ch]
        scored = []
        for nbr_eid in neighbors:
            if nbr_eid not in elem_to_col:
                continue
            V_nbr = V_subset[:, elem_to_col[nbr_eid]]
            if np.any(np.isnan(V_nbr)):
                continue
            diff = compute_stress_diff(V_origin, V_nbr, DIFF_METRIC)
            nbr_mean = float(V_nbr.mean())
            nbr_std = float(V_nbr.std())
            scored.append((nbr_eid, diff, nbr_mean, nbr_std))

        scored.sort(key=lambda x: -x[1])
        top = scored[:TOP_K_TARGETS]

        for rank, (nbr_eid, diff, nbr_mean, nbr_std) in enumerate(top, 1):
            rows.append({
                "channel_idx": int(ch),
                "origin_id": int(origin_eid),
                "origin_mean": float(ch_means[ch]),
                "origin_std": float(ch_stds[ch]),
                "rank": rank,
                "target_id": int(nbr_eid),
                "target_mean": nbr_mean,
                "target_std": nbr_std,
                "stress_diff": diff,
                "n_neighbors_searched": len(neighbors),
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_REPORT), exist_ok=True)
    df.to_csv(OUTPUT_REPORT, index=False)

    # ---- Print top picks (rank=1 only) for human inspection ----
    print(f"\n[done] Search report saved: {OUTPUT_REPORT}")
    print(f"\nTop-1 best targets per high-gradient candidate (sorted by stress_diff):")
    top1 = df[df["rank"] == 1].sort_values("stress_diff", ascending=False)
    print(top1[["channel_idx", "origin_id", "origin_mean", "origin_std",
                "target_id", "stress_diff"]].to_string(index=False))


if __name__ == "__main__":
    main()
