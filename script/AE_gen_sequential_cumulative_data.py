"""Generate audited cumulative datasets for sequential adaptation.

The three cumulative healthy states are

* CDR: Local Structural Stiffening;
* CSO: Local Structural Stiffening + Sensor Relocation;
* CSD: Local Structural Stiffening + Sensor Relocation + the frozen span-20 C
  temperature-conditioned nonlinear sensor response.

Known all-zero repaired-structure snapshots are removed before any
temperature response is applied.  The same retained row indices are used by
CDR, CSO and CSD.  The final 100 healthy rows and all 100 damaged rows use the
same ordered temperature-condition sequence.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from AE_temperature_sensor_drift import (
    MAIN_TEMPERATURE_SPAN_C,
    build_temperature_sensor_drift_model,
)


FEM_DATA_ROOT = Path("C:/SHM_abaqus_data")
AC_OUTPUT = SCRIPT_DIR / "AC_convert_and_extract_output"
AD_OUTPUT = SCRIPT_DIR / "AD_preprocess_datasets_output"
SEQUENTIAL_OUTPUT = (
    SCRIPT_DIR / "AE_model_train_and_detect_output" / "Sequential_Adaptation"
)

CDR_HEALTH_OUTPUT = "damage_repaired_12_sequential_valid"
CSO_HEALTH_OUTPUT = "damage_repaired_12_offset_count_1_sequential_valid"
CSD_HEALTH_OUTPUT = (
    "damage_repaired_12_offset_count_1_temperature_response_span20_sequential_valid"
)
CSD_DAMAGE_OUTPUT = (
    "second_damage_12_offset_count_1_temperature_response_span20_100"
)

BASE_CDR_HEALTH = "damage_repaired_12_original_500"
BASE_CDR_DAMAGE = "second_damage_12_original_100"
N_FINAL_HEALTHY_CONTROL = 100


def collect_stress(
    fem_dir: str | Path,
    max_samples: int,
    measure_ids: list[int],
) -> np.ndarray:
    """Extract S-Mises values at the specified measurement element IDs."""
    fem_dir = Path(fem_dir)
    folders = sorted(
        [path for path in fem_dir.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    )[:max_samples]
    values = np.zeros((len(folders), len(measure_ids)), dtype=np.float32)
    for row, folder in enumerate(tqdm(folders, desc=f"  {fem_dir.name}")):
        array = np.load(folder / "iteration.npy")
        frame = pd.DataFrame(array, columns=["Element Label", "S-Mises"]).set_index(
            "Element Label"
        )
        values[row] = frame.loc[measure_ids]["S-Mises"].values
    print(f"  collected {values.shape[0]} samples x {values.shape[1]} channels")
    return values


def load_dataset(name: str) -> np.ndarray:
    path = AD_OUTPUT / name / "preprocessed_data_raw.npz"
    with np.load(path) as data:
        values = data["V"].astype(np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"Invalid dataset {path}: shape={values.shape}")
    return values


def save_dataset(name: str, values: np.ndarray) -> Path:
    output_dir = AD_OUTPUT / name
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "preprocessed_data_raw.npz"
    np.savez_compressed(path, V=np.asarray(values, dtype=np.float32))
    print(f"  saved: {name}/ ({values.shape})")
    return path


def relative_nrmse(reference: np.ndarray, shifted: np.ndarray) -> float:
    denominator = float(np.sqrt(np.mean(np.asarray(reference, dtype=np.float64) ** 2)))
    if denominator <= 0.0:
        raise ValueError("Cannot compute relative NRMSE from a zero-RMS reference")
    difference = np.asarray(shifted, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    )
    return float(np.sqrt(np.mean(difference**2)) / denominator)


def array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def main() -> None:
    print("=" * 72)
    print("Generating audited cumulative datasets for sequential adaptation")
    print("=" * 72)

    cdr_health_raw = load_dataset(BASE_CDR_HEALTH)
    cdr_damage = load_dataset(BASE_CDR_DAMAGE)

    offset_path = AC_OUTPUT / "measures_ID_offset_count_1.csv"
    offset_ids = pd.read_csv(offset_path)["offset_measures"].astype(int).tolist()
    if len(offset_ids) != cdr_health_raw.shape[1]:
        raise AssertionError("Offset measurement layout does not match the data width")

    print("\n--- Collecting CSO base data: Repair + Sensor Relocation ---")
    cso_health_raw = collect_stress(
        FEM_DATA_ROOT / "damage_repaired_12", 500, offset_ids
    )
    cso_damage = collect_stress(
        FEM_DATA_ROOT / "second_damage_12", 100, offset_ids
    )
    if cdr_health_raw.shape != cso_health_raw.shape:
        raise AssertionError("CDR and CSO healthy arrays must have identical shapes")
    if cdr_damage.shape != cso_damage.shape:
        raise AssertionError("CDR and CSO damage arrays must have identical shapes")

    zero_cdr = np.all(cdr_health_raw == 0.0, axis=1)
    zero_cso = np.all(cso_health_raw == 0.0, axis=1)
    if not np.array_equal(zero_cdr, zero_cso):
        raise AssertionError("All-zero healthy rows differ between CDR and CSO")
    removed_indices = np.flatnonzero(zero_cdr)
    valid_indices = np.flatnonzero(~zero_cdr)
    if removed_indices.size == 0:
        raise AssertionError("Expected known all-zero repaired-structure snapshots")

    cdr_health = cdr_health_raw[valid_indices]
    cso_health = cso_health_raw[valid_indices]
    if cdr_health.shape[0] <= N_FINAL_HEALTHY_CONTROL:
        raise AssertionError("Too few valid healthy rows for the fixed control set")
    n_train = cdr_health.shape[0] - N_FINAL_HEALTHY_CONTROL

    if np.any(np.all(cdr_health == 0.0, axis=1)) or np.any(
        np.all(cso_health == 0.0, axis=1)
    ):
        raise AssertionError("An all-zero healthy row survived filtering")
    if np.any(np.all(cdr_damage == 0.0, axis=1)) or np.any(
        np.all(cso_damage == 0.0, axis=1)
    ):
        raise AssertionError("A damaged array contains an all-zero row")

    print(
        f"  removed all-zero healthy rows: {removed_indices.tolist()} "
        f"({len(removed_indices)} of {len(cdr_health_raw)})"
    )
    print(
        f"  retained healthy rows: {len(valid_indices)} "
        f"({n_train} train + {N_FINAL_HEALTHY_CONTROL} final control)"
    )

    print("\n--- Applying span-20 temperature-conditioned nonlinear response ---")
    response_model = build_temperature_sensor_drift_model(
        "offset_count_1", coordinate_reference_layout="original"
    )
    cso_train = cso_health[:n_train]
    cso_control = cso_health[n_train:]
    csd_train, train_diag, response_config = response_model.transform(
        cso_train, MAIN_TEMPERATURE_SPAN_C
    )
    csd_control, control_diag, _ = response_model.transform(
        cso_control, MAIN_TEMPERATURE_SPAN_C
    )
    csd_damage, damage_diag, _ = response_model.transform(
        cso_damage, MAIN_TEMPERATURE_SPAN_C
    )
    csd_health = np.concatenate((csd_train, csd_control), axis=0)

    np.testing.assert_allclose(
        control_diag["temperature_c"],
        damage_diag["temperature_c"],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        control_diag["condition_index"], damage_diag["condition_index"]
    )
    for label, values in {
        "CDR healthy": cdr_health,
        "CSO healthy": cso_health,
        "CSD healthy": csd_health,
        "CDR damage": cdr_damage,
        "CSO damage": cso_damage,
        "CSD damage": csd_damage,
    }.items():
        if not np.isfinite(values).all():
            raise AssertionError(f"{label} contains non-finite values")
    if min(
        float(train_diag["gain"].min()),
        float(control_diag["gain"].min()),
        float(damage_diag["gain"].min()),
    ) <= 0.0:
        raise AssertionError("Temperature-conditioned gain must remain positive")

    output_paths = {
        "cdr_health": save_dataset(CDR_HEALTH_OUTPUT, cdr_health),
        "cso_health": save_dataset(CSO_HEALTH_OUTPUT, cso_health),
        "csd_health": save_dataset(CSD_HEALTH_OUTPUT, csd_health),
        "csd_damage": save_dataset(CSD_DAMAGE_OUTPUT, csd_damage),
    }

    csd_dir = AD_OUTPUT / CSD_HEALTH_OUTPUT
    diagnostics_path = csd_dir / "temperature_response_diagnostics.npz"
    np.savez_compressed(
        diagnostics_path,
        valid_source_row_indices=valid_indices,
        removed_all_zero_row_indices=removed_indices,
        train_temperature_c=train_diag["temperature_c"],
        control_temperature_c=control_diag["temperature_c"],
        damage_temperature_c=damage_diag["temperature_c"],
        train_gain=train_diag["gain"],
        control_gain=control_diag["gain"],
        damage_gain=damage_diag["gain"],
        train_additive=train_diag["additive"],
        control_additive=control_diag["additive"],
        damage_additive=damage_diag["additive"],
    )

    response_config = dict(response_config)
    response_config.update(
        {
            "partition_policy": (
                "remove shared all-zero healthy rows; first retained rows are "
                "training; final 100 retained rows are final healthy control"
            ),
            "retained_healthy_samples": int(len(valid_indices)),
            "training_samples": int(n_train),
            "final_healthy_control_samples": N_FINAL_HEALTHY_CONTROL,
            "damaged_test_samples": int(len(cso_damage)),
            "removed_all_zero_row_indices": removed_indices.tolist(),
            "matched_control_damage_temperature_conditions": True,
        }
    )
    config_path = csd_dir / "temperature_response_config.json"
    config_path.write_text(
        json.dumps(response_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = {
        "scenario": "CDR -> CSO -> CSD span-20 cumulative state construction",
        "source_datasets": {
            "cdr_health": BASE_CDR_HEALTH,
            "cdr_damage": BASE_CDR_DAMAGE,
            "cso_health_fem": "C:/SHM_abaqus_data/damage_repaired_12",
            "cso_damage_fem": "C:/SHM_abaqus_data/second_damage_12",
        },
        "output_datasets": {
            key: str(path) for key, path in output_paths.items()
        },
        "raw_healthy_samples": int(len(cdr_health_raw)),
        "removed_all_zero_row_indices": removed_indices.tolist(),
        "removed_all_zero_count": int(len(removed_indices)),
        "retained_healthy_samples": int(len(valid_indices)),
        "training_samples_per_cumulative_state": int(n_train),
        "final_healthy_control_samples_per_state": N_FINAL_HEALTHY_CONTROL,
        "damage_samples_per_state": int(len(cdr_damage)),
        "temperature_span_c": MAIN_TEMPERATURE_SPAN_C,
        "temperature_min_c": float(control_diag["temperature_c"].min()),
        "temperature_max_c": float(control_diag["temperature_c"].max()),
        "gain_min": float(
            min(train_diag["gain"].min(), control_diag["gain"].min())
        ),
        "gain_max": float(
            max(train_diag["gain"].max(), control_diag["gain"].max())
        ),
        "temperature_response_nrmse_healthy": relative_nrmse(
            cso_health, csd_health
        ),
        "full_composite_nrmse_vs_cdr_healthy": relative_nrmse(
            cdr_health, csd_health
        ),
        "temperature_response_nrmse_damage": relative_nrmse(
            cso_damage, csd_damage
        ),
        "negative_fraction_csd_healthy": float(np.mean(csd_health < 0.0)),
        "negative_fraction_csd_damage": float(np.mean(csd_damage < 0.0)),
        "matched_control_damage_temperature_conditions": True,
        "valid_source_row_indices_sha256": array_sha256(valid_indices),
        "response_config_sha256": hashlib.sha256(
            json.dumps(response_config, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "array_sha256": {
            "cdr_health": array_sha256(cdr_health),
            "cso_health": array_sha256(cso_health),
            "csd_health": array_sha256(csd_health),
            "cdr_damage": array_sha256(cdr_damage),
            "cso_damage": array_sha256(cso_damage),
            "csd_damage": array_sha256(csd_damage),
        },
        "diagnostics_file": str(diagnostics_path),
        "config_file": str(config_path),
    }
    SEQUENTIAL_OUTPUT.mkdir(parents=True, exist_ok=True)
    summary_path = SEQUENTIAL_OUTPUT / "cumulative_data_generation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n--- Generation audit ---")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved audit summary: {summary_path}")


if __name__ == "__main__":
    main()
