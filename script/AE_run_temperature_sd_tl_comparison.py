"""Run the formal three-model comparison for temperature-conditioned drift.

The script regenerates the healthy and damaged Temperature-conditioned Sensor
Response datasets from the unshifted FE responses, validates that healthy and
damaged data use the same 100 prescribed temperature conditions, and then runs
the unadapted source, full-fine-tuning and equal-budget scratch comparison.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from AE_run_dr_tl_comparison import SETTINGS
from AE_temperature_sensor_drift import (
    MAIN_TEMPERATURE_SPAN_C,
    build_temperature_sensor_drift_model,
    save_transformed_dataset,
)
from AE_tl_comparison_auxiliary import run_tl_comparison


SCRIPT_DIR = Path(__file__).resolve().parent
AD_OUTPUT = SCRIPT_DIR / "AD_preprocess_datasets_output"
AE_OUTPUT = SCRIPT_DIR / "AE_model_train_and_detect_output"
PRETRAIN_PTH = (
    AE_OUTPUT / "Damage_Repaired" / "pretrain" / "autoencoder.pth"
)

HEALTH_OUTPUT_NAME = "health_temperature_response_span20_2000"
DAMAGE_OUTPUT_NAME = "first_damage_temperature_response_span20_100"


def _load_values(dataset_name: str) -> np.ndarray:
    path = AD_OUTPUT / dataset_name / "preprocessed_data_raw.npz"
    with np.load(path) as data:
        return data["V"].astype(np.float32)


def generate_temperature_response_datasets() -> dict[str, object]:
    """Generate paired formal datasets and return a compact audit summary."""
    health = _load_values("health_original_2000")
    damage = _load_values("first_damage_original_100")
    model = build_temperature_sensor_drift_model("original")

    health_shifted, health_diag, health_config = model.transform(
        health, MAIN_TEMPERATURE_SPAN_C
    )
    damage_shifted, damage_diag, damage_config = model.transform(
        damage, MAIN_TEMPERATURE_SPAN_C
    )

    np.testing.assert_allclose(
        health_diag["temperature_c"][:100],
        damage_diag["temperature_c"],
        rtol=0.0,
        atol=0.0,
    )
    if np.min(health_diag["gain"]) <= 0.0 or np.min(damage_diag["gain"]) <= 0.0:
        raise AssertionError("Temperature-dependent sensitivity must remain positive")

    health_paths = save_transformed_dataset(
        AD_OUTPUT / HEALTH_OUTPUT_NAME,
        health_shifted,
        health_diag,
        health_config,
        overwrite=True,
    )
    damage_paths = save_transformed_dataset(
        AD_OUTPUT / DAMAGE_OUTPUT_NAME,
        damage_shifted,
        damage_diag,
        damage_config,
        overwrite=True,
    )

    health_nrmse = float(
        np.sqrt(np.mean((health_shifted - health) ** 2))
        / np.sqrt(np.mean(health**2))
    )
    config_hash = hashlib.sha256(
        json.dumps(health_config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    summary: dict[str, object] = {
        "temperature_span_c": MAIN_TEMPERATURE_SPAN_C,
        "health_shape": list(health.shape),
        "damage_shape": list(damage.shape),
        "temperature_min_c": float(health_diag["temperature_c"].min()),
        "temperature_max_c": float(health_diag["temperature_c"].max()),
        "gain_min": float(health_diag["gain"].min()),
        "gain_max": float(health_diag["gain"].max()),
        "additive_min": float(health_diag["additive"].min()),
        "additive_max": float(health_diag["additive"].max()),
        "health_relative_nrmse": health_nrmse,
        "negative_output_fraction": float(np.mean(health_shifted < 0.0)),
        "matched_first_100_temperature_conditions": True,
        "config_sha256": config_hash,
        "health_files": {key: str(path) for key, path in health_paths.items()},
        "damage_files": {key: str(path) for key, path in damage_paths.items()},
    }
    summary_path = (
        AE_OUTPUT
        / "Sensor_Drift"
        / "temperature_sensor_drift_generation_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def run_sensor_drift_comparison(
    device: torch.device | None = None,
) -> dict[str, dict]:
    """Generate the formal data and run the temperature-response comparison."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not PRETRAIN_PTH.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {PRETRAIN_PTH}")

    summary = generate_temperature_response_datasets()
    print("[temperature drift generation]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    settings = {**SETTINGS}
    settings["training"] = {
        **SETTINGS["training"],
        "l2_sp_alpha": 0.0,
    }
    settings["detection"] = {**SETTINGS["detection"]}
    settings["detection"]["render_3d"] = True
    settings["detection"]["inp_file_path"] = "C:/SHM_abaqus_models/first_damage.inp"

    scenario_config = {
        "name": "Sensor_Drift",
        "description": (
            "temperature-conditioned nonlinear sensor-response shift at "
            f"span={MAIN_TEMPERATURE_SPAN_C:.0f} C, N=400"
        ),
        "new_data_folder": HEALTH_OUTPUT_NAME,
        "test_damage_folder": DAMAGE_OUTPUT_NAME,
        "tl_comparison": {"train_samples": 400, "freeze_strategy": "none"},
    }
    return run_tl_comparison(
        scenario_name="Sensor_Drift",
        scenario_config=scenario_config,
        pretrain_pth=str(PRETRAIN_PTH),
        settings=settings,
        device=device,
    )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    results = run_sensor_drift_comparison(device)
    print("\n[Temperature-conditioned Sensor Response results]")
    for key, result in results.items():
        losses = result.get("val_losses", [])
        final = losses[-1] if losses else float("nan")
        print(f"  {key}: final_val_loss={final:.6f}")


if __name__ == "__main__":
    main()
