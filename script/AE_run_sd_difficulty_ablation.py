"""Temperature-span and target-sample-size ablation for the
Temperature-conditioned Sensor Response scenario.

The six temperature-response spans are crossed with four target
training-set sizes, giving 24 full-parameter fine-tuning runs. Every span
reuses the same 100 temperature conditions, sensor coordinates, channel
coefficients and source-health-derived channel scales from
``AE_temperature_sensor_drift.py``. No damaged data are loaded or used for
model selection in this experiment.

Usage:
    cd script
    python AE_run_sd_difficulty_ablation.py --check-only
    python AE_run_sd_difficulty_ablation.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

AD_OUTPUT = os.path.join(SCRIPT_DIR, "AD_preprocess_datasets_output")
AE_OUTPUT = os.path.join(SCRIPT_DIR, "AE_model_train_and_detect_output")
PRETRAIN_PTH = os.path.join(AE_OUTPUT, "Damage_Repaired", "pretrain", "autoencoder.pth")

from AE_temperature_sensor_drift import (
    COEFFICIENT_SEED,
    CONDITION_SEED,
    DIFFICULTY_TEMPERATURE_SPANS_C,
    MAIN_TEMPERATURE_SPAN_C,
    build_temperature_sensor_drift_model,
)


# Integers keep model-directory names unambiguous (for example, diff_20_n400).
DIFFICULTY_VALUES = [int(value) for value in DIFFICULTY_TEMPERATURE_SPANS_C]
DATA_SIZES = [50, 100, 200, 400]
TRAINING_SEED = 42

# Fixed before examining damage-detection results.
SETTINGS = {
    "model": {
        "encoder_dims": [768, 384, 192],
        "latent_dim": 192,
        "decoder_dims": [192, 384, 768],
        "dropout": 0.0,
        "activation": "relu",
    },
    "training": {
        "finetune": {"epochs": 1000, "lr": 1e-3, "batch_size": 256},
        "val_samples": 100,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "use_lr_scheduler": True,
        "lr_scheduler_type": "cosine",
        "cosine_eta_min": 1e-6,
        "shuffle": True,
        "optimizer_type": "adam",
        "sgd_momentum": 0.9,
        "noise_std": 0.005,
        "encoder_lr": 1e-5,
    },
}


def _load_health_data() -> np.ndarray:
    health_npz = os.path.join(
        AD_OUTPUT, "health_original_2000", "preprocessed_data_raw.npz"
    )
    if not os.path.exists(health_npz):
        raise FileNotFoundError(f"Health data not found: {health_npz}")
    with np.load(health_npz) as data:
        if "V" not in data:
            raise KeyError(f"Dataset has no 'V' array: {health_npz}")
        values = data["V"].astype(np.float32)
    if values.ndim != 2 or values.shape[1] != 252:
        raise ValueError(f"Expected health data with shape (N, 252), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Source healthy data contain non-finite values")
    return values


def _build_and_check_shifted_health(
    health: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, dict]]:
    expected_spans = (0, 10, 20, 30, 40, 50)
    if tuple(DIFFICULTY_VALUES) != expected_spans:
        raise AssertionError(
            f"Unexpected temperature-response spans: {DIFFICULTY_VALUES}"
        )

    model = build_temperature_sensor_drift_model("original")
    if model.n_channels != health.shape[1]:
        raise AssertionError(
            f"Model/data channel mismatch: {model.n_channels} vs {health.shape[1]}"
        )

    shifted: dict[int, np.ndarray] = {}
    diagnostics: dict[int, dict] = {}
    reference_parameters = None
    for span in DIFFICULTY_VALUES:
        transformed, diag, _ = model.transform(health, float(span))
        if transformed.shape != health.shape or not np.isfinite(transformed).all():
            raise AssertionError(f"Invalid transformed healthy data at span={span}")

        frozen_parameters = tuple(
            np.asarray(diag[name])
            for name in ("measurement_ids", "a", "b", "c", "q")
        )
        if reference_parameters is None:
            reference_parameters = frozen_parameters
        else:
            for name, actual, expected in zip(
                ("measurement_ids", "a", "b", "c", "q"),
                frozen_parameters,
                reference_parameters,
            ):
                np.testing.assert_array_equal(
                    actual, expected, err_msg=f"{name} changed across difficulty levels"
                )

        shifted[span] = transformed
        diagnostics[span] = diag

    np.testing.assert_allclose(shifted[0], health, rtol=0.0, atol=1e-6)

    # The last 100 rows are validation-only; training uses an earlier prefix.
    val_samples = SETTINGS["training"]["val_samples"]
    train_pool_rows = health.shape[0] - val_samples
    if train_pool_rows < max(DATA_SIZES):
        raise AssertionError(
            f"Only {train_pool_rows} training rows remain after validation split"
        )

    # Enforce exact protocol reuse when the main span-20 dataset is available.
    main_path = Path(AD_OUTPUT) / (
        f"health_temperature_response_span{int(MAIN_TEMPERATURE_SPAN_C)}_2000"
    ) / "preprocessed_data_raw.npz"
    if main_path.is_file():
        with np.load(main_path) as data:
            main_values = data["V"].astype(np.float32)
        np.testing.assert_allclose(
            shifted[int(MAIN_TEMPERATURE_SPAN_C)],
            main_values,
            rtol=0.0,
            atol=0.0,
            err_msg=(
                "Difficulty span-20 data do not match the main "
                "Temperature-conditioned Sensor Response dataset"
            ),
        )
        print(f"  [check] span-20 matches main dataset: {main_path}")
    else:
        print(f"  [check] main span-20 dataset absent; cross-file check skipped: {main_path}")

    print("  [check] zero span is identity")
    print("  [check] measurement IDs and channel coefficients are fixed across spans")
    print(
        f"  [check] disjoint split: first <= {max(DATA_SIZES)} training rows; "
        f"last {val_samples} validation rows"
    )
    return shifted, diagnostics


def _write_protocol_summary(output_dir: str, diagnostics: dict[int, dict]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    summary = {
        "scenario": "Sensor_Drift",
        "difficulty_variable": "temperature_span_c",
        "temperature_spans_c": DIFFICULTY_VALUES,
        "data_sizes": DATA_SIZES,
        "condition_seed": CONDITION_SEED,
        "coefficient_seed": COEFFICIENT_SEED,
        "training_seed": TRAINING_SEED,
        "fixed_across_spans": [
            "temperature condition sequence",
            "measurement coordinates",
            "channel coefficients a, b, c, q",
            "source-health-derived channel scales",
            "training configuration",
        ],
        "selection_policy": "healthy validation loss only; damaged data are not loaded",
        "span_diagnostics": {},
    }
    for span, diag in diagnostics.items():
        summary["span_diagnostics"][str(span)] = {
            "temperature_c_min": float(np.min(diag["temperature_c"])),
            "temperature_c_max": float(np.max(diag["temperature_c"])),
            "gain_min": float(np.min(diag["gain"])),
            "gain_max": float(np.max(diag["gain"])),
            "additive_min": float(np.min(diag["additive"])),
            "additive_max": float(np.max(diag["additive"])),
        }
    path = os.path.join(output_dir, "temperature_difficulty_protocol.json")
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"  [saved] {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate healthy data and the frozen transformation without training",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("Temperature-Response Difficulty Ablation")
    print(f"  temperature spans (deg C): {DIFFICULTY_VALUES}")
    print(f"  target training sizes: {DATA_SIZES}")
    print("  fixed temperature sequence and channel coefficients across all spans")
    print(f"  total training runs: {len(DIFFICULTY_VALUES) * len(DATA_SIZES)}")
    print("=" * 72)

    health = _load_health_data()
    print(f"[health] health_original_2000: shape={health.shape}")
    shifted_data, diagnostics = _build_and_check_shifted_health(health)

    output_dir = os.path.join(AE_OUTPUT, "Sensor_Drift", "difficulty_ablation")
    _write_protocol_summary(output_dir, diagnostics)
    if args.check_only:
        print("[check-only] Healthy-data and protocol checks: PASS")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    if not os.path.exists(PRETRAIN_PTH):
        raise FileNotFoundError(f"Pretrain model not found: {PRETRAIN_PTH}")

    random.seed(TRAINING_SEED)
    np.random.seed(TRAINING_SEED)
    torch.manual_seed(TRAINING_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TRAINING_SEED)
    print(f"[seed] training seed: {TRAINING_SEED}")

    from AE_difficulty_ablation_auxiliary import run_difficulty_ablation

    val_samples = SETTINGS["training"]["val_samples"]

    def data_loader_func(difficulty: int, data_size: int):
        values = shifted_data[int(difficulty)]
        train_pool = values[:-val_samples]
        validation = values[-val_samples:]
        if data_size > train_pool.shape[0]:
            raise ValueError(
                f"Requested {data_size} rows, only {train_pool.shape[0]} are available"
            )
        return train_pool[:data_size], validation

    loss_matrix, before_tl_values = run_difficulty_ablation(
        scenario_name="Sensor_Drift",
        difficulty_values=DIFFICULTY_VALUES,
        data_sizes=DATA_SIZES,
        pretrain_model_path=PRETRAIN_PTH,
        data_loader_func=data_loader_func,
        output_dir=output_dir,
        settings=SETTINGS,
        device=device,
        xlabel="Temperature-deviation amplitude (°C)",
        freeze_strategy="none",
    )

    print("\nBefore TL (pretrained model, direct inference):")
    for j, span in enumerate(DIFFICULTY_VALUES):
        print(f"  temperature_span={span} °C: MSE = {before_tl_values[j]:.6f}")

    print("\nTL validation-loss matrix (rows=sample size, columns=temperature span):")
    print("data_size\t" + "\t".join(f"{span}°C" for span in DIFFICULTY_VALUES))
    for i, data_size in enumerate(DATA_SIZES):
        values = "\t".join(
            f"{loss_matrix[i, j]:.6f}" for j in range(len(DIFFICULTY_VALUES))
        )
        print(f"{data_size}\t\t{values}")

    engine_fig = os.path.join(output_dir, "difficulty_surface_Sensor_Drift.png")
    paper_fig = os.path.join(output_dir, "fig_sd_difficulty_surface.png")
    if os.path.exists(engine_fig):
        shutil.copy2(engine_fig, paper_fig)
        print(f"[copy] {os.path.basename(engine_fig)} -> {os.path.basename(paper_fig)}")

    print(f"[done] Output directory: {output_dir}")


if __name__ == "__main__":
    main()
