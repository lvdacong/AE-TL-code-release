from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "script"
REVIEW_DIR = ROOT / "\u7528\u4e8e\u67e5\u770b\u7684\u4e2d\u6587\u7248"
OUT_DIR = SCRIPT_DIR / "AE_model_train_and_detect_output" / "Sensor_Offset" / "independent_offset_count4_medium"
REVIEW_CSV = REVIEW_DIR / "\u72ec\u7acb\u504f\u79fb\u6d4b\u8bd5\u590d\u6838.csv"
REVIEW_RAW_DIR = REVIEW_DIR / "\u72ec\u7acb\u504f\u79fb\u6d4b\u8bd5raw"

INDEPENDENT_TEST_CONFIG = {
    "scenario_name": "Sensor_Offset",
    "output_subdir": "independent_offset_count4_medium",
    "description": "Independent SO test: offset_count=4_medium, fixed final config, N=400",
    "new_data_folder": "health_offset_count_4_2000_medium",
    "test_damage_folder": "first_damage_offset_count_4_100_medium",
    "train_samples": 400,
    "freeze_strategy": "none",
    "seed": 42,
    "training_overrides": {
        "l2_sp_alpha": 0.0,
        "encoder_lr": 1e-5,
    },
    "detection_overrides": {
        "render_3d": False,
        "inp_file_path": "C:/SHM_abaqus_models/first_damage.inp",
    },
}


def maybe_run_experiment() -> None:
    import random

    import numpy as np
    import torch

    sys.path.insert(0, str(SCRIPT_DIR))
    from AE_run_dr_tl_comparison import SETTINGS  # noqa: WPS433
    from AE_tl_comparison_auxiliary import run_tl_comparison  # noqa: WPS433

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    settings = copy.deepcopy(SETTINGS)
    settings["training"]["l2_sp_alpha"] = 0.0
    settings["training"]["encoder_lr"] = 1e-5
    settings["detection"]["render_3d"] = False
    settings["detection"]["inp_file_path"] = "C:/SHM_abaqus_models/first_damage.inp"

    cfg = {
        "name": "Sensor_Offset_Independent",
        "description": INDEPENDENT_TEST_CONFIG["description"],
        "new_data_folder": INDEPENDENT_TEST_CONFIG["new_data_folder"],
        "test_damage_folder": INDEPENDENT_TEST_CONFIG["test_damage_folder"],
        "tl_comparison": {
            "train_samples": INDEPENDENT_TEST_CONFIG["train_samples"],
            "freeze_strategy": INDEPENDENT_TEST_CONFIG["freeze_strategy"],
        },
    }

    pretrain = SCRIPT_DIR / "AE_model_train_and_detect_output" / "Damage_Repaired" / "pretrain" / "autoencoder.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_tl_comparison("Sensor_Offset", cfg, str(pretrain), settings, device, output_subdir="independent_offset_count4_medium")


def read_metrics(path: Path, score: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["score"] = score
    return rows


def write_review_csv(rerun_requested: bool) -> None:
    max_csv = OUT_DIR / "detection_metrics_max.csv"
    mean_csv = OUT_DIR / "detection_metrics.csv"
    if not max_csv.exists() or not mean_csv.exists():
        raise FileNotFoundError("Independent-test raw metrics are missing; rerun with --rerun first.")

    rows = read_metrics(max_csv, "max-channel") + read_metrics(mean_csv, "mean-channel")
    fieldnames = ["score", "Model", "AUC", "AUC_CI_lo", "AUC_CI_hi", "TPR@FPR5%", "Best_F1"]
    with REVIEW_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    REVIEW_RAW_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(max_csv, REVIEW_RAW_DIR / "detection_metrics_max.csv")
    shutil.copyfile(mean_csv, REVIEW_RAW_DIR / "detection_metrics.csv")
    for src, dst_name in [
        (OUT_DIR / "TL" / "training_losses.csv", "TL_training_losses.csv"),
        (OUT_DIR / "fromscratch" / "training_losses.csv", "fromscratch_training_losses.csv"),
    ]:
        if src.exists():
            shutil.copyfile(src, REVIEW_RAW_DIR / dst_name)
    for src, dst_name in [
        (OUT_DIR / "training_curve_comparison.png", "training_curve_comparison.png"),
        (OUT_DIR / "roc_comparison_merged.png", "roc_comparison_merged.png"),
        (OUT_DIR / "comparison_residuals.png", "comparison_residuals.png"),
    ]:
        if src.exists():
            shutil.copyfile(src, REVIEW_RAW_DIR / dst_name)

    evidence = dict(INDEPENDENT_TEST_CONFIG)
    evidence["raw_output_dir"] = str(OUT_DIR)
    evidence["raw_output_dir_exists_when_written"] = OUT_DIR.exists()
    evidence["review_writer_script"] = str(Path(__file__).resolve())
    evidence["review_rerun_requested"] = rerun_requested
    evidence["python_executable"] = sys.executable
    evidence["written_at_local"] = datetime.now().astimezone().isoformat(timespec="seconds")
    evidence["raw_files_copied"] = [
        "detection_metrics_max.csv",
        "detection_metrics.csv",
        "TL_training_losses.csv",
        "fromscratch_training_losses.csv",
        "training_curve_comparison.png",
        "roc_comparison_merged.png",
        "comparison_residuals.png",
    ]
    (REVIEW_RAW_DIR / "run_config.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize or rerun the independent offset-count-4 medium test.")
    parser.add_argument("--rerun", action="store_true", help="rerun TL/OldModel/FromScratch before writing the review CSV")
    args = parser.parse_args()

    if args.rerun:
        maybe_run_experiment()
    write_review_csv(args.rerun)
    print(REVIEW_CSV)


if __name__ == "__main__":
    main()
