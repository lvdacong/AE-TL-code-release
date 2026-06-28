from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "script"
OUT_BASE = SCRIPT_DIR / "AE_model_train_and_detect_output"
DATA_BASE = SCRIPT_DIR / "AD_preprocess_datasets_output"
REVIEW_DIR = ROOT / "用于查看的中文版"

sys.path.insert(0, str(SCRIPT_DIR))

from AE_model_train_and_detect_auxiliary import compute_anomaly_scores, compute_detection_metrics
from AE_train_model_auxiliary import Autoencoder


AE_CONFIG = {
    "encoder_dims": [768, 384, 192],
    "latent_dim": 192,
    "decoder_dims": [192, 384, 768],
    "dropout": 0.0,
    "activation": "relu",
}

SCENARIOS = [
    {
        "scenario": "结构修复",
        "key": "Damage_Repaired",
        "new_data": "damage_repaired_12_original_500",
        "damage": "second_damage_12_original_100",
        "train_samples": 400,
        "output_subdir": "tl_comparison",
    },
    {
        "scenario": "传感器迁移",
        "key": "Sensor_Offset",
        "new_data": "health_offset_count_1_2000",
        "damage": "first_damage_offset_count_1_100",
        "train_samples": 400,
        "output_subdir": "tl_comparison",
    },
    {
        "scenario": "传感器漂移",
        "key": "Sensor_Drift",
        "new_data": "health_drift_nonuniform_2000",
        "damage": "first_damage_drift_nonuniform_100",
        "train_samples": 400,
        "output_subdir": "tl_comparison",
    },
]

MODELS = [
    ("TL", ("{key}", "{output_subdir}", "TL", "autoencoder.pth")),
    ("未适应源域模型", ("Damage_Repaired", "pretrain", "autoencoder.pth")),
    ("随机初始化基线", ("{key}", "{output_subdir}", "fromscratch", "autoencoder.pth")),
]


def load_npz(folder: str) -> np.ndarray:
    path = DATA_BASE / folder / "preprocessed_data_raw.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)["V"].astype(np.float32)


def load_model(path: Path, input_dim: int, device: torch.device) -> Autoencoder:
    model = Autoencoder(
        input_dim,
        AE_CONFIG["encoder_dims"],
        AE_CONFIG["latent_dim"],
        AE_CONFIG["decoder_dims"],
        AE_CONFIG["dropout"],
        AE_CONFIG["activation"],
    ).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def model_path(parts: tuple[str, ...], sc: dict[str, object]) -> Path:
    resolved = [p.format(**sc) for p in parts]
    return OUT_BASE.joinpath(*resolved)


def control_split(V_all: np.ndarray, train_samples: int, n_control: int, val_samples: int = 100) -> np.ndarray:
    pure_start = train_samples
    pure_end = V_all.shape[0] - val_samples
    if pure_end - pure_start >= n_control:
        return V_all[pure_start:pure_start + n_control]
    return V_all[pure_end:pure_end + n_control]


def evaluate(scores_damage: np.ndarray, scores_control: np.ndarray) -> dict[str, float]:
    metrics = compute_detection_metrics(scores_damage, scores_control)
    return {
        "AUC": round(metrics["auc"], 4),
        "AUC_CI_lo": round(metrics["auc_ci_lo"], 4),
        "AUC_CI_hi": round(metrics["auc_ci_hi"], 4),
        "TPR@FPR5%": round(metrics["tpr_at_fpr"], 4),
        "Best_F1": round(metrics["best_f1"], 4),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(20260625)
    rows: list[dict[str, object]] = []

    for sc in SCENARIOS:
        V_all = load_npz(sc["new_data"])
        V_train = V_all[: int(sc["train_samples"])]
        V_damage = load_npz(sc["damage"])
        V_control = control_split(V_all, int(sc["train_samples"]), V_damage.shape[0])

        channel_std = V_train.std(axis=0, ddof=1).astype(np.float32)
        noise_scale = (0.05 * channel_std).astype(np.float32)
        V_damage_noisy = V_damage + rng.normal(0.0, noise_scale, size=V_damage.shape).astype(np.float32)
        V_control_noisy = V_control + rng.normal(0.0, noise_scale, size=V_control.shape).astype(np.float32)

        input_dim = V_all.shape[1]
        for model_label, parts in MODELS:
            path = model_path(parts, sc)
            model = load_model(path, input_dim, device)

            clean_damage = compute_anomaly_scores(V_damage, model, device, method="max")
            clean_control = compute_anomaly_scores(V_control, model, device, method="max")
            noisy_damage = compute_anomaly_scores(V_damage_noisy, model, device, method="max")
            noisy_control = compute_anomaly_scores(V_control_noisy, model, device, method="max")

            clean = evaluate(clean_damage, clean_control)
            noisy = evaluate(noisy_damage, noisy_control)
            rows.append({
                "场景": sc["scenario"],
                "模型": model_label,
                "评分": "max-channel MAE",
                "噪声设置": "clean",
                **clean,
            })
            rows.append({
                "场景": sc["scenario"],
                "模型": model_label,
                "评分": "max-channel MAE",
                "噪声设置": "5% per-channel train-std Gaussian noise",
                **noisy,
            })

    out_csv = REVIEW_DIR / "AE_TL噪声敏感性复核.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(out_csv)


if __name__ == "__main__":
    main()
