from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.svm import OneClassSVM

ROOT = Path(__file__).resolve().parents[2]
DATA_BASE = ROOT / "script" / "AD_preprocess_datasets_output"
REVIEW_DIR = ROOT / "用于查看的中文版"

SCENARIOS = [
    ("结构修复", "damage_repaired_12_original_500", "second_damage_12_original_100"),
    ("传感器迁移", "health_offset_count_1_2000", "first_damage_offset_count_1_100"),
    ("传感器漂移", "health_drift_nonuniform_2000", "first_damage_drift_nonuniform_100"),
]


def load_npz(folder: str) -> np.ndarray:
    return np.load(DATA_BASE / folder / "preprocessed_data_raw.npz")["V"].astype(np.float32)


def control_split(V_all: np.ndarray, train_samples: int = 400, n_control: int = 100, val_samples: int = 100) -> np.ndarray:
    pure_start = train_samples
    pure_end = V_all.shape[0] - val_samples
    if pure_end - pure_start >= n_control:
        return V_all[pure_start:pure_start + n_control]
    return V_all[pure_end:pure_end + n_control]


def metrics(scores_damage: np.ndarray, scores_control: np.ndarray) -> dict[str, float]:
    y_true = np.concatenate([np.ones(len(scores_damage)), np.zeros(len(scores_control))])
    y_score = np.concatenate([scores_damage, scores_control])
    auc = roc_auc_score(y_true, y_score)
    fprs, tprs, thresholds = roc_curve(y_true, y_score)
    idx = min(np.searchsorted(fprs, 0.05), len(fprs) - 1)
    best_f1 = max(f1_score(y_true, (y_score >= t).astype(int), zero_division=0) for t in thresholds)
    return {
        "AUC": round(float(auc), 4),
        "TPR@FPR5%": round(float(tprs[idx]), 4),
        "Best_F1": round(float(best_f1), 4),
    }


def zscore_scores(train: np.ndarray, X: np.ndarray) -> np.ndarray:
    mu = train.mean(axis=0)
    sigma = train.std(axis=0, ddof=1)
    sigma[sigma == 0] = 1.0
    return np.max(np.abs((X - mu) / sigma), axis=1)


def ledoit_scores(train: np.ndarray, X: np.ndarray) -> np.ndarray:
    model = LedoitWolf().fit(train)
    return model.mahalanobis(X)


def ocsvm_scores(train: np.ndarray, X: np.ndarray) -> np.ndarray:
    model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(train)
    return -model.decision_function(X)


def iforest_scores(train: np.ndarray, X: np.ndarray) -> np.ndarray:
    model = IsolationForest(n_estimators=300, random_state=20260628, contamination="auto").fit(train)
    return -model.score_samples(X)


def main() -> None:
    rows: list[dict[str, object]] = []

    for scenario, train_folder, damage_folder in SCENARIOS:
        V_all = load_npz(train_folder)
        V_train = V_all[:400]
        V_control = control_split(V_all, n_control=100)
        V_damage = load_npz(damage_folder)

        for baseline, scorer in [
            ("逐通道 z-score max", zscore_scores),
            ("Ledoit-Wolf Mahalanobis", ledoit_scores),
            ("One-Class SVM", ocsvm_scores),
            ("Isolation Forest", iforest_scores),
        ]:
            result = metrics(scorer(V_train, V_damage), scorer(V_train, V_control))
            rows.append({
                "场景": scenario,
                "非神经网络基线": baseline,
                **result,
            })

    out_csv = REVIEW_DIR / "轻量基线复核.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(out_csv)


if __name__ == "__main__":
    main()
