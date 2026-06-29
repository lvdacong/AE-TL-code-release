from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
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
SOURCE_FOLDER = "health_original_2000"


def load_npz(folder: str) -> np.ndarray:
    return np.load(DATA_BASE / folder / "preprocessed_data_raw.npz")["V"].astype(np.float64)


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


def linear_calibration_scores(train: np.ndarray, X: np.ndarray) -> np.ndarray:
    mu = train.mean(axis=0)
    sigma = train.std(axis=0, ddof=1)
    sigma[sigma == 0] = 1.0
    z = (X - mu) / sigma
    return np.sqrt(np.mean(z * z, axis=1))


def ledoit_scores(train: np.ndarray, X: np.ndarray) -> np.ndarray:
    model = LedoitWolf().fit(train)
    return model.mahalanobis(X)


def covariance_sqrt_and_invsqrt(X: np.ndarray, eps: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    cov = np.cov(X, rowvar=False) + eps * np.eye(X.shape[1])
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, eps)
    sqrt = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
    invsqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
    return sqrt, invsqrt


def make_coral_mahalanobis_scorer(source_train: np.ndarray, target_train: np.ndarray):
    source_mean = source_train.mean(axis=0)
    target_mean = target_train.mean(axis=0)
    source_sqrt, _ = covariance_sqrt_and_invsqrt(source_train)
    _, target_invsqrt = covariance_sqrt_and_invsqrt(target_train)
    model = LedoitWolf().fit(source_train)

    def scorer(X: np.ndarray) -> np.ndarray:
        aligned = (X - target_mean) @ target_invsqrt @ source_sqrt + source_mean
        return model.mahalanobis(aligned)

    return scorer


def ocsvm_scores(train: np.ndarray, X: np.ndarray) -> np.ndarray:
    model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(train)
    return -model.decision_function(X)


def iforest_scores(train: np.ndarray, X: np.ndarray) -> np.ndarray:
    model = IsolationForest(n_estimators=300, random_state=20260628, contamination="auto").fit(train)
    return -model.score_samples(X)


def pca_reconstruction_scores(train: np.ndarray, X: np.ndarray, n_components: int = 192) -> np.ndarray:
    model = PCA(n_components=n_components, svd_solver="full").fit(train)
    X_hat = model.inverse_transform(model.transform(X))
    return np.max(np.abs(X - X_hat), axis=1)


def main() -> None:
    rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    V_source = load_npz(SOURCE_FOLDER)[:400]
    source_path = DATA_BASE / SOURCE_FOLDER / "preprocessed_data_raw.npz"

    for scenario, train_folder, damage_folder in SCENARIOS:
        train_path = DATA_BASE / train_folder / "preprocessed_data_raw.npz"
        damage_path = DATA_BASE / damage_folder / "preprocessed_data_raw.npz"
        V_all = load_npz(train_folder)
        V_train = V_all[:400]
        V_control = control_split(V_all, n_control=100)
        V_damage = load_npz(damage_folder)
        coral_mahalanobis_scores = make_coral_mahalanobis_scorer(V_source, V_train)

        for baseline, scores_damage, scores_control in [
            ("逐通道 z-score 最大值", zscore_scores(V_train, V_damage), zscore_scores(V_train, V_control)),
            ("线性校准 RMS z-score", linear_calibration_scores(V_train, V_damage), linear_calibration_scores(V_train, V_control)),
            ("收缩协方差 Mahalanobis 距离", ledoit_scores(V_train, V_damage), ledoit_scores(V_train, V_control)),
            ("CORAL + 源域 Mahalanobis", coral_mahalanobis_scores(V_damage), coral_mahalanobis_scores(V_control)),
            ("PCA-192 重构最大残差", pca_reconstruction_scores(V_train, V_damage), pca_reconstruction_scores(V_train, V_control)),
            ("One-Class SVM", ocsvm_scores(V_train, V_damage), ocsvm_scores(V_train, V_control)),
            ("Isolation Forest", iforest_scores(V_train, V_damage), iforest_scores(V_train, V_control)),
        ]:
            result = metrics(scores_damage, scores_control)
            rows.append({
                "场景": scenario,
                "非神经网络基线": baseline,
                **result,
            })
            audit_rows.append({
                "场景": scenario,
                "非神经网络基线": baseline,
                "target_train_file": str(train_path),
                "target_control_file": str(train_path),
                "damage_file": str(damage_path),
                "source_file": str(source_path) if "源域" in baseline else "",
                "n_target_train": len(V_train),
                "n_target_control": len(V_control),
                "n_damage": len(V_damage),
                **result,
            })

    out_csv = REVIEW_DIR / "轻量基线复核.csv"
    audit_csv = REVIEW_DIR / "轻量基线审计日志.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(audit_rows).to_csv(audit_csv, index=False, encoding="utf-8-sig")
    print(out_csv)
    print(audit_csv)


if __name__ == "__main__":
    main()
