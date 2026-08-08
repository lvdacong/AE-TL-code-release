"""Direct three-model comparison for the composite nonlinear CSD state.

The experiment compares an unadapted source AE, full fine-tuning, and
equal-epoch scratch training on a cumulative state containing local structural
stiffening, sensor relocation, and the frozen nonlinear temperature-conditioned
sensor response.  Damage labels are used only after checkpoint selection for
final detection evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from AE_model_train_and_detect_auxiliary import (
    compute_anomaly_scores,
    compute_detection_metrics,
)
from AE_train_model_auxiliary import Autoencoder


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "AD_preprocess_datasets_output"
MODEL_DIR = SCRIPT_DIR / "AE_model_train_and_detect_output"
OUTPUT_DIR = MODEL_DIR / "Composite_CSD_Comparison"
SOURCE_CHECKPOINT = MODEL_DIR / "Damage_Repaired" / "pretrain" / "autoencoder.pth"
DATA_AUDIT = MODEL_DIR / "Sequential_Adaptation" / "cumulative_data_generation_summary.json"

HEALTH_DATASET = "damage_repaired_12_offset_count_1_temperature_response_span20_sequential_valid"
DAMAGE_DATASET = "second_damage_12_offset_count_1_temperature_response_span20_100"
TRAIN_SAMPLES = 394
CONTROL_SAMPLES = 50
VAL_SAMPLES = 50
SEEDS = (42, 123, 2026)
EPOCHS = 1000
BATCH_SIZE = 256
NOISE_STD = 0.005
GRAD_CLIP = 1.0
DECODER_LR = 1e-3
ENCODER_LR = 1e-5
ETA_MIN = 1e-6

METHOD_LABELS = {
    "source": "Unadapted source",
    "full": "Full FT",
    "scratch": "Scratch-trained",
}
PLOT_LABELS = {
    "source": "Unadapted source",
    "full": "AE-TL",
    "scratch": "Scratch-trained",
}
METHOD_COLORS = {
    "source": "#C97A6C",
    "full": "#7BA7BC",
    "scratch": "#AAAAAA",
}
METHOD_LINESTYLES = {"source": "--", "full": "-", "scratch": ":"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate the compact figure from the saved result JSON.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def load_v(dataset: str) -> Tuple[np.ndarray, Path]:
    path = DATA_DIR / dataset / "preprocessed_data_raw.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        values = data["V"].astype(np.float32)
    return values, path


def load_arrays() -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    health, health_path = load_v(HEALTH_DATASET)
    damage, damage_path = load_v(DAMAGE_DATASET)
    required = TRAIN_SAMPLES + CONTROL_SAMPLES + VAL_SAMPLES
    if health.shape != (required, 252):
        raise AssertionError(f"Expected healthy shape {(required, 252)}, got {health.shape}")
    if damage.shape != (100, 252):
        raise AssertionError(f"Expected damage shape {(100, 252)}, got {damage.shape}")
    if not DATA_AUDIT.exists():
        raise FileNotFoundError(DATA_AUDIT)
    audit = json.loads(DATA_AUDIT.read_text(encoding="utf-8"))
    if not audit.get("matched_control_damage_temperature_conditions", False):
        raise AssertionError("Healthy-control and damage temperature conditions are not matched")
    if audit["array_sha256"]["csd_health"] != array_sha256(health):
        raise AssertionError("CSD healthy dataset hash differs from the cumulative-data audit")
    if audit["array_sha256"]["csd_damage"] != array_sha256(damage):
        raise AssertionError("CSD damage dataset hash differs from the cumulative-data audit")

    arrays = {
        "train": health[:TRAIN_SAMPLES],
        "control": health[TRAIN_SAMPLES:TRAIN_SAMPLES + CONTROL_SAMPLES],
        "val": health[-VAL_SAMPLES:],
        "damage": damage,
    }
    split_audit = {
        "health_dataset": HEALTH_DATASET,
        "damage_dataset": DAMAGE_DATASET,
        "health_path": str(health_path),
        "damage_path": str(damage_path),
        "health_file_sha256": sha256(health_path),
        "damage_file_sha256": sha256(damage_path),
        "health_array_sha256": array_sha256(health),
        "damage_array_sha256": array_sha256(damage),
        "train_rows_zero_based": [0, TRAIN_SAMPLES - 1],
        "control_rows_zero_based": [TRAIN_SAMPLES, TRAIN_SAMPLES + CONTROL_SAMPLES - 1],
        "validation_rows_zero_based": [required - VAL_SAMPLES, required - 1],
        "damage_samples": int(damage.shape[0]),
        "matched_control_damage_temperature_conditions": True,
        "temperature_span_c": float(audit["temperature_span_c"]),
        "response_config_sha256": audit["response_config_sha256"],
    }
    return arrays, split_audit


def make_model(device: torch.device, load_source: bool) -> Autoencoder:
    model = Autoencoder(
        input_dim=252,
        encoder_dims=[768, 384, 192],
        latent_dim=192,
        decoder_dims=[192, 384, 768],
        dropout=0.0,
        activation="relu",
    ).to(device)
    if load_source:
        checkpoint = torch.load(SOURCE_CHECKPOINT, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
    return model


def clone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def reconstruction_mse(model: nn.Module, values: np.ndarray, device: torch.device) -> float:
    model.eval()
    tensor = torch.from_numpy(values).to(device)
    criterion = nn.MSELoss()
    with torch.no_grad():
        return float(criterion(model(tensor), tensor).item())


def detection_results(
    model: nn.Module,
    arrays: Dict[str, np.ndarray],
    device: torch.device,
) -> Tuple[Dict[str, dict], Dict[str, np.ndarray]]:
    metrics_by_score: Dict[str, dict] = {}
    scores: Dict[str, np.ndarray] = {}
    for score_method in ("mean", "max"):
        damage_scores = compute_anomaly_scores(
            arrays["damage"], model, device, method=score_method
        )
        control_scores = compute_anomaly_scores(
            arrays["control"], model, device, method=score_method
        )
        metrics = compute_detection_metrics(
            damage_scores,
            control_scores,
            n_bootstrap=1000,
        )
        metrics_by_score[score_method] = {
            key: float(value)
            for key, value in metrics.items()
            if key not in {"fprs", "tprs"}
        }
        scores[f"{score_method}_damage"] = damage_scores.astype(np.float64)
        scores[f"{score_method}_control"] = control_scores.astype(np.float64)
    return metrics_by_score, scores


def optimizer_for(model: Autoencoder, method: str) -> Adam:
    if method == "full":
        groups = [
            {"params": list(model.encoder.parameters()), "lr": ENCODER_LR},
            {"params": list(model.decoder.parameters()), "lr": DECODER_LR},
        ]
        return Adam(groups, weight_decay=0.0)
    if method == "scratch":
        return Adam(model.parameters(), lr=DECODER_LR, weight_decay=0.0)
    raise ValueError(method)


def first_epoch_within_fraction(losses: List[float], best: float, fraction: float = 1.05) -> int:
    threshold = best * fraction
    return next(index for index, loss in enumerate(losses, start=1) if loss <= threshold)


def train_one(
    arrays: Dict[str, np.ndarray],
    method: str,
    seed: int,
    epochs: int,
    device: torch.device,
    output_dir: Path,
) -> Tuple[List[dict], dict, Dict[str, np.ndarray]]:
    set_seed(seed)
    model = make_model(device, load_source=(method == "full"))
    optimizer = optimizer_for(model, method)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=ETA_MIN)
    criterion = nn.MSELoss()
    x_train = torch.from_numpy(arrays["train"]).to(device)
    x_val = torch.from_numpy(arrays["val"]).to(device)

    best_val = math.inf
    best_epoch = 0
    best_state = clone_state_dict(model)
    rows: List[dict] = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(x_train.shape[0], device=device)
        running = 0.0
        count = 0
        for start in range(0, x_train.shape[0], BATCH_SIZE):
            indices = permutation[start:start + BATCH_SIZE]
            clean = x_train[indices]
            noisy = clean + torch.randn_like(clean) * NOISE_STD
            prediction = model(noisy)
            loss = criterion(prediction, clean)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            running += float(loss.item()) * clean.shape[0]
            count += int(clean.shape[0])
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(x_val), x_val).item())
        train_loss = running / max(1, count)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = clone_state_dict(model)
        rows.append({
            "method": method,
            "method_label": METHOD_LABELS[method],
            "seed": seed,
            "epoch": epoch,
            "train_reconstruction_mse": train_loss,
            "validation_reconstruction_mse": val_loss,
        })

    model.load_state_dict(best_state)
    run_dir = output_dir / f"seed_{seed}" / method
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "autoencoder.pth")
    metrics, scores = detection_results(model, arrays, device)
    val_losses = [row["validation_reconstruction_mse"] for row in rows]
    summary = {
        "method": method,
        "method_label": METHOD_LABELS[method],
        "seed": seed,
        "best_epoch": best_epoch,
        "epoch_within_5pct_of_best": first_epoch_within_fraction(val_losses, best_val),
        "best_validation_reconstruction_mse": best_val,
        "final_validation_reconstruction_mse": val_losses[-1],
        "mean_channel": metrics["mean"],
        "max_channel": metrics["max"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    return rows, summary, scores


def source_result(
    arrays: Dict[str, np.ndarray],
    device: torch.device,
) -> Tuple[dict, Dict[str, np.ndarray]]:
    model = make_model(device, load_source=True)
    metrics, scores = detection_results(model, arrays, device)
    summary = {
        "method": "source",
        "method_label": METHOD_LABELS["source"],
        "seed": None,
        "best_epoch": None,
        "epoch_within_5pct_of_best": None,
        "best_validation_reconstruction_mse": reconstruction_mse(
            model, arrays["val"], device
        ),
        "final_validation_reconstruction_mse": reconstruction_mse(
            model, arrays["val"], device
        ),
        "mean_channel": metrics["mean"],
        "max_channel": metrics["max"],
        "elapsed_seconds": 0.0,
    }
    return summary, scores


def aggregate_summaries(run_summaries: List[dict]) -> List[dict]:
    rows: List[dict] = []
    for method in ("source", "full", "scratch"):
        selected = [row for row in run_summaries if row["method"] == method]
        values = {
            "validation_mse": [row["best_validation_reconstruction_mse"] for row in selected],
            "max_auc": [row["max_channel"]["auc"] for row in selected],
            "max_tpr_at_fpr5": [row["max_channel"]["tpr_at_fpr"] for row in selected],
            "max_best_f1": [row["max_channel"]["best_f1"] for row in selected],
            "mean_auc": [row["mean_channel"]["auc"] for row in selected],
            "best_epoch": [row["best_epoch"] for row in selected if row["best_epoch"] is not None],
            "epoch_within_5pct": [
                row["epoch_within_5pct_of_best"]
                for row in selected
                if row["epoch_within_5pct_of_best"] is not None
            ],
        }
        aggregate: Dict[str, object] = {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "n_training_seeds": len(selected) if method != "source" else 0,
        }
        for key, observations in values.items():
            if not observations:
                aggregate[f"{key}_mean"] = None
                aggregate[f"{key}_sample_sd"] = None
                continue
            array = np.asarray(observations, dtype=float)
            aggregate[f"{key}_mean"] = float(array.mean())
            aggregate[f"{key}_sample_sd"] = (
                float(array.std(ddof=1)) if array.size > 1 else None
            )
        rows.append(aggregate)
    return rows


def roc_arrays(score_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    control = score_dict["max_control"]
    damage = score_dict["max_damage"]
    labels = np.concatenate([np.zeros(control.size), np.ones(damage.size)])
    scores = np.concatenate([control, damage])
    fpr, tpr, _ = roc_curve(labels, scores)
    return fpr, tpr


def plot_results(
    curves: pd.DataFrame,
    run_summaries: List[dict],
    scores_by_run: Dict[str, Dict[str, np.ndarray]],
    output_dir: Path,
) -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 8,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.frameon": False,
        "axes.grid": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), constrained_layout=True)

    source_summary = next(row for row in run_summaries if row["method"] == "source")
    source_val = source_summary["best_validation_reconstruction_mse"]
    axes[0].axhline(
        source_val,
        color=METHOD_COLORS["source"],
        linestyle=METHOD_LINESTYLES["source"],
        linewidth=1.0,
    )
    for method in ("full", "scratch"):
        selected = curves[curves["method"] == method]
        grouped = selected.groupby("epoch")["validation_reconstruction_mse"]
        mean = grouped.mean()
        sd = grouped.std(ddof=1).fillna(0.0)
        smooth_mean = mean.rolling(11, center=True, min_periods=1).mean()
        smooth_sd = sd.rolling(11, center=True, min_periods=1).mean()
        x = smooth_mean.index.to_numpy()
        y = smooth_mean.to_numpy()
        s = smooth_sd.to_numpy()
        axes[0].plot(
            x,
            y,
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=1.05,
        )
        axes[0].fill_between(
            x,
            np.maximum(y - s, 1e-12),
            y + s,
            color=METHOD_COLORS[method],
            alpha=0.12,
            linewidth=0,
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Fine-Tuning Epoch")
    axes[0].set_ylabel("Target Validation Loss (MSE)")
    axes[0].set_title("(a) Target-Domain Reconstruction", fontsize=8.5)
    axes[0].tick_params(direction="in")

    common_fpr = np.linspace(0.0, 1.0, 201)
    for method in ("source", "full", "scratch"):
        keys = [key for key in scores_by_run if key == method or key.startswith(f"{method}_seed_")]
        interpolated: List[np.ndarray] = []
        for key in keys:
            fpr, tpr = roc_arrays(scores_by_run[key])
            curve = np.interp(common_fpr, fpr, tpr)
            curve[0] = 0.0
            curve[-1] = 1.0
            interpolated.append(curve)
        matrix = np.vstack(interpolated)
        mean_tpr = matrix.mean(axis=0)
        axes[1].plot(
            common_fpr,
            mean_tpr,
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=1.1,
        )
        if matrix.shape[0] > 1:
            sd_tpr = matrix.std(axis=0, ddof=1)
            axes[1].fill_between(
                common_fpr,
                np.maximum(mean_tpr - sd_tpr, 0.0),
                np.minimum(mean_tpr + sd_tpr, 1.0),
                color=METHOD_COLORS[method],
                alpha=0.12,
                linewidth=0,
            )
    axes[1].plot([0, 1], [0, 1], color="#C8C8C8", linewidth=0.7, linestyle="--")
    axes[1].axvline(0.05, color="#9A9A9A", linewidth=0.7, linestyle="--")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.01)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("(b) Max-Channel MAE", fontsize=8.5)
    axes[1].tick_params(direction="in")

    legend_handles = [
        plt.Line2D(
            [0], [0],
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=1.2,
            label=PLOT_LABELS[method],
        )
        for method in ("source", "full", "scratch")
    ]
    fig.legend(
        handles=legend_handles,
        loc="outside upper center",
        ncol=3,
        fontsize=7.0,
        handlelength=2.2,
    )
    fig.savefig(output_dir / "fig_composite_csd_three_model_comparison.png", dpi=300)
    fig.savefig(output_dir / "fig_composite_csd_three_model_comparison.pdf")
    plt.close(fig)


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        results_path = output_dir / "composite_csd_results.json"
        if not results_path.exists():
            raise FileNotFoundError(results_path)
        saved = json.loads(results_path.read_text(encoding="utf-8"))
        curves_frame = pd.DataFrame(saved["epoch_curves"])
        scores_by_run = {
            run_key: {
                score_key: np.asarray(score_values, dtype=np.float64)
                for score_key, score_values in score_map.items()
            }
            for run_key, score_map in saved["scores"].items()
        }
        plot_results(
            curves_frame,
            saved["run_summaries"],
            scores_by_run,
            output_dir,
        )
        print(f"[replotted] {output_dir}")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not SOURCE_CHECKPOINT.exists():
        raise FileNotFoundError(SOURCE_CHECKPOINT)
    arrays, split_audit = load_arrays()

    config = {
        "status": "core composite-state experiment",
        "scenario": (
            "CSD = local structural stiffening + sensor relocation + nonlinear "
            "temperature-conditioned response"
        ),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_checkpoint_sha256": sha256(SOURCE_CHECKPOINT),
        "data": split_audit,
        "model": {
            "input_dim": 252,
            "encoder_dims": [768, 384, 192],
            "latent_dim": 192,
            "decoder_dims": [192, 384, 768],
            "activation": "relu",
            "dropout": 0.0,
        },
        "training": {
            "seeds": list(args.seeds),
            "epochs": args.epochs,
            "batch_size": BATCH_SIZE,
            "optimizer": "Adam",
            "full_ft_encoder_lr": ENCODER_LR,
            "full_ft_decoder_lr": DECODER_LR,
            "scratch_uniform_lr": DECODER_LR,
            "scheduler": "CosineAnnealingLR",
            "eta_min": ETA_MIN,
            "noise_std": NOISE_STD,
            "grad_clip": GRAD_CLIP,
            "weight_decay": 0.0,
            "checkpoint_rule": "minimum target healthy validation MSE",
        },
        "evaluation": {
            "primary_score": "max-channel MAE",
            "secondary_score": "mean-channel MAE",
            "auc_test_sample_bootstrap_replicates": 1000,
            "damage_labels_used_only_for_final_evaluation": True,
        },
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"[device] {device}")
    if device.type == "cuda":
        print(f"[gpu] {torch.cuda.get_device_name(0)}")
    print(f"[data] train={arrays['train'].shape}, val={arrays['val'].shape}, "
          f"control={arrays['control'].shape}, damage={arrays['damage'].shape}")

    source_summary, source_scores = source_result(arrays, device)
    print(
        f"[source] val={source_summary['best_validation_reconstruction_mse']:.6g} | "
        f"max AUC={source_summary['max_channel']['auc']:.4f}"
    )
    all_curve_rows: List[dict] = []
    run_summaries: List[dict] = [source_summary]
    scores_by_run: Dict[str, Dict[str, np.ndarray]] = {"source": source_scores}

    for method in ("full", "scratch"):
        for seed in args.seeds:
            print(f"[{METHOD_LABELS[method]}] seed={seed}, epochs={args.epochs}")
            curves, summary, scores = train_one(
                arrays=arrays,
                method=method,
                seed=seed,
                epochs=args.epochs,
                device=device,
                output_dir=output_dir,
            )
            all_curve_rows.extend(curves)
            run_summaries.append(summary)
            scores_by_run[f"{method}_seed_{seed}"] = scores
            print(
                f"  best val={summary['best_validation_reconstruction_mse']:.6g} "
                f"@ {summary['best_epoch']} | max AUC={summary['max_channel']['auc']:.4f} | "
                f"time={summary['elapsed_seconds']:.1f}s"
            )

    curves_frame = pd.DataFrame(all_curve_rows)
    curves_frame.to_csv(output_dir / "epoch_curves.csv", index=False)
    pd.json_normalize(run_summaries).to_csv(output_dir / "run_summary.csv", index=False)
    aggregates = aggregate_summaries(run_summaries)
    pd.DataFrame(aggregates).to_csv(output_dir / "aggregate_summary.csv", index=False)

    raw_payload = {
        "run_summaries": run_summaries,
        "aggregate_summary": aggregates,
        "epoch_curves": all_curve_rows,
        "scores": scores_by_run,
    }
    (output_dir / "composite_csd_results.json").write_text(
        json.dumps(json_ready(raw_payload), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    plot_results(curves_frame, run_summaries, scores_by_run, output_dir)
    print(f"[saved] {output_dir}")


if __name__ == "__main__":
    main()
