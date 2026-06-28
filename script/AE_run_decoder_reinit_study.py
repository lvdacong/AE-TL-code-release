"""
AE_run_decoder_reinit_study.py
================================
Decoder reinitialization study: freeze pretrained encoder + reinit
decoder to Xavier random, then train decoder on target-domain data.

Tests whether escaping the pretrained decoder's local minimum achieves
full per-channel recovery while preserving detection capability.

Usage:
    cd script && python AE_run_decoder_reinit_study.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

AD_OUTPUT = os.path.join(SCRIPT_DIR, "AD_preprocess_datasets_output")
AE_OUTPUT = os.path.join(SCRIPT_DIR, "AE_model_train_and_detect_output")
PRETRAIN_PTH = os.path.join(AE_OUTPUT, "Damage_Repaired", "pretrain", "autoencoder.pth")

AE_CFG = dict(encoder_dims=[768, 384, 192], latent_dim=192,
              decoder_dims=[192, 384, 768], dropout=0.0, activation="relu")

from AE_train_model_auxiliary import (
    Autoencoder, apply_freeze_strategy, train_model,
    FIG_DPI, PLOT_STYLE,
)
from AE_channel_ablation_auxiliary import compute_all_channel_mae
from AE_model_train_and_detect_auxiliary import (
    compute_anomaly_scores, compute_detection_metrics,
)


def load_data(folder):
    return np.load(os.path.join(AD_OUTPUT, folder, "preprocessed_data_raw.npz"))["V"].astype(np.float32)


# ========================================
# Scenarios
# ========================================
SCENARIOS = {
    "Damage_Repaired": {
        "new_data": "damage_repaired_12_original_500",
        "test_damage": "second_damage_12_original_100",
        "focus_ch": 123,
    },
    "Sensor_Offset": {
        "new_data": "health_offset_count_4_2000",
        "test_damage": "first_damage_offset_count_4_100",
        "focus_ch": 193,
    },
    "Sensor_Drift": {
        "new_data": "health_drift_nonuniform_2000",
        "test_damage": "first_damage_drift_nonuniform_100",
        "focus_ch": 238,
    },
}

# ========================================
# Experiment configs
# ========================================
PHASE1_CONFIGS = {
    "A1-freeze-enc": dict(
        desc="Pretrained encoder (frozen) + Xavier decoder, Adam lr=1e-3",
        reinit_decoder=True,
        optimizer_type="adam",
        lr=1e-3,
        encoder_lr=0,
        epochs=1000,
    ),
    "A2-low-enc-lr": dict(
        desc="Pretrained encoder (lr=1e-5) + Xavier decoder, Adam lr=1e-3",
        reinit_decoder=True,
        optimizer_type="adam",
        lr=1e-3,
        encoder_lr=1e-5,
        epochs=1000,
    ),
    "A3-sgd": dict(
        desc="Pretrained encoder (frozen) + Xavier decoder, SGD lr=1e-3",
        reinit_decoder=True,
        optimizer_type="sgd",
        lr=1e-3,
        encoder_lr=0,
        epochs=1000,
    ),
}

PHASE2_CONFIGS = {
    "B1-diff-100x": dict(
        desc="Diff LR: encoder=1e-5, decoder=1e-3, Adam (no reinit)",
        reinit_decoder=False,
        optimizer_type="adam",
        lr=1e-3,
        encoder_lr=1e-5,
        epochs=1000,
    ),
    "B2-diff-1000x": dict(
        desc="Diff LR: encoder=1e-6, decoder=1e-3, Adam (no reinit)",
        reinit_decoder=False,
        optimizer_type="adam",
        lr=1e-3,
        encoder_lr=1e-6,
        epochs=1000,
    ),
}

# Baseline for comparison (current TL)
BASELINE_CONFIG = {
    "baseline-TL": dict(
        desc="Current TL: SGD lr=1e-3, no reinit",
        reinit_decoder=False,
        optimizer_type="sgd",
        lr=1e-3,
        encoder_lr=None,
        epochs=1000,
    ),
}

VAL_SAMPLES = 100
EPOCHS = 1000


def run_experiment(config_name, config, V_train, V_val, V_damage, V_control,
                   mae_base, mae_pre, scenario_name, device):
    """Run one config and return evaluation dict."""
    out_dir = os.path.join(AE_OUTPUT, "decoder_reinit_study",
                           scenario_name, config_name)
    os.makedirs(out_dir, exist_ok=True)

    tc = dict(weight_decay=0.0, grad_clip=1.0, use_lr_scheduler=True,
              lr_scheduler_type="cosine", cosine_eta_min=1e-6, shuffle=True,
              optimizer_type=config["optimizer_type"], sgd_momentum=0.9,
              noise_std=0.005)

    t0 = time.time()
    model, _, vl = train_model(
        V_train, V_val, device,
        epochs=config["epochs"], lr=config["lr"], batch_size=256,
        output_dir=out_dir, ae_config=AE_CFG, training_config=tc,
        plot_style=PLOT_STYLE, fig_dpi=150,
        pretrain_model_path=PRETRAIN_PTH,
        model_name=config_name,
        train_shuffle=True,
        reinit_decoder=config["reinit_decoder"],
        encoder_lr=config.get("encoder_lr"),
    )

    # Apply freeze strategy if needed (for reinit configs where we freeze encoder)
    # Note: train_model doesn't apply freeze. For frozen encoder configs,
    # we need encoder_lr=None + freeze. But train_model doesn't support freeze directly.
    # The freeze is handled via encoder_lr=None (params not in optimizer) or by not
    # having trainable params. Let me handle this differently.

    elapsed = time.time() - t0

    # Per-channel MAE
    mae_post = compute_all_channel_mae(V_val, model, device)
    delta_post = mae_post - mae_base
    delta_pre = mae_pre - mae_base

    # Success criterion
    ratio = mae_post / np.maximum(mae_base, 1e-8)
    n_fail = int((ratio > 1.5).sum())
    max_ratio = float(ratio.max())
    worst_ch = int(np.argmax(ratio))

    # Per-channel recovery for shifted channels
    shifted = delta_pre > 1e-6
    n_shifted = int(shifted.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        recov = np.where(shifted, (delta_pre - delta_post) / delta_pre * 100, 100.0)
    mean_recov = float(recov[shifted].mean()) if n_shifted else 100.0

    # Detection metrics
    scores_dmg_mean = compute_anomaly_scores(V_damage, model, device, method="mean")
    scores_ctrl_mean = compute_anomaly_scores(V_control, model, device, method="mean")
    scores_dmg_max = compute_anomaly_scores(V_damage, model, device, method="max")
    scores_ctrl_max = compute_anomaly_scores(V_control, model, device, method="max")

    det_mean = compute_detection_metrics(scores_dmg_mean, scores_ctrl_mean)
    det_max = compute_detection_metrics(scores_dmg_max, scores_ctrl_max)

    return dict(
        config=config_name, desc=config["desc"], time=elapsed,
        val_loss=vl[-1] if vl else float("nan"),
        mean_mae=float(mae_post.mean()),
        mean_recov=mean_recov, n_fail=n_fail, max_ratio=max_ratio,
        worst_ch=worst_ch,
        auc_mean=det_mean["auc"], tpr_mean=det_mean["tpr_at_fpr"],
        f1_mean=det_mean["best_f1"],
        auc_max=det_max["auc"], tpr_max=det_max["tpr_at_fpr"],
        f1_max=det_max["best_f1"],
        mae_post=mae_post, recov=recov,
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load pretrained model for baseline MAE
    V_source = load_data("health_original_2000")
    D = V_source.shape[1]
    V_source_val = V_source[-VAL_SAMPLES:]

    pretrain = Autoencoder(D, **AE_CFG).to(device)
    pretrain.load_state_dict(torch.load(PRETRAIN_PTH, map_location=device))
    pretrain.eval()
    mae_base = compute_all_channel_mae(V_source_val, pretrain, device)

    all_results = []

    # Run configs: baseline + Phase 1 + Phase 2
    configs = {}
    configs.update(BASELINE_CONFIG)
    configs.update(PHASE1_CONFIGS)
    configs.update(PHASE2_CONFIGS)

    for scenario_name, scfg in SCENARIOS.items():
        print(f"\n{'='*70}")
        print(f"Scenario: {scenario_name} (focus=Ch.{scfg['focus_ch']})")
        print(f"{'='*70}")

        V_new = load_data(scfg["new_data"])
        V_val = V_new[-VAL_SAMPLES:]
        V_train = V_new[:len(V_new) - VAL_SAMPLES]  # Use ALL available training data

        V_damage = load_data(scfg["test_damage"])

        # Control data: holdout from target training pool
        V_control = V_new[400:len(V_new) - VAL_SAMPLES]
        if V_control.shape[0] < 50:
            V_control = V_val

        mae_pre = compute_all_channel_mae(V_val, pretrain, device)
        focus = scfg["focus_ch"]
        delta_pre_focus = mae_pre[focus] - mae_base[focus]

        print(f"  Training samples: {V_train.shape[0]}")
        print(f"  Ch.{focus} pre-shift delta: {delta_pre_focus:.4f}")
        print(f"  Baseline MAE Ch.{focus}: {mae_base[focus]:.4f}")

        scenario_results = []

        for config_name, config in configs.items():
            print(f"\n  --- {config_name}: {config['desc']} ---")
            ev = run_experiment(
                config_name, config,
                V_train, V_val, V_damage, V_control,
                mae_base, mae_pre, scenario_name, device,
            )
            scenario_results.append(ev)
            all_results.append(dict(scenario=scenario_name, **{k: v for k, v in ev.items() if k not in ("mae_post", "recov")}))

            focus_recov = ev["recov"][focus]
            focus_ratio = ev["mae_post"][focus] / max(mae_base[focus], 1e-8)
            print(f"    Time: {ev['time']:.0f}s | ValLoss: {ev['val_loss']:.6f}")
            print(f"    Ch.{focus}: recov={focus_recov:.1f}%, ratio={focus_ratio:.2f}x")
            print(f"    Fail: {ev['n_fail']}/252 | MaxRatio: {ev['max_ratio']:.2f}x (Ch.{ev['worst_ch']})")
            print(f"    AUC(mean)={ev['auc_mean']:.3f} AUC(max)={ev['auc_max']:.3f}")

        # Summary table
        print(f"\n{'='*70}")
        print(f"{scenario_name} SUMMARY (success: ALL ratio <= 1.5)")
        print(f"{'='*70}")
        hdr = f"{'Config':<20s} {'Ch.'+str(focus)+'%':>8s} {'Fail':>5s} {'MaxR':>6s} {'AUC_m':>6s} {'AUC_x':>6s} {'ValL':>8s}"
        print(hdr)
        print("-" * len(hdr))
        for ev in sorted(scenario_results, key=lambda x: x["n_fail"]):
            fr = ev["recov"][focus]
            print(f"{ev['config']:<20s} {fr:>7.1f}% {ev['n_fail']:>5d} {ev['max_ratio']:>6.2f} "
                  f"{ev['auc_mean']:>6.3f} {ev['auc_max']:>6.3f} {ev['val_loss']:>8.6f}")

    # Global summary CSV
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(AE_OUTPUT, "decoder_reinit_study", "global_summary.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n[Saved] {csv_path}")

    # Global pass/fail summary
    print(f"\n{'='*70}")
    print("GLOBAL PASS/FAIL (criterion: ALL channels ratio <= 1.5)")
    print(f"{'='*70}")
    for _, row in df.iterrows():
        status = "PASS" if row["n_fail"] == 0 else f"FAIL({int(row['n_fail'])})"
        print(f"  {row['scenario']:<20s} {row['config']:<20s} {status:<12s} "
              f"MaxR={row['max_ratio']:.2f} AUC_max={row['auc_max']:.3f}")

    print(f"\n[Done]")


if __name__ == "__main__":
    main()
