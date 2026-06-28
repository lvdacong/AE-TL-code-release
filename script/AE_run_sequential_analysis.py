"""
AE_run_sequential_analysis.py
==============================
Supplementary analysis for the cumulative sequential adaptation experiment:
1. Direct TL baselines: pretrained → each cumulative stage (no intermediate steps)
2. Cross-domain distance analysis: explain WHY orderings matter

Usage:
    cd script && python AE_run_sequential_analysis.py
"""
from __future__ import annotations

import os
import sys
import json
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from AE_sequential_adaptation_auxiliary import (
    DOMAINS, DEVICE, load_domain_data, load_pretrained, run_sequential,
    compute_val_loss,
)
from AE_model_train_and_detect_auxiliary import (
    compute_anomaly_scores, compute_detection_metrics,
)

OUTPUT_DIR = os.path.join(
    SCRIPT_DIR, "AE_model_train_and_detect_output", "Sequential_Adaptation"
)


def run_direct_tl_baselines(domain_data: dict) -> dict:
    """Run single-stage (direct) TL from pretrained to each cumulative domain."""
    results = {}
    for key in ["DR", "SO", "SD"]:
        print(f"\n{'='*60}")
        print(f"Direct TL: pretrained → {key} ({DOMAINS[key]['label']})")
        print(f"{'='*60}")
        res = run_sequential([key], domain_data, label=f"(direct→{key})")
        results[key] = res
    return results


def compute_pairwise_distances(domain_data: dict) -> dict:
    """Compute pairwise MSE between domain healthy val sets to explain coupling."""
    keys = list(DOMAINS.keys())
    dists = {}
    for ki in keys:
        Vi = domain_data[ki]["val"]
        for kj in keys:
            Vj = domain_data[kj]["val"]
            n = min(Vi.shape[0], Vj.shape[0])
            mse = np.mean((Vi[:n] - Vj[:n]) ** 2)
            dists[f"{ki}→{kj}"] = float(mse)
    return dists


def compute_pretrained_losses(domain_data: dict) -> dict:
    """Evaluate pretrained model on all domains."""
    model = load_pretrained()
    losses = {}
    for key in DOMAINS:
        losses[key] = compute_val_loss(model, domain_data[key]["val"])
    return losses


def main():
    print("Loading cumulative domain data...")
    domain_data = load_domain_data()

    # ── Analysis 1: Pairwise data distances ──
    print("\n" + "=" * 60)
    print("ANALYSIS 1: Pairwise MSE between domain healthy data")
    print("=" * 60)
    dists = compute_pairwise_distances(domain_data)
    print(f"\n{'':12s}", end="")
    for kj in DOMAINS:
        print(f"{kj:>12s}", end="")
    print()
    for ki in DOMAINS:
        print(f"{ki:12s}", end="")
        for kj in DOMAINS:
            d = dists[f"{ki}→{kj}"]
            print(f"{d:12.4f}", end="")
        print()

    # ── Analysis 2: Pretrained model losses ──
    print("\n" + "=" * 60)
    print("ANALYSIS 2: Pretrained model val loss on each domain")
    print("=" * 60)
    pt_losses = compute_pretrained_losses(domain_data)
    for k, v in pt_losses.items():
        print(f"  {k:8s}: {v:.6f}")

    # ── Analysis 3: Direct TL baselines ──
    print("\n" + "=" * 60)
    print("ANALYSIS 3: Direct TL baselines (pretrained → each domain)")
    print("=" * 60)
    direct_results = run_direct_tl_baselines(domain_data)

    # ── Summary comparison ──
    print("\n" + "=" * 60)
    print("COMPARISON: Direct TL vs Sequential (max-ch AUC)")
    print("=" * 60)

    # Load sequential results
    seq_json = os.path.join(OUTPUT_DIR, "sequential_results.json")
    if os.path.exists(seq_json):
        with open(seq_json) as f:
            seq_all = json.load(f)
    else:
        seq_all = {}

    # Main sequence DR→SO→SD
    seq_key = "DR→SO→SD"
    if seq_key in seq_all:
        seq_main = seq_all[seq_key]
    else:
        seq_main = None

    print(f"\n{'Domain':<15s} {'Direct TL':>12s} {'Sequential':>12s} {'Diff':>10s}")
    print("-" * 50)
    for key in ["DR", "SO", "SD"]:
        direct_auc = direct_results[key]["detection_max"][key]["auc"]

        if seq_main:
            seq_auc = seq_main["detection_max"][key]["auc"]
        else:
            seq_auc = float("nan")

        diff = direct_auc - seq_auc if seq_main else float("nan")
        print(f"{key} ({DOMAINS[key]['short']})"
              f"{'':>3s}{direct_auc:8.4f}    {seq_auc:8.4f}    {diff:+8.4f}")

    # Cross-domain performance of direct TL
    print(f"\n{'--- Cross-domain detection for Direct TL to SD ---'}")
    direct_sd = direct_results["SD"]
    for dk in DOMAINS:
        m = direct_sd["detection_max"][dk]
        print(f"  {dk:8s}: AUC={m['auc']:.4f}")

    # Save analysis results
    analysis = {
        "pairwise_distances": dists,
        "pretrained_losses": pt_losses,
        "direct_tl": {
            k: {
                "detection_max": {dk: {kk: vv for kk, vv in dv.items()
                                       if kk not in ("fprs", "tprs")}
                                  for dk, dv in v["detection_max"].items()},
                "final_losses": v["history"][-1]["losses"],
            }
            for k, v in direct_results.items()
        },
    }
    out_path = os.path.join(OUTPUT_DIR, "sequential_analysis.json")
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
