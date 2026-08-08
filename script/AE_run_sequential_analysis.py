"""Direct-TL and saved-result analysis for cumulative sequential adaptation.

Run without arguments to train the three direct-TL references after the six
sequential orderings have completed.  Use ``--postprocess-only`` to rebuild
the final-state summary and AUC heatmap from saved results without training.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from AE_sequential_adaptation_auxiliary import (
    DOMAINS,
    compute_val_loss,
    load_domain_data,
    load_pretrained,
    run_sequential,
)


OUTPUT_DIR = os.path.join(
    SCRIPT_DIR, "AE_model_train_and_detect_output", "Sequential_Adaptation"
)
DATA_AUDIT_PATH = os.path.join(OUTPUT_DIR, "cumulative_data_generation_summary.json")
STATE_LABELS = {"DR": "CDR", "SO": "CSO", "SD": "CSD"}


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def _sequence_label(sequence: list[str]) -> str:
    return "→".join(STATE_LABELS[key] for key in sequence)


def run_direct_tl_baselines(domain_data: dict) -> dict:
    """Run single-stage direct TL from the common source checkpoint."""
    results = {}
    for key in ["DR", "SO", "SD"]:
        print(f"\n{'=' * 60}")
        print(f"Direct TL: pretrained -> {key} ({DOMAINS[key]['label']})")
        print("=" * 60)
        results[key] = run_sequential(
            [key], domain_data, label=f"(direct->{key})"
        )
    return results


def compute_pairwise_distances(domain_data: dict) -> dict:
    """Compute pairwise MSE between matched healthy control sets."""
    keys = list(DOMAINS.keys())
    distances = {}
    for left_key in keys:
        left = domain_data[left_key]["val"]
        for right_key in keys:
            right = domain_data[right_key]["val"]
            count = min(left.shape[0], right.shape[0])
            distances[f"{left_key}->{right_key}"] = float(
                np.mean((left[:count] - right[:count]) ** 2)
            )
    return distances


def compute_pretrained_losses(domain_data: dict) -> dict:
    """Evaluate the common source checkpoint on every control set."""
    model = load_pretrained()
    return {
        key: compute_val_loss(model, domain_data[key]["val"])
        for key in DOMAINS
    }


def build_final_state_summary(seq_all: dict, analysis: dict) -> tuple[dict, list]:
    """Compare each direct result only with paths sharing its final state."""
    domains = ["Source", "DR", "SO", "SD"]
    summary = {}
    plot_rows = []
    for final_state in ["DR", "SO", "SD"]:
        final_column = domains.index(final_state)
        direct_metrics = analysis["direct_tl"][final_state]["detection_max"]
        direct_values = [float(direct_metrics[key]["auc"]) for key in domains]
        plot_rows.append(
            (final_state, f"Direct {STATE_LABELS[final_state]}", direct_values)
        )

        sequential_rows = []
        for result in seq_all.values():
            sequence = list(result["sequence"])
            if sequence[-1] != final_state:
                continue
            values = [float(result["detection_max"][key]["auc"]) for key in domains]
            sequential_rows.append(
                {
                    "sequence": sequence,
                    "label": _sequence_label(sequence),
                    "all_domain_auc": dict(zip(domains, values)),
                    "final_state_auc": values[final_column],
                }
            )
        sequential_rows.sort(key=lambda item: item["label"])
        for row in sequential_rows:
            plot_rows.append(
                (
                    final_state,
                    row["label"],
                    [row["all_domain_auc"][key] for key in domains],
                )
            )
        summary[final_state] = {
            "label": STATE_LABELS[final_state],
            "direct_tl_auc": direct_values[final_column],
            "sequential": sequential_rows,
            "sequential_auc_min": min(
                row["final_state_auc"] for row in sequential_rows
            ),
            "sequential_auc_max": max(
                row["final_state_auc"] for row in sequential_rows
            ),
        }
    return summary, plot_rows


def plot_final_state_auc(plot_rows: list, output_path: str) -> None:
    """Plot all direct and sequential AUC rows, grouped by final state."""
    domains = ["Source", "DR", "SO", "SD"]
    matrix = np.asarray([row[2] for row in plot_rows], dtype=float)
    row_labels = [row[1] for row in plot_rows]
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 11,
        }
    )
    fig, axis = plt.subplots(figsize=(9.2, 7.2), constrained_layout=True)
    image = axis.imshow(
        matrix, cmap="YlGnBu", vmin=0.45, vmax=1.0, aspect="auto"
    )
    axis.set_xticks(range(4), ["Source", "CDR", "CSO", "CSD"])
    axis.set_yticks(range(len(row_labels)), row_labels)
    axis.set_xlabel("Evaluation state")
    axis.set_ylabel("Adaptation path")

    for row_index, (final_state, _, values) in enumerate(plot_rows):
        final_column = domains.index(final_state)
        for column_index, value in enumerate(values):
            axis.text(
                column_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value < 0.68 else "black",
                fontweight="bold" if column_index == final_column else "normal",
                fontsize=11,
            )
        axis.add_patch(
            plt.Rectangle(
                (final_column - 0.49, row_index - 0.49),
                0.98,
                0.98,
                fill=False,
                edgecolor="#8B3A3A",
                linewidth=1.5,
            )
        )
    for boundary in [2.5, 5.5]:
        axis.axhline(boundary, color="white", linewidth=2.0)
    colour_bar = fig.colorbar(image, ax=axis, shrink=0.88)
    colour_bar.set_label("Max-channel AUC")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def postprocess_saved_results() -> dict:
    """Attach audit evidence, write final-state summaries and refresh the plot."""
    seq_path = os.path.join(OUTPUT_DIR, "sequential_results.json")
    analysis_path = os.path.join(OUTPUT_DIR, "sequential_analysis.json")
    seq_all = _load_json(seq_path)
    analysis = _load_json(analysis_path)
    data_audit = _load_json(DATA_AUDIT_PATH)
    summary, plot_rows = build_final_state_summary(seq_all, analysis)

    with open(DATA_AUDIT_PATH, "rb") as stream:
        audit_sha256 = hashlib.sha256(stream.read()).hexdigest()
    analysis["data_audit"] = {
        "path": DATA_AUDIT_PATH,
        "sha256": audit_sha256,
        "training_samples_per_cumulative_state": data_audit[
            "training_samples_per_cumulative_state"
        ],
        "final_healthy_control_samples_per_state": data_audit[
            "final_healthy_control_samples_per_state"
        ],
        "damage_samples_per_state": data_audit["damage_samples_per_state"],
        "removed_all_zero_row_indices": data_audit[
            "removed_all_zero_row_indices"
        ],
        "temperature_span_c": data_audit["temperature_span_c"],
    }
    analysis["final_state_auc_summary"] = summary
    with open(analysis_path, "w", encoding="utf-8") as stream:
        json.dump(analysis, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    plot_final_state_auc(
        plot_rows,
        os.path.join(OUTPUT_DIR, "fig_sequential_auc_heatmap.png"),
    )
    return summary


def main() -> None:
    print("Loading cumulative domain data...")
    domain_data = load_domain_data()

    print("\n" + "=" * 60)
    print("ANALYSIS 1: Pairwise MSE between domain healthy control sets")
    print("=" * 60)
    distances = compute_pairwise_distances(domain_data)
    for left_key in DOMAINS:
        print(
            f"  {left_key}: "
            + ", ".join(
                f"{right_key}={distances[f'{left_key}->{right_key}']:.4f}"
                for right_key in DOMAINS
            )
        )

    print("\n" + "=" * 60)
    print("ANALYSIS 2: Pretrained model control loss on each domain")
    print("=" * 60)
    pretrained_losses = compute_pretrained_losses(domain_data)
    for key, value in pretrained_losses.items():
        print(f"  {key:8s}: {value:.6f}")

    print("\n" + "=" * 60)
    print("ANALYSIS 3: Direct TL baselines")
    print("=" * 60)
    direct_results = run_direct_tl_baselines(domain_data)

    analysis = {
        "pairwise_distances": distances,
        "pretrained_losses": pretrained_losses,
        "direct_tl": {
            key: {
                "detection_max": {
                    domain: {
                        metric: value
                        for metric, value in metrics.items()
                        if metric not in ("fprs", "tprs")
                    }
                    for domain, metrics in result["detection_max"].items()
                },
                "final_losses": result["history"][-1]["losses"],
            }
            for key, result in direct_results.items()
        },
    }
    analysis_path = os.path.join(OUTPUT_DIR, "sequential_analysis.json")
    with open(analysis_path, "w", encoding="utf-8") as stream:
        json.dump(analysis, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"\nSaved: {analysis_path}")

    final_summary = postprocess_saved_results()
    print("\nFinal-state max-channel AUC")
    for state in ["DR", "SO", "SD"]:
        item = final_summary[state]
        print(
            f"  {item['label']}: direct={item['direct_tl_auc']:.4f}; "
            f"sequential={item['sequential_auc_min']:.4f}--"
            f"{item['sequential_auc_max']:.4f}"
        )


if __name__ == "__main__":
    if "--postprocess-only" in sys.argv:
        postprocess_saved_results()
    else:
        main()
