"""
Generate compact comparison figures added during the review revision.

Outputs:
  - fig_lightweight_baseline_auc.png
  - fig_noise_sensitivity_auc.png
  - Sequential_Adaptation/fig_sequential_auc_heatmap.png
  - Sequential_Adaptation/fig_sequential_all_orders_overlay.png
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.transforms as mtransforms


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "script"
OUT_DIR = SCRIPT_DIR / "AE_model_train_and_detect_output"
REVIEW_DIR = ROOT / "用于查看的中文版"
GENERATED_DIR = ROOT / "paper_TL" / "generated"

FIG_DPI = 300

SCENARIOS = [
    ("结构修复", "Damage_Repaired", "Repair"),
    ("传感器迁移", "Sensor_Offset", "Relocation"),
    ("传感器漂移", "Sensor_Drift", "Drift"),
]

BASELINE_LABELS = {
    "逐通道 z-score 最大值": "z-score max",
    "线性校准 RMS z-score": "linear RMS z",
    "Ledoit-Wolf Mahalanobis": "LW Mahalanobis",
    "CORAL 对齐 + 源域 Mahalanobis": "CORAL Mahalanobis",
    "PCA-192 重构最大残差": "PCA-192 recon.",
    "One-Class SVM": "OCSVM",
    "Isolation Forest": "Isolation Forest",
}

BASELINE_ORDER = [
    "TL",
    "z-score max",
    "linear RMS z",
    "LW Mahalanobis",
    "CORAL Mahalanobis",
    "PCA-192 recon.",
    "OCSVM",
    "Isolation Forest",
]

CUMULATIVE_STATE_LABELS = {
    "Source": "Source",
    "DR": "CDR",
    "SO": "CSO",
    "SD": "CSD",
}

METHOD_COLORS = {
    "TL": "#7BA7BC",
    "z-score max": "#B8B8B8",
    "linear RMS z": "#D8A15D",
    "LW Mahalanobis": "#8DAF77",
    "CORAL Mahalanobis": "#6F8FAF",
    "PCA-192 recon.": "#A66FB5",
    "OCSVM": "#C97A6C",
    "Isolation Forest": "#9E9E9E",
}


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "savefig.dpi": FIG_DPI,
        }
    )


def load_tl_auc_by_scenario() -> dict[str, float]:
    values: dict[str, float] = {}
    for scenario_cn, scenario_dir, _ in SCENARIOS:
        csv_path = OUT_DIR / scenario_dir / "tl_comparison" / "detection_metrics_max.csv"
        df = pd.read_csv(csv_path).set_index("Model")
        values[scenario_cn] = float(df.loc["TL", "AUC"])
    return values


def plot_lightweight_baselines() -> Path:
    apply_plot_style()
    baseline_csv = REVIEW_DIR / "轻量基线复核.csv"
    baseline_df = pd.read_csv(baseline_csv, encoding="utf-8-sig")
    tl_auc = load_tl_auc_by_scenario()

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.2), sharey=True, constrained_layout=True)
    y = np.arange(len(BASELINE_ORDER))

    for ax, (scenario_cn, _, scenario_en) in zip(axes, SCENARIOS):
        rows = [{"method": "TL", "auc": tl_auc[scenario_cn]}]
        sub = baseline_df[baseline_df["场景"] == scenario_cn]
        for _, row in sub.iterrows():
            rows.append(
                {
                    "method": BASELINE_LABELS[str(row["非神经网络基线"])],
                    "auc": float(row["AUC"]),
                }
            )
        values = {row["method"]: row["auc"] for row in rows}
        aucs = [values[m] for m in BASELINE_ORDER]
        colors = [METHOD_COLORS[m] for m in BASELINE_ORDER]

        ax.barh(y, aucs, color=colors, edgecolor="none", height=0.68)
        for yi, val in zip(y, aucs):
            label_x = min(val + 0.012, 0.985)
            ha = "left" if val < 0.97 else "right"
            ax.text(label_x, yi, f"{val:.3f}", va="center", ha=ha, fontsize=7.2)

        ax.set_title(scenario_en, fontsize=9, pad=5)
        ax.set_xlim(0.45, 1.02)
        ax.set_xlabel("AUC")
        ax.set_xticks([0.5, 0.7, 0.9, 1.0])
        ax.tick_params(direction="in", length=2.5)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(BASELINE_ORDER)
    axes[0].invert_yaxis()
    for ax in axes[1:]:
        ax.tick_params(axis="y", left=False, labelleft=False)

    output_path = OUT_DIR / "fig_lightweight_baseline_auc.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_noise_sensitivity() -> Path:
    apply_plot_style()
    noise_csv = REVIEW_DIR / "AE_TL噪声敏感性复核.csv"
    df = pd.read_csv(noise_csv, encoding="utf-8-sig")
    df = df[df["模型"] == "TL"].copy()

    clean_values = []
    noisy_values = []
    x_labels = []
    for scenario_cn, _, scenario_en in SCENARIOS:
        sub = df[df["场景"] == scenario_cn].set_index("噪声设置")
        clean_values.append(float(sub.loc["clean", "AUC"]))
        noisy_values.append(float(sub.loc["5% per-channel train-std Gaussian noise", "AUC"]))
        x_labels.append(scenario_en)

    x = np.arange(len(x_labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.6, 3.1), constrained_layout=True)
    ax.bar(x - width / 2, clean_values, width, label="Clean", color="#7BA7BC", edgecolor="none")
    ax.bar(x + width / 2, noisy_values, width, label="5% noise", color="#C97A6C", edgecolor="none")

    for xpos, val in zip(x - width / 2, clean_values):
        ax.text(xpos, val + 0.006, f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)
    for xpos, val in zip(x + width / 2, noisy_values):
        ax.text(xpos, val + 0.006, f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.80, 1.02)
    ax.set_yticks([0.80, 0.90, 1.00])
    ax.tick_params(direction="in", length=2.5)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

    output_path = OUT_DIR / "fig_noise_sensitivity_auc.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def clean_latex_cell(cell: str) -> str:
    cell = re.sub(r"\\multirow\{[^}]+\}\{\*\}\{([^}]*)\}", r"\1", cell)
    cell = cell.replace(r"\textbf{", "").replace("}", "")
    cell = cell.replace(r"\cline{2-6}", "")
    cell = cell.replace("\\", "").strip()
    return cell


def parse_sequential_table() -> tuple[list[str], np.ndarray]:
    table_path = GENERATED_DIR / "tab_sequential_detection_body_cn.tex"
    rows: list[tuple[str, list[float]]] = []
    current_group = ""
    group_suffix = {
        "结构修复": "CDR",
        "传感器迁移": "CSO",
        "传感器漂移": "CSD",
        "CDR": "CDR",
        "CSO": "CSO",
        "CSD": "CSD",
    }

    for raw_line in table_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if r"\multirow" in line:
            match = re.search(r"\\multirow\{[^}]+\}\{\*\}\{([^}]*)\}", line)
            if match:
                current_group = match.group(1)
        if "&" not in line or not line.endswith(r"\\"):
            continue

        line = line[:-2].strip()
        parts = [clean_latex_cell(part) for part in line.split("&")]
        if len(parts) < 6:
            continue
        order = parts[1] if parts[0] else parts[1]
        if order == "排列顺序":
            continue
        if order == "直接TL":
            label = f"Direct TL ({group_suffix[current_group]})"
        else:
            label = order
        values = [float(clean_latex_cell(cell)) for cell in parts[-4:]]
        rows.append((label, values))

    labels = [row[0] for row in rows]
    matrix = np.array([row[1] for row in rows], dtype=float)
    return labels, matrix


def plot_sequential_auc_heatmap() -> Path:
    apply_plot_style()
    row_labels, matrix = parse_sequential_table()
    col_labels = ["Source", "CDR", "CSO", "CSD"]

    cmap = LinearSegmentedColormap.from_list(
        "auc_heatmap",
        ["#F7F7F7", "#D8E7C8", "#8DAF77", "#4F7FA4"],
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    im = ax.imshow(matrix, cmap=cmap, vmin=0.50, vmax=1.00, aspect="auto")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.tick_params(axis="both", length=0)

    ax.set_xticks(np.arange(len(col_labels) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(row_labels) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    for edge in ["top", "bottom", "left", "right"]:
        ax.spines[edge].set_visible(False)

    for y_idx in [2.5, 5.5]:
        ax.axhline(y_idx, color="#666666", linewidth=0.7)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            text_color = "white" if val >= 0.84 else "#333333"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7.5, color=text_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.86, pad=0.02)
    cbar.set_label("AUC")
    cbar.ax.tick_params(labelsize=8)

    output_path = OUT_DIR / "Sequential_Adaptation" / "fig_sequential_auc_heatmap.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def display_sequence_label(order_label: str) -> str:
    return "→".join(CUMULATIVE_STATE_LABELS.get(part, part) for part in order_label.split("→"))


def plot_sequential_overlay() -> Path:
    seq_path = OUT_DIR / "Sequential_Adaptation" / "sequential_results.json"
    pretrain_csv = OUT_DIR / "Damage_Repaired" / "pretrain" / "training_losses.csv"
    seq_data = json.loads(seq_path.read_text(encoding="utf-8"))
    pretrain_df = pd.read_csv(pretrain_csv)

    pretrain_epochs = len(pretrain_df)
    epochs_per_stage = 200
    pt_x = np.arange(1, pretrain_epochs + 1)
    pt_train = pretrain_df["train_loss"].to_numpy()

    colors = [
        "#2B6DA1",
        "#CB4335",
        "#28874F",
        "#8E44AD",
        "#D68910",
        "#16A085",
    ]

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 9.5,
            "savefig.dpi": FIG_DPI,
        }
    )
    fig, ax = plt.subplots(figsize=(12, 5.6), constrained_layout=True)
    pretrain_color = "#C97A6C"
    ax.plot(pt_x, pt_train, color=pretrain_color, linewidth=0.9, linestyle="-")

    for idx, (order_label, result) in enumerate(seq_data.items()):
        history = result["train_loss_history"]
        epochs = np.array([item["epoch"] for item in history])
        losses = np.array([item["train_loss"] for item in history])
        ax.plot(
            pretrain_epochs + epochs,
            losses,
            color=colors[idx % len(colors)],
            linewidth=0.75,
            alpha=0.9,
        )

    for stage_idx in range(3):
        ax.axvline(
            pretrain_epochs + stage_idx * epochs_per_stage,
            color="#BBBBBB",
            linewidth=0.8,
            linestyle="--",
        )

    blend = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(pretrain_epochs / 2, 0.84, "Pre-train", ha="center", fontsize=12, fontstyle="italic", transform=blend)
    for stage_idx, label in enumerate(["Stage 1", "Stage 2", "Stage 3"]):
        cx = pretrain_epochs + (stage_idx + 0.5) * epochs_per_stage
        ax.text(cx, 0.84, label, ha="center", fontsize=11, fontstyle="italic", transform=blend)

    total_epochs = pretrain_epochs + 3 * epochs_per_stage
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss (MSE)")
    ax.set_yscale("log")
    ax.set_xlim(0, total_epochs)
    ax.set_xticks([0, 500, 1000, 1500, 2000, 2200, 2400, 2600])
    ax.tick_params(direction="in", length=3)

    legend_elements = [Line2D([0], [0], color=pretrain_color, lw=1.0, label="Pre-train")]
    for idx, order_label in enumerate(seq_data.keys()):
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color=colors[idx % len(colors)],
                lw=1.0,
                alpha=0.9,
                label=display_sequence_label(order_label),
            )
        )
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.6,
    )

    output_path = OUT_DIR / "Sequential_Adaptation" / "fig_sequential_all_orders_overlay.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def main() -> None:
    outputs = [
        plot_lightweight_baselines(),
        plot_noise_sensitivity(),
        plot_sequential_auc_heatmap(),
        plot_sequential_overlay(),
    ]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
