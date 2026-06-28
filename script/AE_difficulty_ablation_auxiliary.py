"""
AE_difficulty_ablation_auxiliary.py
====================================
Difficulty ablation engine: runs difficulty x data_size grid experiments
and generates 2D line plots. Shared by all three scenarios.
Imported by run_*_difficulty_ablation.py; not executed directly.
    - AE_model_train_and_detect_auxiliary.py
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from AE_train_model_auxiliary import (
    Autoencoder, apply_freeze_strategy, load_data_from_path,
    train_model, FIG_DPI, PLOT_STYLE, apply_style,
)
from AE_model_train_and_detect_auxiliary import compute_residuals


# ========================================================================
# 2D 折线图
# ========================================================================

# 难度标签映射：xlabel -> LaTeX 图例前缀
_XLABEL_TO_DIFF_LABEL = {
    "Number of Repaired Elements": "$N_r$",
    "Number of Offset Sensors": "$N_o$",
    "Drift Ratio": "$r$",
}

# 莫兰迪6色
_MORANDI_6 = [
    "#8EAABD",  # blue-gray
    "#CB8E8E",  # dusty rose
    "#A8B5A2",  # sage green
    "#C4A882",  # warm sand
    "#9B8EC4",  # lavender
    "#D4A9A9",  # pink
]

_MARKERS_6 = ["o", "s", "^", "D", "v", "P"]


def plot_difficulty_lines_2d(
    difficulty_values: list,
    data_sizes: list,
    loss_matrix: np.ndarray,
    xlabel: str,
    output_path: str,
    before_tl_values: Optional[np.ndarray] = None,
    ylabel: str = "Validation Loss (MSE)",
) -> str:
    """
    2D折线图：每个 difficulty 一条线，X轴=训练样本数，Y轴=验证损失

    Args:
        difficulty_values: 难度值列表，如 [0, 4, 8, 12, 16, 20]
        data_sizes:        训练样本数列表，如 [50, 100, 200, 400]
        loss_matrix:       shape = (len(data_sizes), len(difficulty_values))
        xlabel:            场景难度标签，如 "Number of Repaired Elements"
        output_path:       图像保存路径（含文件名）
        before_tl_values:  可选，Before TL 参考值，长度 = len(difficulty_values)
        ylabel:            Y轴标签

    Returns:
        保存的图像路径
    """
    apply_style(PLOT_STYLE)

    diff_label = _XLABEL_TO_DIFF_LABEL.get(xlabel, "Difficulty")
    lw = 1.5
    ms = 7

    # figsize 高度 3.6 为顶部外部图例预留空间（禁止 2.8-3.0 扁平值）
    fig, ax = plt.subplots(figsize=(10, 3.6), constrained_layout=True)

    # 主数据线：每个 difficulty 一条
    for j, diff in enumerate(difficulty_values):
        y_vals = loss_matrix[:, j]
        color = _MORANDI_6[j % len(_MORANDI_6)]
        marker = _MARKERS_6[j % len(_MARKERS_6)]
        ax.plot(
            data_sizes, y_vals,
            marker=marker, markersize=ms, linewidth=lw,
            color=color,
            label=f"{diff_label} = {diff}",
        )

    # Before-TL reference lines removed: btl_max is typically one to two orders
    # of magnitude higher than post-TL losses (e.g. SO: 0.11 vs 0.008), so
    # axhline(btl_max) drags the auto ylim up and compresses main curves into
    # the bottom of the plot area. Before-TL values are reported in the caption
    # text and Table of detection metrics instead.

    ax.set_xlabel(r"Number of Training Samples ($N_{\mathrm{tgt}}$)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(data_sizes)
    ax.set_xticklabels([str(s) for s in data_sizes])

    # 图框外部上方横排图例，字号 12 (比刻度 16 小 4pt)
    # 单行铺开：figsize 宽 10in 可容纳 6~8 条 "$X$ = N" 形式图例条目
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="outside upper center",
               ncol=len(labels),
               frameon=False, fontsize=12)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=FIG_DPI)
    plt.close(fig)

    print(f"[保存] {os.path.basename(output_path)}  (2D line plot)")
    print(f"       {len(difficulty_values)} difficulty x {len(data_sizes)} data_sizes")
    return output_path


# ========================================================================
# 结果保存 / 加载
# ========================================================================

def save_ablation_results(
    difficulty_values: list,
    data_sizes: list,
    loss_matrix: np.ndarray,
    before_tl_values: np.ndarray,
    output_path: str,
) -> str:
    """
    保存消融结果到 CSV

    CSV 格式（长表）：
        difficulty, data_size, val_loss, before_tl
        0,          50,        0.00xxx,  0.00yyy
        0,          100,       0.00xxx,  0.00yyy
        ...

    Args:
        difficulty_values: difficulty 轴的值列表
        data_sizes:        data_size 轴的值列表
        loss_matrix:       shape = (len(data_sizes), len(difficulty_values))
        before_tl_values:  shape = (len(difficulty_values),)
        output_path:       CSV 文件保存路径

    Returns:
        保存的 CSV 路径
    """
    rows = []
    for j, diff in enumerate(difficulty_values):
        btl = float(before_tl_values[j])
        for i, ds in enumerate(data_sizes):
            rows.append({
                "difficulty": diff,
                "data_size": ds,
                "val_loss": float(loss_matrix[i, j]),
                "before_tl": btl,
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False, float_format="%.8f")

    print(f"[保存] {os.path.basename(output_path)}")
    print(f"       {len(difficulty_values)} difficulty x {len(data_sizes)} data_size = {len(rows)} rows")
    return output_path


def load_ablation_results(csv_path: str) -> Tuple[list, list, np.ndarray, np.ndarray]:
    """
    从 CSV 加载消融结果

    Args:
        csv_path: CSV 文件路径

    Returns:
        (difficulty_values, data_sizes, loss_matrix, before_tl_values)
        - difficulty_values: list, 排序后的 difficulty 值
        - data_sizes:        list, 排序后的 data_size 值
        - loss_matrix:       np.ndarray, shape = (len(data_sizes), len(difficulty_values))
        - before_tl_values:  np.ndarray, shape = (len(difficulty_values),)
    """
    df = pd.read_csv(csv_path)

    # 提取唯一值并排序
    difficulty_values = sorted(df["difficulty"].unique().tolist())
    data_sizes = sorted(df["data_size"].unique().tolist())

    # 构建 loss_matrix
    n_ds = len(data_sizes)
    n_diff = len(difficulty_values)
    loss_matrix = np.full((n_ds, n_diff), np.nan)
    before_tl_values = np.full(n_diff, np.nan)

    diff_to_j = {d: j for j, d in enumerate(difficulty_values)}
    ds_to_i = {d: i for i, d in enumerate(data_sizes)}

    for _, row in df.iterrows():
        j = diff_to_j[row["difficulty"]]
        i = ds_to_i[row["data_size"]]
        loss_matrix[i, j] = row["val_loss"]
        before_tl_values[j] = row["before_tl"]

    print(f"[加载] {os.path.basename(csv_path)}")
    print(f"       difficulty: {difficulty_values}")
    print(f"       data_sizes: {data_sizes}")
    return difficulty_values, data_sizes, loss_matrix, before_tl_values


# ========================================================================
# 统一消融引擎（框架）
# ========================================================================

def run_difficulty_ablation(
    scenario_name: str,
    difficulty_values: list,
    data_sizes: list,
    pretrain_model_path: str,
    data_loader_func: Callable,
    output_dir: str,
    settings: dict,
    device: torch.device,
    xlabel: str = "Difficulty",
    freeze_strategy: str = "none",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    统一消融引擎：对任意场景执行 difficulty x data_size 网格实验

    对每个 (difficulty, data_size) 组合：
        1) 调用 data_loader_func(difficulty, data_size) 获取 (V_train, V_val)
        2) 从 pretrain 权重初始化模型 -> 微调 -> 记录 val_loss
    对每个 difficulty：
        3) 计算 Before TL（pretrain 模型直接在 new_data 验证集上的 val_loss）
    最后保存 CSV 并生成 2D 折线图。

    Args:
        scenario_name:      场景名称 "Damage_Repaired" / "Sensor_Offset" / "Sensor_Drift"
        difficulty_values:  difficulty 轴的值列表
        data_sizes:         data_size 轴的值列表
        pretrain_model_path: pretrain 模型 .pth 路径
        data_loader_func:   callable(difficulty, data_size) -> (V_train, V_val)
                            V_train: np.ndarray (n_samples, D), 已截取 data_size 个样本
                            V_val:   np.ndarray (M, D), 验证集
        output_dir:         输出根目录
        settings:           训练配置 dict（需包含 'model' 和 'training' 子项）
        device:             torch.device
        xlabel:             场景难度标签（用于图例前缀推导）
        freeze_strategy:    冻结策略，默认 'none'

    Returns:
        loss_matrix:       np.ndarray shape=(len(data_sizes), len(difficulty_values))
        before_tl_values:  np.ndarray shape=(len(difficulty_values),)
    """
    print(f"\n{'=' * 60}")
    print(f"[{scenario_name}] Difficulty x Data-size Ablation")
    print(f"{'=' * 60}")
    print(f"[配置] difficulty: {difficulty_values}")
    print(f"[配置] data_sizes: {data_sizes}")
    print(f"[配置] freeze_strategy: {freeze_strategy}")
    print(f"[配置] pretrain: {pretrain_model_path}")

    os.makedirs(output_dir, exist_ok=True)

    ae_config = settings["model"]
    training_config = settings["training"]

    n_diff = len(difficulty_values)
    n_ds = len(data_sizes)
    loss_matrix = np.full((n_ds, n_diff), np.nan)
    before_tl_values = np.full(n_diff, np.nan)

    all_model_paths: List[str] = []

    # ------------------------------------------------------------------
    # Step 1: 为每个 difficulty 计算 Before TL
    #         （pretrain 模型直接推理 new_data 验证集的 MSE）
    # ------------------------------------------------------------------
    print(f"\n[Phase 1] Computing Before-TL baselines ...")

    # 获取输入维度 D（用第一个 difficulty 的数据）
    _V_train_probe, _V_val_probe = data_loader_func(difficulty_values[0], data_sizes[0])
    D = _V_train_probe.shape[1]

    # 加载 pretrain 模型
    pretrain_model = Autoencoder(
        D,
        ae_config["encoder_dims"],
        ae_config["latent_dim"],
        ae_config["decoder_dims"],
        ae_config["dropout"],
        ae_config["activation"],
    ).to(device)
    pretrain_model.load_state_dict(torch.load(pretrain_model_path, map_location=device))
    pretrain_model.eval()

    for j, diff in enumerate(difficulty_values):
        # 只需要验证集即可；data_size 不影响验证集
        _, V_val = data_loader_func(diff, data_sizes[0])
        # 计算 MSE
        residuals = compute_residuals(V_val, pretrain_model, device)
        mse = float(np.mean(residuals ** 2))
        before_tl_values[j] = mse
        print(f"  difficulty={diff}  Before-TL MSE = {mse:.6f}")

    # ------------------------------------------------------------------
    # Step 2: 对每个 (difficulty, data_size) 组合训练 TL 模型
    # ------------------------------------------------------------------
    print(f"\n[Phase 2] Training TL models ({n_diff} x {n_ds} = {n_diff * n_ds} runs) ...")

    # 从 training_config 读取新参数（不影响旧配置的默认行为）
    reinit_decoder = training_config.get("reinit_decoder", False)
    encoder_lr_val = training_config.get("encoder_lr", None)
    l2_sp_alpha_val = training_config.get("l2_sp_alpha", 0.0)

    for j, diff in enumerate(difficulty_values):
        for i, ds in enumerate(data_sizes):
            diff_str = f"{diff}".replace(".", "p")
            run_name = f"diff_{diff_str}_n{ds}"
            run_dir = os.path.join(output_dir, run_name)
            os.makedirs(run_dir, exist_ok=True)

            model_path = os.path.join(run_dir, "autoencoder.pth")
            losses_path = os.path.join(run_dir, "training_losses.csv")

            print(f"\n  [{run_name}]")

            # 加载数据
            V_train, V_val = data_loader_func(diff, ds)

            # 微调
            model, train_losses, val_losses = train_model(
                V_train, V_val, device,
                epochs=training_config["finetune"]["epochs"],
                lr=training_config["finetune"]["lr"],
                batch_size=training_config["finetune"]["batch_size"],
                output_dir=run_dir,
                ae_config=ae_config,
                training_config=training_config,
                plot_style=PLOT_STYLE,
                fig_dpi=FIG_DPI,
                pretrain_model_path=pretrain_model_path,
                model_name=f"TL-{run_name}",
                train_shuffle=training_config.get("shuffle", True),
                reinit_decoder=reinit_decoder,
                encoder_lr=encoder_lr_val,
                l2_sp_alpha=l2_sp_alpha_val,
            )
            final_val = val_losses[-1] if val_losses else float("nan")

            loss_matrix[i, j] = final_val
            all_model_paths.append(model_path)
            print(f"    val_loss = {final_val:.6f}")

    # ------------------------------------------------------------------
    # Step 3: 保存 CSV
    # ------------------------------------------------------------------
    csv_path = os.path.join(output_dir, f"difficulty_ablation_{scenario_name}.csv")
    save_ablation_results(difficulty_values, data_sizes, loss_matrix, before_tl_values, csv_path)

    # ------------------------------------------------------------------
    # Step 4: 生成 2D 折线图
    # ------------------------------------------------------------------
    fig_path = os.path.join(output_dir, f"difficulty_surface_{scenario_name}.png")
    plot_difficulty_lines_2d(
        difficulty_values=difficulty_values,
        data_sizes=data_sizes,
        loss_matrix=loss_matrix,
        xlabel=xlabel,
        output_path=fig_path,
        before_tl_values=before_tl_values,
    )

    print(f"\n[完成] {output_dir}")
    return loss_matrix, before_tl_values
