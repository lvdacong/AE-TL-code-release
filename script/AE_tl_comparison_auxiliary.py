"""
AE_tl_comparison_auxiliary.py
==============================
TL comparison engine: runs TL vs OldModel vs FromScratch, generates
training curves, residual analysis, ROC, and 3D comparison figures.
Called by run_*_tl_comparison.py; not executed directly.
    3. FromScratch: 在新数据上从头训练

输出：
    - training_curve_comparison.png - 三模型训练曲线对比
    - comparison_residuals.png - 残差分析(2行×3列)
    - comparison_3d.png - 3D渲染对比(2行×3列)

依赖：
    - AE_train_model_auxiliary.py
    - AE_model_train_and_detect_auxiliary.py
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from AE_train_model_auxiliary import (
    Autoencoder, apply_freeze_strategy, load_data_from_path,
    train_model, plot_dual_comparison_curves,
    FIG_DPI, PLOT_STYLE, apply_style,
)
import copy as _copy
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 强制使用与 merge_images_grid (PIL ImageFont) 同一个字体文件
# 以保证 comparison_residuals (matplotlib) 与 comparison_3d (PIL 合成)
# 的 column/row 标题字形完全一致
def _find_times_font_file() -> Optional[str]:
    candidates = [
        "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/Times New Roman.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        "/System/Library/Fonts/Times.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

_TIMES_FONT_PATH = _find_times_font_file()
_TIMES_FP = FontProperties(fname=_TIMES_FONT_PATH) if _TIMES_FONT_PATH else None


def _times_fp(size: int) -> Optional[FontProperties]:
    """Return a sized copy of the base Times FontProperties.

    FontProperties 会覆盖 set_title 的 fontsize 参数，所以必须把 size
    写进 FontProperties 本身。
    """
    if _TIMES_FP is None:
        return None
    fp = _copy.copy(_TIMES_FP)
    fp.set_size(size)
    return fp

from AE_model_train_and_detect_auxiliary import (
    compute_residuals, compute_normal_params,
    compute_anomaly_scores, compute_detection_metrics,
    visualize_combined_residual_analysis,
    merge_images_grid, HAS_PYVISTA,
    parse_elsets_from_inp, extract_middlewhole_submesh,
    interpolate_to_nodes, render_damage_3d, load_camera_position
)

# 尝试导入pyvista
try:
    import pyvista as pv
except ImportError:
    pv = None


def load_render_assets(settings: dict) -> Optional[dict]:
    """
    加载3D渲染所需的资源

    Args:
        settings: 全局设置

    Returns:
        渲染资源字典，包含base_mesh, id_mapping, measure_ids, middlewhole_mesh, camera_position
        如果加载失败返回None
    """
    if not HAS_PYVISTA:
        print("[警告] PyVista未安装，跳过3D渲染资源加载")
        return None

    workspace_dir = settings['global']['workspace_dir']
    detection = settings['detection']

    # 构建路径
    ac_output_dir = os.path.join(workspace_dir, detection['ac_output_dir'])
    vtu_path = os.path.join(ac_output_dir, detection['vtu_file'])
    id_mapping_path = os.path.join(ac_output_dir, detection['id_mapping_file'])
    measures_path = os.path.join(ac_output_dir, detection['measures_file'])
    inp_path = detection['inp_file_path']
    camera_path = os.path.join(workspace_dir, detection['camera_config_path'])

    # 检查文件是否存在
    missing_files = []
    if not os.path.exists(vtu_path):
        missing_files.append(f"VTU文件: {vtu_path}")
    if not os.path.exists(id_mapping_path):
        missing_files.append(f"ID映射文件: {id_mapping_path}")
    if not os.path.exists(measures_path):
        missing_files.append(f"测点文件: {measures_path}")
    if not os.path.exists(inp_path):
        missing_files.append(f"INP文件: {inp_path}")
    if not os.path.exists(camera_path):
        missing_files.append(f"相机配置文件: {camera_path}")

    if missing_files:
        print(f"[警告] 以下渲染资源文件不存在，跳过3D渲染:")
        for f in missing_files:
            print(f"  - {f}")
        return None

    try:
        # 加载VTU网格
        print(f"[3D渲染] 加载VTU网格...")
        base_mesh = pv.read(vtu_path)

        # 加载ID映射
        id_mapping_df = pd.read_csv(id_mapping_path)
        id_mapping = dict(zip(id_mapping_df['abaqus_id'], id_mapping_df['vtu_index']))

        # 加载测点ID - 第一行是标题'all_measures'，需要跳过并转换为整数
        measures_df = pd.read_csv(measures_path, header=0)
        measure_ids = measures_df.iloc[:, 0].astype(int).tolist()
        print(f"[3D渲染] 测点数量: {len(measure_ids)}")

        # 解析INP获取元素集ID
        elsets = parse_elsets_from_inp(inp_path, detection['middlewhole_elset'])
        middlewhole_ids = elsets.get(detection['middlewhole_elset'], [])

        if not middlewhole_ids:
            print(f"[警告] 未找到元素集 '{detection['middlewhole_elset']}'")
            return None

        # 提取子网格
        middlewhole_mesh, _ = extract_middlewhole_submesh(base_mesh, middlewhole_ids, id_mapping)
        print(f"[3D渲染] 子网格: {middlewhole_mesh.n_cells} cells, {middlewhole_mesh.n_points} points")

        # 加载相机位置
        camera_position = load_camera_position(camera_path)

        return {
            'base_mesh': base_mesh,
            'middlewhole_mesh': middlewhole_mesh,
            'id_mapping': id_mapping,
            'measure_ids': measure_ids,
            'camera_position': camera_position
        }

    except Exception as e:
        print(f"[警告] 加载渲染资源失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def _compute_and_save_detection_metrics(
    comparison_results: dict,
    V_test: np.ndarray,
    V_control: np.ndarray,
    output_dir: str,
    device: torch.device
) -> None:
    """
    计算三模型的损伤检测指标并保存CSV和合并版ROC曲线图（mean + max 并排）

    生成文件：
        - detection_metrics.csv / detection_metrics_max.csv
        - roc_comparison_merged.png（(a) Mean + (b) Max 并排，直接用于论文）

    Args:
        comparison_results: 包含 TL/OldModel/FromScratch 模型的字典
        V_test: 损伤测试数据 (N_damage, D)
        V_control: 正常对照数据 (N_control, D)
        output_dir: 输出目录
        device: 计算设备
    """
    model_names = ['TL', 'OldModel', 'FromScratch']
    result_keys = ['tl', 'pretrain', 'fromscratch']
    display_labels = {
        'TL': 'TL',
        'OldModel': 'Unadapted source',
        'FromScratch': 'Scratch-trained',
    }
    roc_colors = {'TL': '#2166ac', 'OldModel': '#d73027', 'FromScratch': '#878787'}
    roc_ls     = {'TL': '-',       'OldModel': '--',       'FromScratch': ':'}

    # 收集两种评分方式的ROC数据
    all_roc_data = {}  # method -> {name: (fprs, tprs, auc)}

    for method, csv_name, score_label in [
        ('mean', 'detection_metrics.csv', 'Mean-Channel MAE'),
        ('max',  'detection_metrics_max.csv', 'Max-Channel MAE'),
    ]:
        rows = []
        roc_data = {}

        for name, key in zip(model_names, result_keys):
            model = comparison_results[key]['model']
            scores_damage = compute_anomaly_scores(V_test, model, device, method=method)
            scores_control = compute_anomaly_scores(V_control, model, device, method=method)
            metrics = compute_detection_metrics(scores_damage, scores_control)
            rows.append({
                'Model': name,
                'AUC': round(metrics['auc'], 4),
                'AUC_CI_lo': round(metrics['auc_ci_lo'], 4),
                'AUC_CI_hi': round(metrics['auc_ci_hi'], 4),
                'TPR@FPR5%': round(metrics['tpr_at_fpr'], 4),
                'Best_F1': round(metrics['best_f1'], 4)
            })
            roc_data[name] = (metrics['fprs'], metrics['tprs'], metrics['auc'])

        all_roc_data[method] = roc_data

        # 保存CSV
        csv_path = os.path.join(output_dir, csv_name)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"[检测指标] 已保存 ({method}): {os.path.relpath(csv_path)}")

    # 绘制合并版 ROC：(a) Mean + (b) Max 并排，正方形子图
    roc_style = PLOT_STYLE.copy()
    roc_style.update({
        "font.size": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 12,
    })
    apply_style(roc_style)

    # figsize 高度 7.3 预留顶部外部图例空间（原 6.5 压扁方形 Axes）
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.3), constrained_layout=True)
    panel_labels = [
        ('mean', '(a) Mean-Channel MAE'),
        ('max', '(b) Max-Channel MAE'),
    ]

    handles_for_legend = []
    for ax, (method, panel_title) in zip(axes, panel_labels):
        roc_data = all_roc_data[method]
        for name in model_names:
            fprs, tprs, auc = roc_data[name]
            line, = ax.plot(fprs, tprs, label=display_labels[name],
                    color=roc_colors.get(name, '#333333'),
                    lw=1.5, linestyle=roc_ls.get(name, '-'))
            if method == 'mean':
                handles_for_legend.append(line)
        ax.plot([0, 1], [0, 1], color='#cccccc', lw=0.8, linestyle='--')
        ax.set_xlabel("False Positive Rate")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(direction='in')
        ax.grid(False)
        ax.set_title(panel_title, fontsize=20, pad=10)

    # 只在左子图显示y轴标签
    axes[0].set_ylabel("True Positive Rate")
    axes[1].set_ylabel("")

    # 图例放在整图外部上方横排，不含AUC数值
    fig.legend(handles_for_legend, [display_labels[name] for name in model_names],
               loc="outside upper center",
               ncol=3, frameon=False, fontsize=12)

    merged_roc_path = os.path.join(output_dir, "roc_comparison_merged.png")
    fig.savefig(merged_roc_path, dpi=FIG_DPI)
    plt.close(fig)
    apply_style(PLOT_STYLE)
    print(f"[检测指标] 已保存 (merged): {os.path.relpath(merged_roc_path)}")


def run_tl_comparison(
    scenario_name: str,
    scenario_config: dict,
    pretrain_pth: str,
    settings: dict,
    device: torch.device,
    output_subdir: str = "tl_comparison",
) -> Dict[str, dict]:
    """
    运行三模型对比实验

    比较三种模型：
    1. TL: 预训练 → 微调
    2. OldModel: 仅预训练（不微调）
    3. FromScratch: 从头训练

    Args:
        scenario_name: 场景名称（如 'damage_repair'）
        scenario_config: 场景配置
        pretrain_pth: 预训练模型路径
        settings: 全局设置
        device: 计算设备

    Returns:
        对比结果字典: {
            'pretrain': {'pth': str, 'train_losses': List, 'val_losses': List, 'model': nn.Module},
            'tl': {...},
            'fromscratch': {...}
        }
    """
    print(f"\n{'='*60}")
    print(f"[{scenario_name}] TL效果对比实验（三模型对比）")
    print(f"{'='*60}")

    tl_config = scenario_config['tl_comparison']
    train_samples = tl_config['train_samples']
    freeze_strategy = tl_config.get('freeze_strategy', 'none')

    # 获取路径
    workspace_dir = settings['global']['workspace_dir']
    output_base = os.path.join(workspace_dir, settings['global']['output_dir'])
    preprocess_base = os.path.join(workspace_dir, settings['global']['preprocess_output_base'])

    # 创建输出目录
    output_dir = os.path.join(output_base, scenario_name, output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    # 加载新数据
    new_data_path = os.path.join(preprocess_base, scenario_config['new_data_folder'])
    V_train, V_val = load_data_from_path(
        new_data_path,
        settings['training']['val_samples'],
        "新数据"
    )

    # 限制训练样本数
    V_train_limited = V_train[:train_samples]
    print(f"[配置] 训练样本数: {train_samples}")
    print(f"[配置] 冻结策略: {freeze_strategy}")

    # 构建配置
    ae_config = settings['model']
    training_config = settings['training']
    D = V_train.shape[1]

    # 存储结果
    comparison_results = {}

    # ========== 1. 预训练模型（OldModel） ==========
    print(f"\n[模型1] Pre-train (OldModel)")

    # 加载预训练模型和损失记录
    # 从实际的pretrain_pth路径推导目录（支持跨场景复用预训练模型）
    pretrain_dir = os.path.dirname(pretrain_pth)
    pretrain_losses_path = os.path.join(pretrain_dir, 'training_losses.csv')

    model_pretrain = Autoencoder(
        D,
        ae_config['encoder_dims'],
        ae_config['latent_dim'],
        ae_config['decoder_dims'],
        ae_config['dropout'],
        ae_config['activation']
    ).to(device)
    model_pretrain.load_state_dict(torch.load(pretrain_pth, map_location=device))

    if os.path.exists(pretrain_losses_path):
        pretrain_losses_df = pd.read_csv(pretrain_losses_path)
        pretrain_train_losses = pretrain_losses_df['train_loss'].tolist()
        pretrain_val_losses = pretrain_losses_df['val_loss'].tolist()
    else:
        # 如果没有损失记录，创建空列表
        pretrain_train_losses = []
        pretrain_val_losses = []

    comparison_results['pretrain'] = {
        'pth': pretrain_pth,
        'train_losses': pretrain_train_losses,
        'val_losses': pretrain_val_losses,
        'model': model_pretrain
    }
    print(f"  [加载] 预训练模型: {pretrain_pth}")

    # ========== 2. 迁移学习模型（TL） ==========
    print(f"\n[模型2] Transfer Learning (TL)")

    tl_output_dir = os.path.join(output_dir, 'TL')
    os.makedirs(tl_output_dir, exist_ok=True)

    tl_model_path = os.path.join(tl_output_dir, "autoencoder.pth")

    # 训练新模型（会生成 pth, csv, png）
    model_tl, tl_train_losses, tl_val_losses = train_tl_model(
        V_train_limited, V_val, pretrain_pth, freeze_strategy,
        tl_output_dir, ae_config, training_config, device
    )

    comparison_results['tl'] = {
        'pth': tl_model_path,
        'train_losses': tl_train_losses,
        'val_losses': tl_val_losses,
        'model': model_tl
    }

    # ========== 3. 从头训练模型（FromScratch） ==========
    print(f"\n[模型3] From Scratch")

    scratch_output_dir = os.path.join(output_dir, 'fromscratch')
    os.makedirs(scratch_output_dir, exist_ok=True)

    scratch_model_path = os.path.join(scratch_output_dir, "autoencoder.pth")

    # 训练新模型（会生成 pth, csv, png）
    model_scratch, scratch_train_losses, scratch_val_losses = train_from_scratch_model(
        V_train_limited, V_val,
        scratch_output_dir, ae_config, training_config, device
    )

    comparison_results['fromscratch'] = {
        'pth': scratch_model_path,
        'train_losses': scratch_train_losses,
        'val_losses': scratch_val_losses,
        'model': model_scratch
    }

    # ========== 收集路径 ==========
    all_model_paths = [pretrain_pth, tl_model_path, scratch_model_path]

    # ========== 统一加载 V_test 和 V_control ==========
    workspace_dir = settings['global']['workspace_dir']
    preprocess_base = os.path.join(workspace_dir, settings['global']['preprocess_output_base'])
    test_data_path = os.path.join(preprocess_base, scenario_config['test_damage_folder'])
    test_npz = os.path.join(test_data_path, 'preprocessed_data_raw.npz')
    new_data_path = os.path.join(preprocess_base, scenario_config['new_data_folder'])

    V_test = None
    V_control = None
    if os.path.exists(test_npz):
        V_test = np.load(test_npz)['V'].astype(np.float32)
        print(f"[加载] 测试数据: {V_test.shape}")
    else:
        print(f"[警告] 测试数据不存在: {test_npz}")

    # 计算统一的 V_control：样本数与 V_test 保持一致，确保所有场景的
    # 三模型对比图（热力图、3D渲染、检测指标）在视觉和统计上可比。
    # 原实现中 DR 场景 fallback 到最后 100 个会与 V_val 完全重叠，SO/SD
    # 场景取整个 holdout 区间（1500 样本）导致三场景 control 样本数差 15×。
    new_data_npz = os.path.join(new_data_path, 'preprocessed_data_raw.npz')
    if os.path.exists(new_data_npz):
        V_all = np.load(new_data_npz)['V'].astype(np.float32)
        val_samples = settings['training']['val_samples']
        N = V_all.shape[0]
        n_control = V_test.shape[0] if V_test is not None else val_samples
        pure_start = train_samples
        pure_end = N - val_samples
        if pure_end - pure_start >= n_control:
            # 优先使用与 train/val 都不重叠的 pure holdout 区间
            V_control = V_all[pure_start:pure_start + n_control]
            print(f"[对照组] 独立 holdout control: 样本 [{pure_start}:{pure_start + n_control}], "
                  f"共 {V_control.shape[0]} 个 (与 train 0-{train_samples} 及 val {pure_end}-{N} 无重叠)")
        else:
            # 数据量不足时退化到 V_val（held-out healthy，未参与梯度更新）
            fb_start = N - val_samples
            V_control = V_all[fb_start:fb_start + n_control]
            print(f"[对照组] fallback control (= V_val): 样本 [{fb_start}:{fb_start + n_control}], "
                  f"共 {V_control.shape[0]} 个 (DR 类数据受限场景)")

    # ========== 生成可视化 ==========

    # 1. 训练曲线对比
    if pretrain_val_losses and tl_val_losses and scratch_val_losses:
        plot_dual_comparison_curves(
            old_val_losses=pretrain_val_losses,
            transfer_val_losses=tl_val_losses,
            scratch_val_losses=scratch_val_losses,
            output_dir=output_dir,
            plot_style=PLOT_STYLE,
            fig_dpi=FIG_DPI,
            filename="training_curve_comparison.png",
            old_train_losses=pretrain_train_losses,
            transfer_train_losses=tl_train_losses,
            scratch_train_losses=scratch_train_losses
        )

    # 2. 残差分析（依赖于所有模型）
    visualize_tl_comparison_results(
        comparison_results,
        scenario_config,
        output_dir,
        settings,
        device,
        all_model_paths,
        skip_3d=True,  # 这里只生成残差图，3D渲染单独处理
        V_control_override=V_control
    )

    # 3. 3D渲染
    if settings['detection'].get('render_3d', False) and HAS_PYVISTA:
        if V_test is not None and V_control is not None:
            render_tl_comparison_3d(
                comparison_results,
                scenario_config,
                output_dir,
                settings,
                device,
                V_test,
                V_control
            )
        else:
            print(f"[警告] 测试数据或对照组数据不存在，跳过3D渲染")

    # 4. 损伤检测指标（ROC-AUC / TPR@FPR5% / Best F1）
    if V_test is not None and V_control is not None:
        _compute_and_save_detection_metrics(
            comparison_results, V_test, V_control, output_dir, device
        )
    else:
        print(f"[警告] 测试数据或对照组数据不存在，跳过检测指标计算")

    print(f"\n[完成] {os.path.relpath(output_dir)}")
    return comparison_results


def train_tl_model(
    V_train: np.ndarray,
    V_val: np.ndarray,
    pretrain_pth: str,
    freeze_strategy: str,
    output_dir: str,
    ae_config: dict,
    training_config: dict,
    device: torch.device,
) -> Tuple[nn.Module, List[float], List[float]]:
    """
    训练迁移学习模型

    Args:
        V_train: 训练数据
        V_val: 验证数据
        pretrain_pth: 预训练模型路径
        freeze_strategy: 冻结策略
        output_dir: 输出目录
        ae_config: 模型配置
        training_config: 训练配置
        device: 计算设备

    Returns:
        (model, train_losses, val_losses)
    """
    D = V_train.shape[1]

    # 创建模型
    model = Autoencoder(
        D,
        ae_config['encoder_dims'],
        ae_config['latent_dim'],
        ae_config['decoder_dims'],
        ae_config['dropout'],
        ae_config['activation']
    ).to(device)

    # 加载预训练权重
    print(f"  [加载] 预训练权重: {pretrain_pth}")
    checkpoint = torch.load(pretrain_pth, map_location=device)
    model.load_state_dict(checkpoint)

    # 应用冻结策略
    apply_freeze_strategy(model, freeze_strategy, verbose=True)

    # 训练
    model, train_losses, val_losses = train_model(
        V_train, V_val, device,
        epochs=training_config['finetune']['epochs'],
        lr=training_config['finetune']['lr'],
        batch_size=training_config['finetune']['batch_size'],
        output_dir=output_dir,
        ae_config=ae_config,
        training_config=training_config,
        plot_style=PLOT_STYLE,
        fig_dpi=FIG_DPI,
        pretrain_model_path=pretrain_pth,
        model_name="Transfer Learning",
        train_shuffle=training_config.get('shuffle', True),
        encoder_lr=training_config.get("encoder_lr"),
        l2_sp_alpha=training_config.get("l2_sp_alpha", 0.0),
    )

    return model, train_losses, val_losses


def train_from_scratch_model(
    V_train: np.ndarray,
    V_val: np.ndarray,
    output_dir: str,
    ae_config: dict,
    training_config: dict,
    device: torch.device
) -> Tuple[nn.Module, List[float], List[float]]:
    """
    从头训练模型（无预训练）

    Args:
        V_train: 训练数据
        V_val: 验证数据
        output_dir: 输出目录
        ae_config: 模型配置
        training_config: 训练配置
        device: 计算设备

    Returns:
        (model, train_losses, val_losses)
    """
    # 训练（不加载预训练权重）
    model, train_losses, val_losses = train_model(
        V_train, V_val, device,
        epochs=training_config['finetune']['epochs'],
        lr=training_config['finetune']['lr'],
        batch_size=training_config['finetune']['batch_size'],
        output_dir=output_dir,
        ae_config=ae_config,
        training_config=training_config,
        plot_style=PLOT_STYLE,
        fig_dpi=FIG_DPI,
        pretrain_model_path=None,  # 关键：不加载预训练
        model_name="From Scratch",
        train_shuffle=training_config.get('shuffle', True)
    )

    return model, train_losses, val_losses


def _plot_comparison_residuals_grid(
    all_residuals: dict,
    column_titles: List[str],
    output_path: str,
    n_samples: int = 5
) -> None:
    """
    直接用 matplotlib 生成 2x3 残差组合图（热力图 + 均值曲线双段式）。

    每列 = 一个模型，每行 = Test(Damaged) / Control(Healthy)。
    每个单元格包含上方热力图和下方均值|残差|曲线。
    精简轴标题：共享的标签只出现一次。
    列/行大标题字号与 comparison_3d 保持一致。
    """
    from AE_train_model_auxiliary import apply_style, PLOT_STYLE, FIG_DPI
    from matplotlib.gridspec import GridSpec

    apply_style(PLOT_STYLE)

    model_keys = ['pretrain', 'tl', 'fromscratch']
    row_labels = ["Test (Damaged)", "Control (Healthy)"]
    data_types = ['damage', 'control']

    # Morandi blue for mean curve
    MEAN_COLOR = "#7BA7BC"

    # Reduced figsize to keep width < 6000px at 300 dpi (20*300=6000)
    fig = plt.figure(figsize=(19.5, 8))
    outer_gs = GridSpec(2, 3, figure=fig, hspace=0.12, wspace=0.10,
                        left=0.055, right=0.95, top=0.88, bottom=0.06)

    for col_idx, (model_key, col_title) in enumerate(zip(model_keys, column_titles)):
        for row_idx, data_type in enumerate(data_types):
            residuals = all_residuals[model_key][data_type]
            N, D = residuals.shape
            abs_res = np.abs(residuals)
            mean_abs = abs_res.mean(axis=0)

            inner_gs = outer_gs[row_idx, col_idx].subgridspec(6, 1, hspace=0.08)
            ax_heat = fig.add_subplot(inner_gs[0:4, 0])
            ax_mean = fig.add_subplot(inner_gs[4:6, 0], sharex=ax_heat)

            # --- Heatmap ---
            vmax = np.percentile(abs_res, 99.8)
            ax_heat.imshow(abs_res, aspect='auto', cmap='YlOrRd',
                          interpolation='nearest', vmin=0, vmax=vmax)
            ystep = 20 if N <= 100 else (50 if N <= 300 else (100 if N <= 600 else 200))
            ax_heat.set_yticks(np.arange(0, N, ystep))
            ax_heat.set_xlim([0, D - 1])
            ax_heat.tick_params(labelsize=16)
            plt.setp(ax_heat.get_xticklabels(), visible=False)

            # --- Mean residual curve ---
            # Single-curve cell; column titles state the model and row labels state
            # Test/Control, so no per-subplot legend is needed.
            ax_mean.plot(range(D), mean_abs, color=MEAN_COLOR, linewidth=1.0)
            ax_mean.fill_between(range(D), 0, mean_abs, color=MEAN_COLOR, alpha=0.2)
            ax_mean.set_xlim([0, D - 1])
            ax_mean.set_xticks(np.arange(0, D, 50))
            ax_mean.tick_params(labelsize=16)
            if col_idx == 0:
                ax_mean.set_ylabel('Mean |Residual|', fontsize=16)

            if row_idx == 1 and col_idx == 0:
                ax_mean.set_xlabel('Channel Index', fontsize=20)
            else:
                plt.setp(ax_mean.get_xticklabels(), visible=(row_idx == 1))

            if row_idx == 0:
                fp_title = _times_fp(26)
                if fp_title is not None:
                    ax_heat.set_title(col_title, pad=10, fontproperties=fp_title)
                else:
                    ax_heat.set_title(col_title, fontsize=26, pad=10)

    for row_idx, label in enumerate(row_labels):
        y_center = 0.88 - (row_idx * 0.43) - 0.16
        fp_row = _times_fp(24)
        if fp_row is not None:
            fig.text(0.015, y_center, label,
                    rotation=90, va='center', ha='center',
                    fontproperties=fp_row)
        else:
            fig.text(0.015, y_center, label, fontsize=24,
                    rotation=90, va='center', ha='center')

    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[保存] {os.path.basename(output_path)}  (2x3 heatmap+mean grid)")
    apply_style(PLOT_STYLE)


def visualize_tl_comparison_results(
    comparison_results: Dict[str, dict],
    scenario_config: dict,
    output_dir: str,
    settings: dict,
    device: torch.device,
    all_model_paths: List[str] = None,
    skip_3d: bool = False,
    V_control_override: Optional[np.ndarray] = None
) -> None:
    """
    生成三模型对比的可视化结果

    Args:
        comparison_results: 三模型的结果
        scenario_config: 场景配置
        output_dir: 输出目录
        settings: 全局设置
        device: 计算设备
        all_model_paths: 所有模型路径列表（用于依赖检查）
        skip_3d: 是否跳过3D渲染（当3D渲染在外部单独处理时设为True）
        V_control_override: 可选，统一的对照组数据（与检测指标使用相同来源）
    """
    workspace_dir = settings['global']['workspace_dir']
    preprocess_base = os.path.join(workspace_dir, settings['global']['preprocess_output_base'])

    # 加载测试数据（损伤数据）
    test_data_path = os.path.join(preprocess_base, scenario_config['test_damage_folder'])
    test_npz = os.path.join(test_data_path, 'preprocessed_data_raw.npz')
    if os.path.exists(test_npz):
        V_test = np.load(test_npz)['V'].astype(np.float32)
        print(f"[加载] 测试数据: {V_test.shape}")
    else:
        print(f"[警告] 测试数据不存在: {test_npz}")
        return

    # 使用统一的对照组数据（如果提供），否则回退到验证集
    if V_control_override is not None:
        V_control = V_control_override
        print(f"[加载] 对照组 (统一来源): {V_control.shape}")
    else:
        new_data_path = os.path.join(preprocess_base, scenario_config['new_data_folder'])
        _, V_control = load_data_from_path(
            new_data_path,
            settings['training']['val_samples'],
            "对照组"
        )

    # 模型显示名称
    model_names = {
        'pretrain': 'Pre-train',
        'tl': 'Transfer Learning',
        'fromscratch': 'From Scratch'
    }

    # 收集所有残差数据
    all_residuals = {}  # {model_key: {'damage': residuals, 'control': residuals}}
    column_titles = []

    for model_key in ['pretrain', 'tl', 'fromscratch']:
        result = comparison_results[model_key]
        model = result['model']
        display_name = model_names[model_key]
        column_titles.append(display_name)

        # 计算残差
        residuals_damage = compute_residuals(V_test, model, device)
        residuals_control = compute_residuals(V_control, model, device)
        all_residuals[model_key] = {
            'damage': residuals_damage,
            'control': residuals_control
        }

    # 直接用 matplotlib 生成 2x3 组合图（热力图+均值曲线双段式）
    output_path = os.path.join(output_dir, "comparison_residuals.png")
    _plot_comparison_residuals_grid(all_residuals, column_titles, output_path)

    # 3D渲染（如果skip_3d=True则跳过，由外部单独处理）
    if not skip_3d and settings['detection'].get('render_3d', False) and HAS_PYVISTA:
        render_tl_comparison_3d(
            comparison_results,
            scenario_config,
            output_dir,
            settings,
            device,
            V_test,
            V_control
        )


def render_tl_comparison_3d(
    comparison_results: Dict[str, dict],
    scenario_config: dict,
    output_dir: str,
    settings: dict,
    device: torch.device,
    V_test: np.ndarray,
    V_control: np.ndarray
) -> None:
    """
    生成三模型对比的3D渲染图

    Args:
        comparison_results: 三模型的结果
        scenario_config: 场景配置
        output_dir: 输出目录
        settings: 全局设置
        device: 计算设备
        V_test: 测试数据（损伤数据）
        V_control: 对照组数据
    """
    # 加载渲染资源
    render_assets = load_render_assets(settings)
    if render_assets is None:
        return

    base_mesh = render_assets['base_mesh']
    middlewhole_mesh = render_assets['middlewhole_mesh']
    id_mapping = render_assets['id_mapping']
    measure_ids = render_assets['measure_ids']
    camera_position = render_assets['camera_position']

    # 模型显示名称
    model_names = {
        'pretrain': 'Pre-train',
        'tl': 'Transfer Learning',
        'fromscratch': 'From Scratch'
    }

    damage_3d_images = []
    control_3d_images = []
    column_titles = []

    temp_dir = os.path.join(output_dir, '_temp_3d')
    os.makedirs(temp_dir, exist_ok=True)

    for model_key in ['pretrain', 'tl', 'fromscratch']:
        result = comparison_results[model_key]
        model = result['model']
        display_name = model_names[model_key]
        column_titles.append(display_name)

        # 计算残差
        residuals_damage = compute_residuals(V_test, model, device)
        residuals_control = compute_residuals(V_control, model, device)

        # 计算平均绝对残差作为损伤值
        scores_damage = np.abs(residuals_damage).mean(axis=0)
        scores_control = np.abs(residuals_control).mean(axis=0)

        # 插值到节点 - 损伤
        try:
            node_values_damage = interpolate_to_nodes(
                base_mesh, middlewhole_mesh, scores_damage, measure_ids, id_mapping
            )
            damage_3d_path = os.path.join(temp_dir, f"{model_key}_damage_3d.png")
            render_damage_3d(
                base_mesh, middlewhole_mesh, node_values_damage,
                damage_3d_path, camera_position
            )
            damage_3d_images.append(damage_3d_path)
            print(f"  [3D渲染] {display_name} - Damage 完成")
        except Exception as e:
            print(f"  [警告] 3D渲染失败 ({model_key} damage): {e}")
            damage_3d_images.append(None)

        # 插值到节点 - 对照组
        try:
            node_values_control = interpolate_to_nodes(
                base_mesh, middlewhole_mesh, scores_control, measure_ids, id_mapping
            )
            control_3d_path = os.path.join(temp_dir, f"{model_key}_control_3d.png")
            render_damage_3d(
                base_mesh, middlewhole_mesh, node_values_control,
                control_3d_path, camera_position
            )
            control_3d_images.append(control_3d_path)
            print(f"  [3D渲染] {display_name} - Control 完成")
        except Exception as e:
            print(f"  [警告] 3D渲染失败 ({model_key} control): {e}")
            control_3d_images.append(None)

    # 过滤None值并合并
    valid_damage = [img for img in damage_3d_images if img and os.path.exists(img)]
    valid_control = [img for img in control_3d_images if img and os.path.exists(img)]
    valid_titles = [column_titles[i] for i, img in enumerate(damage_3d_images) if img and os.path.exists(img)]

    if valid_damage and valid_control and len(valid_damage) == len(valid_control):
        output_path = os.path.join(output_dir, "comparison_3d.png")
        merge_images_grid(
            [valid_damage, valid_control],
            output_path,
            valid_titles,
            row_titles=["Test (Damaged)", "Control (Healthy)"]
        )
        print(f"[保存] comparison_3d.png")
