"""
AE_freeze_ablation_auxiliary.py
================================
Freeze strategy ablation: compares none/bottom/encoder freeze during
fine-tuning. Generates training curves, heatmap, and 3D renders.
Called by AE_model_train_and_detect.py; not executed directly.
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
    train_model, plot_freeze_strategy_comparison,
    FIG_DPI, PLOT_STYLE, apply_style
)
from AE_model_train_and_detect_auxiliary import (
    compute_residuals, compute_normal_params,
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


# 冻结策略显示名称映射
STRATEGY_DISPLAY_NAMES = {
    'none': 'Full Fine-tuning',
    'bottom': 'Freeze Bottom',
    'encoder': 'Freeze Encoder'
}


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


def run_freeze_ablation(
    scenario_name: str,
    scenario_config: dict,
    pretrain_pth: str,
    settings: dict,
    device: torch.device
) -> Dict[str, dict]:
    """
    运行冻结策略消融实验

    Args:
        scenario_name: 场景名称（如 'damage_repair'）
        scenario_config: 场景配置
        pretrain_pth: 预训练模型路径
        settings: 全局设置
        device: 计算设备

    Returns:
        策略结果字典
    """
    print(f"\n{'='*60}")
    print(f"[{scenario_name}] 冻结策略消融实验")
    print(f"{'='*60}")

    freeze_config = scenario_config['freeze_ablation']
    strategies = freeze_config['strategies']
    train_samples = freeze_config['train_samples']

    # 获取路径
    workspace_dir = settings['global']['workspace_dir']
    output_base = os.path.join(workspace_dir, settings['global']['output_dir'])
    preprocess_base = os.path.join(workspace_dir, settings['global']['preprocess_output_base'])

    # 创建输出目录
    output_dir = os.path.join(output_base, scenario_name, 'freeze_ablation')
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
    print(f"[数据] 使用 {train_samples} 个训练样本")

    # 构建AE配置和训练配置
    ae_config = settings['model']
    training_config = settings['training']

    # 存储结果
    strategy_results = {}
    strategy_curves = {}

    # 对每种冻结策略训练模型
    for strategy in strategies:
        print(f"\n[策略] {STRATEGY_DISPLAY_NAMES.get(strategy, strategy)}")

        strategy_output_dir = os.path.join(output_dir, f"freeze_{strategy}")
        os.makedirs(strategy_output_dir, exist_ok=True)

        # 训练新模型（会生成 pth, csv, png）
        model, train_losses, val_losses = train_with_freeze_strategy(
            V_train_limited, V_val, pretrain_pth, strategy,
            strategy_output_dir, ae_config, training_config, device
        )

        strategy_results[strategy] = {
            'pth': os.path.join(strategy_output_dir, "autoencoder.pth"),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'model': model
        }

        display_name = STRATEGY_DISPLAY_NAMES.get(strategy, strategy)
        strategy_curves[display_name] = (train_losses, val_losses)

    # 生成可视化

    # 1. 训练曲线对比
    print(f"\n[可视化] 生成训练曲线对比图...")
    plot_freeze_strategy_comparison(strategy_curves, output_dir)

    # 2. 残差分析和3D渲染
    visualize_freeze_ablation_results(
        strategy_results,
        scenario_config,
        output_dir,
        settings,
        device
    )

    print(f"\n[完成] {output_dir}")
    return strategy_results


def train_with_freeze_strategy(
    V_train: np.ndarray,
    V_val: np.ndarray,
    pretrain_pth: str,
    freeze_strategy: str,
    output_dir: str,
    ae_config: dict,
    training_config: dict,
    device: torch.device
) -> Tuple[nn.Module, List[float], List[float]]:
    """
    使用指定冻结策略训练模型
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
        model_name=f"TL-{freeze_strategy}",
        train_shuffle=training_config.get('shuffle', True)
    )

    return model, train_losses, val_losses


def visualize_freeze_ablation_results(
    strategy_results: Dict[str, dict],
    scenario_config: dict,
    output_dir: str,
    settings: dict,
    device: torch.device
) -> None:
    """
    生成冻结策略消融实验的可视化结果

    Args:
        strategy_results: 策略结果字典
        scenario_config: 场景配置
        output_dir: 输出目录
        settings: 全局设置
        device: 计算设备
    """
    render_3d_enabled = settings['detection'].get('render_3d', False) and HAS_PYVISTA

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

    # 加载新数据的验证集作为对照组
    new_data_path = os.path.join(preprocess_base, scenario_config['new_data_folder'])
    _, V_control = load_data_from_path(
        new_data_path,
        settings['training']['val_samples'],
        "对照组"
    )

    # 生成热力图
    damage_images = []
    control_images = []
    column_titles = []

    temp_dir = os.path.join(output_dir, '_temp_vis')
    os.makedirs(temp_dir, exist_ok=True)

    for strategy, result in strategy_results.items():
        model = result['model']
        display_name = STRATEGY_DISPLAY_NAMES.get(strategy, strategy)
        column_titles.append(display_name)

        # 计算残差
        residuals_damage = compute_residuals(V_test, model, device)
        residuals_control = compute_residuals(V_control, model, device)

        # 计算正态参数（用于可视化）
        normal_params_damage = compute_normal_params(residuals_damage)
        normal_params_control = compute_normal_params(residuals_control)

        # 生成残差分析图 - 损伤数据
        damage_dir = os.path.join(temp_dir, f"{strategy}_damage")
        os.makedirs(damage_dir, exist_ok=True)
        visualize_combined_residual_analysis(
            residuals_damage,
            damage_dir,
            normal_params_damage,
            name="residuals_analysis"
        )
        damage_images.append(os.path.join(damage_dir, "residuals_analysis.png"))

        # 生成残差分析图 - 对照组
        control_dir = os.path.join(temp_dir, f"{strategy}_control")
        os.makedirs(control_dir, exist_ok=True)
        visualize_combined_residual_analysis(
            residuals_control,
            control_dir,
            normal_params_control,
            name="residuals_analysis"
        )
        control_images.append(os.path.join(control_dir, "residuals_analysis.png"))

    # 合并为网格图
    if damage_images and control_images:
        output_path = os.path.join(output_dir, "freeze_ablation_heatmap.png")
        merge_images_grid(
            [damage_images, control_images],
            output_path,
            column_titles
        )
        print(f"  [保存] {output_path}")

    # 3D渲染
    if render_3d_enabled:
        render_freeze_ablation_3d(
            strategy_results,
            scenario_config,
            output_dir,
            settings,
            device,
            V_test,
            V_control
        )


def render_freeze_ablation_3d(
    strategy_results: Dict[str, dict],
    scenario_config: dict,
    output_dir: str,
    settings: dict,
    device: torch.device,
    V_test: np.ndarray,
    V_control: np.ndarray
) -> None:
    """
    生成冻结策略消融的3D渲染对比图
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

    damage_3d_images = []
    control_3d_images = []
    column_titles = []

    temp_dir = os.path.join(output_dir, '_temp_3d')
    os.makedirs(temp_dir, exist_ok=True)

    for strategy, result in strategy_results.items():
        model = result['model']
        display_name = STRATEGY_DISPLAY_NAMES.get(strategy, strategy)
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
            damage_3d_path = os.path.join(temp_dir, f"{strategy}_damage_3d.png")
            render_damage_3d(
                base_mesh, middlewhole_mesh, node_values_damage,
                damage_3d_path, camera_position
            )
            damage_3d_images.append(damage_3d_path)
            print(f"  [3D渲染] {display_name} - Damage 完成")
        except Exception as e:
            print(f"  [警告] 3D渲染失败 ({strategy} damage): {e}")
            damage_3d_images.append(None)

        # 插值到节点 - 对照组
        try:
            node_values_control = interpolate_to_nodes(
                base_mesh, middlewhole_mesh, scores_control, measure_ids, id_mapping
            )
            control_3d_path = os.path.join(temp_dir, f"{strategy}_control_3d.png")
            render_damage_3d(
                base_mesh, middlewhole_mesh, node_values_control,
                control_3d_path, camera_position
            )
            control_3d_images.append(control_3d_path)
            print(f"  [3D渲染] {display_name} - Control 完成")
        except Exception as e:
            print(f"  [警告] 3D渲染失败 ({strategy} control): {e}")
            control_3d_images.append(None)

    # 过滤None值并合并
    valid_damage = [img for img in damage_3d_images if img and os.path.exists(img)]
    valid_control = [img for img in control_3d_images if img and os.path.exists(img)]
    valid_titles = [column_titles[i] for i, img in enumerate(damage_3d_images) if img and os.path.exists(img)]

    if valid_damage and valid_control and len(valid_damage) == len(valid_control):
        output_path = os.path.join(output_dir, "freeze_ablation_3d.png")
        merge_images_grid(
            [valid_damage, valid_control],
            output_path,
            valid_titles
        )
        print(f"[保存] freeze_ablation_3d.png")
