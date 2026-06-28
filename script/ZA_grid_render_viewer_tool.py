"""
ZA_grid_render_viewer_tool.py
==============================
Interactive 3D strain-field viewer for single samples.
Reads VTU mesh from AC stage and renders measurement-point or
full-vector data with optional camera position.

Usage:
    cd script && python ZA_grid_render_viewer_tool.py
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# PyVista导入
try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    print("[Warning] PyVista not available. Please install: pip install pyvista")

# Matplotlib导入（用于自定义colormap）
try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[Warning] Matplotlib not available for custom colormaps")


# ========================================
# 🔧 用户配置区 - 直接修改这里的参数
# ========================================

# 1. 数据范围选择
RENDER_MODE = "full"  # "measures" = 仅测点数据, "full" = 全部单元数据

# 2. 数据文件夹类型
DATA_FOLDER_TYPE = "normal"  # "test" = 使用_test后缀文件夹, "normal" = 不带_test后缀

# 3. 来源模型类别
DATA_CATEGORY = "damage_repaired"  # 如: "health", "first_damage", "damage_repaired", "second_damage"、

# 4. 该类别内要展示的样本ID
SAMPLE_ID = range(1, 2)  # 样本编号，单个数字如: 1，列表如: [1, 2, 3]，范围如: range(1, 21) 表示1到20

# 5. 交互模式选项
INTERACTIVE_MODE = 1  # True = 交互窗口, False = 保存PNG

# 6. 渲染数据类型
CELL_TO_POINT_INTERPOLATION = 1  # True = 插值到节点渲染（平滑）, False = 单元数据直接渲染（分块）

# 7. 颜色映射方式选择 - 尝试不同的配色方案找到最佳观感
COLORMAP_MODE = "turbo"  # 可选值见下方 COLORMAP_OPTIONS 字典

# 8. 颜色映射离散化选项 - 类似Abaqus后处理的条带显示
USE_DISCRETE_COLORMAP = True  # True = 离散条带显示（类似Abaqus）, False = 连续渐变显示
DISCRETE_COLORMAP_BANDS = 24   # 离散条带数量，仅当 USE_DISCRETE_COLORMAP = True 时生效
COLORMAP_RANGE = (0.05, 1.0)   # 颜色映射采样范围 (min, max)，范围0-1，可跳过太暗的颜色。例如(0.15, 1.0)跳过最暗的15%

# ========================================
# 颜色映射选项库 - 选择你喜欢的配色方案
# ========================================
COLORMAP_OPTIONS = {
    # 双向发散型（适合有正负值的应变数据）
    "coolwarm": "蓝→白→红，经典科学可视化配色",
    # 连续型（适合单向变化的数据）
    "turbo": "蓝→绿→黄→红，谷歌Turbo配色",
    # 经典型
    "rainbow": "彩虹全谱，色彩丰富",
}

# ========================================
# 高级配置（一般不需要修改）
# ========================================

# 数据根目录配置
DATA_ROOT_NORMAL = r"C:\SHM_abaqus_data"      # 不带_test后缀
DATA_ROOT_TEST = r"C:\SHM_abaqus_data_test"    # 带_test后缀
FULL_DATA_FILENAME = "iteration.npy"           # 数据文件名

# 3D渲染参数
RENDER_WINDOW_WIDTH = 1920
RENDER_WINDOW_HEIGHT = 1080
RENDER_BACKGROUND = "white"
RENDER_OPACITY = 1.0

# 特征边线参数
FEATURE_EDGE_ANGLE = 30
FEATURE_EDGE_COLOR = "black"
FEATURE_EDGE_WIDTH = 1.5

# 默认相机位置
DEFAULT_CAMERA_POSITION = [
    (-118250.9077302128, 99019.47724416424, 130225.53976681917),
    (16817.598535918256, 18266.62363928763, 31742.253660874267),
    (0.4214857390997515, 0.8935556469878645, -0.1546223705407302)
]


# ========================================
# 路径配置（自动计算）
# ========================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR_AC = os.path.join(SCRIPT_DIR, "AC_convert_and_extract_output")
OUTPUT_DIR_AD = os.path.join(SCRIPT_DIR, "AD_preprocess_datasets_output")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "ZA_grid_render_viewer_output")

# 输入文件
VTU_PATH = os.path.join(OUTPUT_DIR_AC, "whole_from_inp.vtu")
MEASURES_PATH = os.path.join(OUTPUT_DIR_AC, "measures_ID_auto.csv")
ID_MAPPING_PATH = os.path.join(OUTPUT_DIR_AC, "abaqus_id_to_vtu_index.csv")
CAMERA_POSITION_FILE = os.path.join(SCRIPT_DIR, "camera_position.json")


# ========================================
# 工具函数
# ========================================

def get_colormap_with_range(cmap_name: str, vmin: float = 0.0, vmax: float = 1.0):
    """创建范围限制的colormap，跳过太暗或太亮的部分

    Args:
        cmap_name: colormap名称
        vmin: 采样起始位置 (0-1)
        vmax: 采样结束位置 (0-1)

    Returns:
        colormap对象或名称
    """
    if not HAS_MATPLOTLIB or (vmin == 0.0 and vmax == 1.0):
        # 如果没有matplotlib或使用完整范围，直接返回名称
        return cmap_name

    try:
        # 获取原始colormap
        original_cmap = plt.get_cmap(cmap_name)

        # 创建采样点
        colors = original_cmap(np.linspace(vmin, vmax, 256))

        # 创建新的colormap
        new_cmap = LinearSegmentedColormap.from_list(
            f'{cmap_name}_range_{vmin}_{vmax}',
            colors
        )

        return new_cmap
    except Exception as e:
        print(f"[Warning] Failed to create range-limited colormap: {e}")
        return cmap_name


def load_id_mapping(mapping_csv_path: str):
    """加载Abaqus ID到VTU索引的映射"""
    if not os.path.exists(mapping_csv_path):
        raise FileNotFoundError(f"ID mapping file not found: {mapping_csv_path}")

    df = pd.read_csv(mapping_csv_path)
    id_mapping = dict(zip(df['abaqus_id'], df['vtu_index']))
    print(f"[Loaded] ID mapping: {len(id_mapping)} entries")
    return id_mapping


def load_vtu_model(vtu_path: str):
    """加载VTU模型文件"""
    if not os.path.exists(vtu_path):
        raise FileNotFoundError(f"VTU file not found: {vtu_path}")

    mesh = pv.read(vtu_path)
    print(f"[Loaded] VTU model: {mesh.n_cells} cells, {mesh.n_points} points")
    return mesh


def load_measure_ids(measures_csv_path: str, group_name: str = "all_measures"):
    """加载测点ID列表"""
    if not os.path.exists(measures_csv_path):
        raise FileNotFoundError(f"Measures ID file not found: {measures_csv_path}")

    df = pd.read_csv(measures_csv_path)
    if group_name not in df.columns:
        raise ValueError(f"Group '{group_name}' not found in {measures_csv_path}")

    measure_ids = df[group_name].dropna().astype(int).values
    print(f"[Loaded] Measure IDs ({group_name}): {len(measure_ids)} points")
    return measure_ids


def load_camera_position(camera_file: str):
    """加载相机位置配置"""
    if not os.path.exists(camera_file):
        print(f"[Info] Camera position file not found, using default position")
        return DEFAULT_CAMERA_POSITION

    import json
    with open(camera_file, 'r') as f:
        camera_data = json.load(f)

    camera_position = [
        tuple(camera_data['camera_position']),
        tuple(camera_data['focal_point']),
        tuple(camera_data['view_up'])
    ]

    print(f"[Loaded] Camera position from: {camera_file}")
    return camera_position


def get_data_root():
    """根据配置获取数据根目录"""
    if DATA_FOLDER_TYPE == "test":
        return DATA_ROOT_TEST
    else:
        return DATA_ROOT_NORMAL


def load_measures_data(category: str, sample_id: int, measure_ids: np.ndarray,
                       total_cells: int, id_mapping: dict):
    """加载测点模式的数据（从npz文件）"""
    npz_path = os.path.join(OUTPUT_DIR_AD, category, "preprocessed_data_raw.npz")

    if not os.path.exists(npz_path):
        print(f"[Warning] NPZ file not found: {npz_path}")
        return np.zeros(total_cells, dtype=np.float32), None

    data = np.load(npz_path)

    if "V" not in data:
        print(f"[Warning] NPZ file does not contain 'V' key: {npz_path}")
        return np.zeros(total_cells, dtype=np.float32), None

    V = data["V"]
    N, D = V.shape

    # 验证索引
    if sample_id < 0 or sample_id >= N:
        print(f"[Warning] Invalid sample index: {sample_id}, valid range: 0 to {N - 1}")
        return np.zeros(total_cells, dtype=np.float32), None

    # 提取样本数据
    sample = V[sample_id]

    # 映射到VTU单元
    vtu_data = np.zeros(total_cells, dtype=np.float32)

    for i, abaqus_id in enumerate(measure_ids):
        if i < len(sample) and abaqus_id in id_mapping:
            vtu_index = id_mapping[abaqus_id]
            if 0 <= vtu_index < total_cells:
                vtu_data[vtu_index] = sample[i]

    return vtu_data, {'category': category, 'sample_id': sample_id}


def load_full_data(category: str, sample_id: int, total_cells: int, id_mapping: dict):
    """加载全向量模式的数据（从.npy文件）"""
    data_root = get_data_root()
    npy_path = os.path.join(data_root, category, str(sample_id), FULL_DATA_FILENAME)

    if not os.path.exists(npy_path):
        print(f"[Warning] NPY file not found: {npy_path}")
        return np.zeros(total_cells, dtype=np.float32), None

    data = np.load(npy_path)

    # 初始化VTU数据数组
    vtu_data = np.zeros(total_cells, dtype=np.float32)

    # 处理不同的数据格式
    if data.ndim == 1:
        # 一维数组：直接使用
        if len(data) == total_cells:
            vtu_data = data.astype(np.float32)
        else:
            min_len = min(len(data), total_cells)
            vtu_data[:min_len] = data[:min_len]

    elif data.ndim == 2 and data.shape[1] == 2:
        # 两列格式：[Element Label, Value]
        element_labels = data[:, 0].astype(int)  # Abaqus ID
        element_values = data[:, 1]              # 应变值

        # 使用映射表
        for i, abaqus_id in enumerate(element_labels):
            if abaqus_id in id_mapping:
                vtu_index = id_mapping[abaqus_id]
                if 0 <= vtu_index < total_cells:
                    vtu_data[vtu_index] = element_values[i]

    elif data.ndim == 2:
        # 其他二维格式
        if data.shape[0] == total_cells:
            vtu_data = data[:, 0].astype(np.float32)
        elif data.shape[1] == total_cells:
            vtu_data = data[0, :].astype(np.float32)
        else:
            flat_data = data.flatten()
            min_len = min(len(flat_data), total_cells)
            vtu_data[:min_len] = flat_data[:min_len]

    return vtu_data, {'category': category, 'sample_id': sample_id}


def save_camera_position(plotter, camera_file: str):
    """保存当前相机位置到JSON文件"""
    import json

    camera_pos = plotter.camera_position
    camera_data = {
        'camera_position': list(camera_pos[0]),
        'focal_point': list(camera_pos[1]),
        'view_up': list(camera_pos[2])
    }

    with open(camera_file, 'w') as f:
        json.dump(camera_data, f, indent=2)

    print(f"[Saved] Camera position to: {camera_file}")


def render_single_sample(base_mesh, vtu_data, category, sample_id, camera_position, output_path):
    """渲染单个样本的3D应变场

    Args:
        base_mesh: VTU网格对象
        vtu_data: VTU单元数据数组
        category: 类别名称
        sample_id: 样本ID
        camera_position: 相机位置
        output_path: 输出PNG文件路径（仅非交互模式）
    """
    print(f"\n[Render] Creating visualization for {category} - Sample {sample_id}...")

    # 计算数据范围
    if vtu_data is not None and not np.all(vtu_data == 0):
        data_min = np.min(vtu_data)
        data_max = np.max(vtu_data)
        print(f"[Render] Data range: [{data_min:.6f}, {data_max:.6f}]")
    else:
        data_min = 0.0
        data_max = 1.0
        print(f"[Render] No data or all zeros")

    # 创建plotter
    plotter = pv.Plotter(
        window_size=(RENDER_WINDOW_WIDTH, RENDER_WINDOW_HEIGHT),
        off_screen=(not INTERACTIVE_MODE)
    )
    plotter.set_background(RENDER_BACKGROUND)

    if vtu_data is None or np.all(vtu_data == 0):
        # 空数据，显示灰色模型
        plotter.add_mesh(base_mesh, color='lightgray', show_edges=True)
        plotter.add_text(
            f"{category} - Sample {sample_id}\n(No data)",
            position='upper_left',
            font_size=14,
            color='red'
        )
    else:
        # 创建网格副本并添加数据
        mesh = base_mesh.copy()
        mesh.cell_data['strain'] = vtu_data

        # 过滤掉零值单元
        # 使用0作为阈值，只保留有实际应变值的单元
        threshold_value = 1e-10  # 非常小的值，基本等于过滤0值
        mesh = mesh.threshold(threshold_value, scalars='strain')
        print(f"[Render] Filtered zero-value cells (threshold: {threshold_value})")
        print(f"[Render] Remaining cells after filtering: {mesh.n_cells}")

        # 根据配置选择渲染模式
        if CELL_TO_POINT_INTERPOLATION:
            # 插值模式：将单元数据插值到节点
            mesh = mesh.cell_data_to_point_data()
            scalar_name = 'strain'
            print(f"[Render] Using point data interpolation (smooth rendering)")
        else:
            # 直接模式：使用单元数据
            scalar_name = 'strain'
            print(f"[Render] Using cell data directly (block rendering)")

        # 添加主网格
        # 根据配置选择连续或离散colormap，应用颜色范围限制
        colormap = get_colormap_with_range(COLORMAP_MODE, COLORMAP_RANGE[0], COLORMAP_RANGE[1])

        if USE_DISCRETE_COLORMAP:
            # 离散条带模式（类似Abaqus）
            plotter.add_mesh(
                mesh,
                scalars=scalar_name,
                cmap=colormap,
                n_colors=DISCRETE_COLORMAP_BANDS,  # 指定离散条带数量
                clim=(data_min, data_max),
                show_edges=False,
                opacity=RENDER_OPACITY,
                show_scalar_bar=True,
                scalar_bar_args={
                    'title': 'Strain',
                    'vertical': True,
                    'position_x': 0.85,
                    'position_y': 0.1,
                    'n_labels': min(DISCRETE_COLORMAP_BANDS + 1, 11)  # 标签数量限制在合理范围
                }
            )
            print(f"[Render] Using discrete colormap with {DISCRETE_COLORMAP_BANDS} bands (range: {COLORMAP_RANGE[0]:.2f}-{COLORMAP_RANGE[1]:.2f})")
        else:
            # 连续渐变模式
            plotter.add_mesh(
                mesh,
                scalars=scalar_name,
                cmap=colormap,
                clim=(data_min, data_max),
                show_edges=False,
                opacity=RENDER_OPACITY,
                show_scalar_bar=True,
                scalar_bar_args={
                    'title': 'Strain',
                    'vertical': True,
                    'position_x': 0.85,
                    'position_y': 0.1
                }
            )
            print(f"[Render] Using continuous colormap (range: {COLORMAP_RANGE[0]:.2f}-{COLORMAP_RANGE[1]:.2f})")

        # 添加特征边线
        feature_edges = base_mesh.extract_feature_edges(
            boundary_edges=True,
            non_manifold_edges=True,
            feature_edges=True,
            manifold_edges=False,
            feature_angle=FEATURE_EDGE_ANGLE
        )
        plotter.add_mesh(
            feature_edges,
            color=FEATURE_EDGE_COLOR,
            line_width=FEATURE_EDGE_WIDTH,
            render_lines_as_tubes=False
        )
        print(f"[Render] Feature edges added")

        # 添加标题
        plotter.add_text(
            f"{category} - Sample {sample_id}",
            position='upper_left',
            font_size=14,
            color='black'
        )

    # 设置相机位置
    plotter.camera_position = camera_position

    # 显示或保存
    if INTERACTIVE_MODE:
        print(f"[Render] Opening interactive window...")
        print("[Render] 提示：鼠标拖动旋转视角，滚轮缩放，按 'q' 关闭窗口")
        plotter.show()
        print(f"[Render] Interactive window closed.")

        # 保存相机位置供下次使用
        save_camera_position(plotter, CAMERA_POSITION_FILE)
        return None
    else:
        print(f"[Render] Saving to: {output_path}")
        plotter.screenshot(output_path)
        plotter.close()
        print(f"[Success] Image saved to: {output_path}")
        return output_path


# ========================================
# 主程序逻辑
# ========================================

def main():
    """主函数"""
    print("\n" + "="*60)
    print("流程90：单样本3D渲染工具")
    print("="*60)

    # 检查PyVista
    if not HAS_PYVISTA:
        print("\n[Error] PyVista is not installed!")
        print("[Error] Please install it: pip install pyvista")
        sys.exit(1)

    # 处理SAMPLE_ID：支持单个值、列表或range对象
    if isinstance(SAMPLE_ID, (list, tuple, range)):
        sample_id_list = list(SAMPLE_ID)
    else:
        sample_id_list = [SAMPLE_ID]

    # 打印配置信息
    print(f"\n[Configuration]")
    print(f"  - Render mode: {RENDER_MODE}")
    print(f"  - Data folder type: {DATA_FOLDER_TYPE}")
    print(f"  - Category: {DATA_CATEGORY}")
    print(f"  - Sample ID(s): {sample_id_list}")
    print(f"  - Interactive mode: {INTERACTIVE_MODE}")
    print(f"  - Interpolation: {'Point (smooth)' if CELL_TO_POINT_INTERPOLATION else 'Cell (block)'}")
    print(f"  - Colormap: {COLORMAP_MODE} ({COLORMAP_OPTIONS.get(COLORMAP_MODE, 'Unknown')})")
    print(f"  - Colormap style: {'Discrete (' + str(DISCRETE_COLORMAP_BANDS) + ' bands)' if USE_DISCRETE_COLORMAP else 'Continuous'}")
    print(f"  - Data root: {get_data_root()}")

    # 验证模式
    if RENDER_MODE not in ["measures", "full"]:
        print(f"\n[Error] Invalid RENDER_MODE: {RENDER_MODE}")
        print(f"[Error] Must be 'measures' or 'full'")
        sys.exit(1)

    # 验证颜色映射
    if COLORMAP_MODE not in COLORMAP_OPTIONS:
        print(f"\n[Warning] Unknown COLORMAP_MODE: {COLORMAP_MODE}")
        print(f"[Warning] Available options:")
        for cmap_name, cmap_desc in COLORMAP_OPTIONS.items():
            print(f"           - {cmap_name}: {cmap_desc}")
        print(f"[Warning] Will try to use '{COLORMAP_MODE}' anyway (might work if it's a valid matplotlib colormap)")
        print()

    # 检查必需文件
    print("\n[Check] Verifying required files...")

    if not os.path.exists(VTU_PATH):
        print(f"[Error] VTU file not found: {VTU_PATH}")
        print(f"[Error] Please run 02_inp_to_vtu_main.py first")
        sys.exit(1)

    if not os.path.exists(ID_MAPPING_PATH):
        print(f"[Error] ID mapping file not found: {ID_MAPPING_PATH}")
        print(f"[Error] Please run 02_inp_to_vtu_main.py first")
        sys.exit(1)

    if RENDER_MODE == "measures" and not os.path.exists(MEASURES_PATH):
        print(f"[Error] Measures file not found: {MEASURES_PATH}")
        print(f"[Error] Please run 02_inp_to_vtu_main.py first")
        sys.exit(1)

    print("[Check] All required files found!")

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 清空之前的PNG文件（仅非交互模式）
    if not INTERACTIVE_MODE:
        print("\n[Cleanup] Clearing previous PNG files...")
        png_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')]
        if png_files:
            for png_file in png_files:
                png_path = os.path.join(OUTPUT_DIR, png_file)
                os.remove(png_path)
                print(f"  - Removed: {png_file}")
            print(f"[Cleanup] Removed {len(png_files)} PNG file(s)")
        else:
            print("[Cleanup] No previous PNG files found")

    # 1. 加载VTU模型和ID映射
    print("\n[Step 1] Loading VTU model and ID mapping...")
    base_mesh = load_vtu_model(VTU_PATH)
    id_mapping = load_id_mapping(ID_MAPPING_PATH)
    total_cells = base_mesh.n_cells

    # 2. 加载测点ID（仅measures模式需要）
    measure_ids = None
    if RENDER_MODE == "measures":
        print("\n[Step 2] Loading measure IDs...")
        measure_ids = load_measure_ids(MEASURES_PATH, group_name="all_measures")

    # 3. 加载相机位置
    print(f"\n[Step 3] Loading camera position...")
    camera_position = load_camera_position(CAMERA_POSITION_FILE)

    # 4. 批量处理样本
    print(f"\n[Step 4] Processing {len(sample_id_list)} sample(s)...")
    saved_files = []

    for idx, sample_id in enumerate(sample_id_list):
        print(f"\n{'='*60}")
        print(f"Processing sample {idx+1}/{len(sample_id_list)}: {DATA_CATEGORY} - Sample {sample_id}")
        print(f"{'='*60}")

        # 加载数据
        print(f"\n[Load] Loading data for {DATA_CATEGORY} - Sample {sample_id}...")

        if RENDER_MODE == "measures":
            vtu_data, info = load_measures_data(
                DATA_CATEGORY,
                sample_id,
                measure_ids,
                total_cells,
                id_mapping
            )
        else:
            vtu_data, info = load_full_data(
                DATA_CATEGORY,
                sample_id,
                total_cells,
                id_mapping
            )

        # 准备输出文件名
        output_filename = f"{DATA_CATEGORY}_{sample_id}_{RENDER_MODE}.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        # 渲染
        print(f"\n[Render] Rendering sample {sample_id}...")
        result_path = render_single_sample(
            base_mesh,
            vtu_data,
            DATA_CATEGORY,
            sample_id,
            camera_position,
            output_path
        )

        if result_path:
            saved_files.append(result_path)

    # 打印总结
    print("\n" + "="*60)
    print("渲染完成！")
    print("="*60)
    print(f"\n[Summary]")
    print(f"  - Render mode: {RENDER_MODE}")
    print(f"  - Data folder type: {DATA_FOLDER_TYPE}")
    print(f"  - Category: {DATA_CATEGORY}")
    print(f"  - Sample ID(s): {sample_id_list}")
    print(f"  - Interactive mode: {INTERACTIVE_MODE}")
    print(f"  - Interpolation: {'Point (smooth)' if CELL_TO_POINT_INTERPOLATION else 'Cell (block)'}")
    print(f"  - Colormap: {COLORMAP_MODE}")
    print(f"  - Colormap style: {'Discrete (' + str(DISCRETE_COLORMAP_BANDS) + ' bands)' if USE_DISCRETE_COLORMAP else 'Continuous'}")
    if not INTERACTIVE_MODE:
        print(f"  - Output files: {len(saved_files)} image(s) saved")
        for saved_file in saved_files:
            print(f"    * {os.path.basename(saved_file)}")
        print(f"\n[Info] Output directory: {OUTPUT_DIR}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
