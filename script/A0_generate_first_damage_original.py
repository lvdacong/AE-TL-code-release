"""
A0_generate_first_damage_original.py
======================================
Generate first_damage_original_100 preprocessed data from FEM results.
Extracts 252-channel S-Mises stress from 100 damage simulations.

Usage:
    cd script && python A0_generate_first_damage_original.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

# ========================================
# 配置
# ========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = "C:/SHM_abaqus_data/first_damage"
MEASURES_CSV = os.path.join(SCRIPT_DIR, "AC_convert_and_extract_output", "measures_ID_original.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "AD_preprocess_datasets_output", "first_damage_original_100")
MAX_SAMPLES = 100


def collect_data(source_dir: str, max_folders: int, v_ids: list[int]) -> np.ndarray:
    """
    收集数据（与 AD_preprocess_datasets.py 中的 collect_data 逻辑一致）

    从 source_dir 中按数字排序读取子文件夹，加载 iteration.npy，
    提取 v_ids 对应的 S-Mises 值。

    Returns:
        V_data: 形状为 (n_samples, n_features) 的 numpy 数组
    """
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"数据源目录不存在: {source_dir}")

    # 获取数字命名的子文件夹
    folders = [f for f in os.listdir(source_dir)
               if os.path.isdir(os.path.join(source_dir, f)) and f.isdigit()]
    folders = sorted(folders, key=lambda x: int(x))
    total_folders = len(folders)

    if max_folders is not None:
        folders = folders[:max_folders]

    print(f"  找到 {total_folders} 个文件夹, 处理前 {len(folders)} 个")

    num_points = len(v_ids)
    total_samples = len(folders)
    V_data = np.zeros((total_samples, num_points))
    required_ids_set = set(v_ids)

    sample_idx = 0
    skipped_folders = []

    for folder in tqdm(folders, desc="  收集进度"):
        try:
            folder_path = os.path.join(source_dir, folder)
            file_path = os.path.join(folder_path, "iteration.npy")

            if not os.path.exists(file_path):
                # 回退：寻找目录中唯一的 npy 文件
                npy_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.npy')]
                if len(npy_files) == 1:
                    file_path = os.path.join(folder_path, npy_files[0])
                else:
                    skipped_folders.append(folder)
                    continue

            arr = np.load(file_path)
            cols = (
                ["Element Label", "S-Mises"]
                if arr.shape[1] == 2
                else ["Element Label", "S-Mises", "X", "Y", "Z"][:arr.shape[1]]
            )
            df = pd.DataFrame(arr, columns=cols).set_index("Element Label")

            # 检查是否包含所有必需的元素ID
            actual_labels_set = set(df.index)
            missing_required_ids = required_ids_set - actual_labels_set
            if missing_required_ids:
                raise ValueError(f"缺少必要的ID: {sorted(list(missing_required_ids))}")

            V_data[sample_idx] = df.loc[v_ids]["S-Mises"].values
            sample_idx += 1

        except Exception as e:
            skipped_folders.append(folder)
            continue

    if sample_idx < total_samples:
        print(f"  成功处理 {sample_idx} / {len(folders)} 个文件夹")
        V_data = V_data[:sample_idx]

    if skipped_folders:
        print(f"  跳过 {len(skipped_folders)} 个文件夹: {skipped_folders[:10]}{'...' if len(skipped_folders) > 10 else ''}")

    return V_data


def main():
    print("=" * 60)
    print("生成 first_damage_original_100 预处理数据")
    print("=" * 60)

    # 1. 读取测点ID
    print(f"\n[步骤1] 读取测点ID: {MEASURES_CSV}")
    if not os.path.exists(MEASURES_CSV):
        print(f"错误: 测点ID文件不存在: {MEASURES_CSV}")
        sys.exit(1)

    measures_df = pd.read_csv(MEASURES_CSV)
    v_ids = measures_df["all_measures"].dropna().astype(int).tolist()
    print(f"  测点数量: {len(v_ids)}")

    # 2. 收集数据
    print(f"\n[步骤2] 从 {SOURCE_DIR} 收集数据")
    V_data = collect_data(
        source_dir=SOURCE_DIR,
        max_folders=MAX_SAMPLES,
        v_ids=v_ids
    )
    print(f"  收集完成，数据形状: {V_data.shape}")

    # 3. 保存
    print(f"\n[步骤3] 保存数据")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    npz_path = os.path.join(OUTPUT_DIR, "preprocessed_data_raw.npz")
    np.savez(npz_path, V=V_data)
    print(f"  已保存到: {npz_path}")

    # 4. 验证
    print(f"\n[步骤4] 验证输出")
    loaded = np.load(npz_path)
    print(f"  Keys: {list(loaded.keys())}")
    print(f"  V shape: {loaded['V'].shape}")
    print(f"  V dtype: {loaded['V'].dtype}")
    print(f"  V min: {loaded['V'].min():.6f}")
    print(f"  V max: {loaded['V'].max():.6f}")
    print(f"  V mean: {loaded['V'].mean():.6f}")
    print(f"  V 非零比例: {(loaded['V'] != 0).mean():.4f}")

    # 5. 与已有数据集对比
    print(f"\n[步骤5] 与已有数据集对比")
    ref_datasets = [
        ("health_original_2000", (2000, 252)),
        ("second_damage_original_100", (100, 252)),
        ("first_damage_offset_2_100", (100, 252)),
    ]
    for name, expected_shape in ref_datasets:
        ref_path = os.path.join(SCRIPT_DIR, "AD_preprocess_datasets_output", name, "preprocessed_data_raw.npz")
        if os.path.exists(ref_path):
            ref = np.load(ref_path)
            print(f"  {name}: shape={ref['V'].shape}, mean={ref['V'].mean():.6f}, range=[{ref['V'].min():.6f}, {ref['V'].max():.6f}]")
        else:
            print(f"  {name}: 文件不存在")

    print(f"\n  first_damage_original_100: shape={loaded['V'].shape}, mean={loaded['V'].mean():.6f}, range=[{loaded['V'].min():.6f}, {loaded['V'].max():.6f}]")

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
