"""
AE_run_so_difficulty_ablation.py
=================================
Difficulty ablation for the Sensor Relocation scenario.
6 offset counts x 4 data sizes = 24 TL training runs.

Usage:
    cd script && python AE_run_so_difficulty_ablation.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ========================================
# 路径设置
# ========================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

AC_OUTPUT = os.path.join(SCRIPT_DIR, "AC_convert_and_extract_output")
AD_OUTPUT = os.path.join(SCRIPT_DIR, "AD_preprocess_datasets_output")
AE_OUTPUT = os.path.join(SCRIPT_DIR, "AE_model_train_and_detect_output")

HEALTH_SOURCE = "C:/SHM_abaqus_data/health"
FIRST_DAMAGE_SOURCE = "C:/SHM_abaqus_data/first_damage"

PRETRAIN_PTH = os.path.join(AE_OUTPUT, "Damage_Repaired", "pretrain", "autoencoder.pth")

DIFFICULTY_VALUES = [0, 1, 2, 3, 4, 5]
DATA_SIZES = [50, 100, 200, 400]

# 训练配置（与已有实验一致）
SETTINGS = {
    "model": {
        "encoder_dims": [768, 384, 192],
        "latent_dim": 192,
        "decoder_dims": [192, 384, 768],
        "dropout": 0.0,
        "activation": "relu",
    },
    "training": {
        "finetune": {
            "epochs": 1000,
            "lr": 1e-3,
            "batch_size": 256,
        },
        "val_samples": 100,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "use_lr_scheduler": True,
        "lr_scheduler_type": "cosine",
        "cosine_eta_min": 1e-6,
        "shuffle": True,
        "optimizer_type": "adam",
        "sgd_momentum": 0.9,
        "noise_std": 0.005,
        "encoder_lr": 1e-5,
    },
}


# ========================================
# 数据收集（复用 A0 逻辑）
# ========================================

def collect_data(source_dir: str, max_folders: int, v_ids: list[int]) -> np.ndarray:
    """
    从仿真结果目录中收集数据

    从 source_dir 中按数字排序读取子文件夹，加载 iteration.npy，
    提取 v_ids 对应的 S-Mises 值。

    Returns:
        V_data: 形状为 (n_samples, n_features) 的 numpy 数组
    """
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"数据源目录不存在: {source_dir}")

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

    for folder in tqdm(folders, desc="  收集进度"):
        try:
            folder_path = os.path.join(source_dir, folder)
            file_path = os.path.join(folder_path, "iteration.npy")

            if not os.path.exists(file_path):
                npy_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.npy')]
                if len(npy_files) == 1:
                    file_path = os.path.join(folder_path, npy_files[0])
                else:
                    continue

            arr = np.load(file_path)
            cols = (
                ["Element Label", "S-Mises"]
                if arr.shape[1] == 2
                else ["Element Label", "S-Mises", "X", "Y", "Z"][:arr.shape[1]]
            )
            df = pd.DataFrame(arr, columns=cols).set_index("Element Label")

            actual_labels_set = set(df.index)
            missing = required_ids_set - actual_labels_set
            if missing:
                raise ValueError(f"缺少必要的ID: {sorted(list(missing))}")

            V_data[sample_idx] = df.loc[v_ids]["S-Mises"].values
            sample_idx += 1

        except Exception:
            continue

    if sample_idx < total_samples:
        print(f"  成功处理 {sample_idx} / {total_samples} 个文件夹")
        V_data = V_data[:sample_idx]

    return V_data


# ========================================
# 数据预处理
# ========================================

def preprocess_offset_data(offset_count: int) -> tuple[str, str]:
    """
    对指定 offset_count 生成 health 和 first_damage 的预处理数据

    返回:
        (health_data_dir, damage_data_dir): 两个 npz 文件所在目录
    """
    health_dir_name = f"health_offset_count_{offset_count}_2000"
    damage_dir_name = f"first_damage_offset_count_{offset_count}_100"
    health_dir = os.path.join(AD_OUTPUT, health_dir_name)
    damage_dir = os.path.join(AD_OUTPUT, damage_dir_name)

    # 读取测点ID
    measures_csv = os.path.join(AC_OUTPUT, f"measures_ID_offset_count_{offset_count}.csv")
    if not os.path.exists(measures_csv):
        raise FileNotFoundError(f"测点ID文件不存在: {measures_csv}")

    measures_df = pd.read_csv(measures_csv)
    col_name = measures_df.columns[0]  # "offset_measures"
    v_ids = measures_df[col_name].dropna().astype(int).tolist()
    assert len(v_ids) == 252, f"offset_count={offset_count} 测点数不等于252: {len(v_ids)}"

    # 生成 health 数据
    health_npz = os.path.join(health_dir, "preprocessed_data_raw.npz")
    if os.path.exists(health_npz):
        d = np.load(health_npz)
        print(f"  [跳过] {health_dir_name} 已存在, shape={d['V'].shape}")
    else:
        print(f"  [生成] {health_dir_name} ...")
        V_health = collect_data(HEALTH_SOURCE, max_folders=2000, v_ids=v_ids)
        os.makedirs(health_dir, exist_ok=True)
        np.savez(health_npz, V=V_health)
        print(f"  [完成] shape={V_health.shape}")

    # 生成 first_damage 数据
    damage_npz = os.path.join(damage_dir, "preprocessed_data_raw.npz")
    if os.path.exists(damage_npz):
        d = np.load(damage_npz)
        print(f"  [跳过] {damage_dir_name} 已存在, shape={d['V'].shape}")
    else:
        print(f"  [生成] {damage_dir_name} ...")
        V_damage = collect_data(FIRST_DAMAGE_SOURCE, max_folders=100, v_ids=v_ids)
        os.makedirs(damage_dir, exist_ok=True)
        np.savez(damage_npz, V=V_damage)
        print(f"  [完成] shape={V_damage.shape}")

    return health_dir, damage_dir


# ========================================
# 主函数
# ========================================

def main():
    print("=" * 60)
    print("Sensor_Offset Difficulty Ablation")
    print(f"  difficulty (offset_count): {DIFFICULTY_VALUES}")
    print(f"  data_sizes: {DATA_SIZES}")
    print(f"  总训练组数: {len(DIFFICULTY_VALUES) * len(DATA_SIZES)}")
    print("=" * 60)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[设备] {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # 验证 pretrain 模型
    if not os.path.exists(PRETRAIN_PTH):
        print(f"[错误] 预训练模型不存在: {PRETRAIN_PTH}")
        sys.exit(1)
    print(f"[预训练模型] {PRETRAIN_PTH}")

    # ============================================
    # 步骤1: 数据预处理
    # ============================================
    print(f"\n{'=' * 60}")
    print("[步骤1] 数据预处理")
    print(f"{'=' * 60}")

    data_dirs: dict[int, tuple[str, str]] = {}
    for oc in DIFFICULTY_VALUES:
        print(f"\n[offset_count={oc}]")
        health_dir, damage_dir = preprocess_offset_data(oc)
        data_dirs[oc] = (health_dir, damage_dir)

    # ============================================
    # 步骤2: 构建 data_loader_func 并调用统一消融引擎
    # ============================================
    print(f"\n{'=' * 60}")
    print("[步骤2] 运行消融实验")
    print(f"{'=' * 60}")

    from AE_difficulty_ablation_auxiliary import run_difficulty_ablation

    # 预加载所有数据到内存，避免重复IO
    all_health_data: dict[int, np.ndarray] = {}
    all_damage_data: dict[int, np.ndarray] = {}

    val_samples = SETTINGS["training"]["val_samples"]

    for oc in DIFFICULTY_VALUES:
        health_dir, damage_dir = data_dirs[oc]
        h_npz = os.path.join(health_dir, "preprocessed_data_raw.npz")
        d_npz = os.path.join(damage_dir, "preprocessed_data_raw.npz")
        all_health_data[oc] = np.load(h_npz)["V"].astype(np.float32)
        all_damage_data[oc] = np.load(d_npz)["V"].astype(np.float32)
        print(f"  [加载] offset_count={oc}: health={all_health_data[oc].shape}, damage={all_damage_data[oc].shape}")

    def data_loader_func(difficulty: int, data_size: int):
        """
        返回 (V_train, V_val) 用于消融引擎

        V_train: 从 health 数据的前 N-val_samples 个中取前 data_size 个
        V_val:   health 数据的后 val_samples 个
        """
        V_health = all_health_data[difficulty]
        N = V_health.shape[0]

        # 固定验证集为最后 val_samples 个样本
        V_train_pool = V_health[:N - val_samples]
        V_val = V_health[N - val_samples:]

        # 截取训练集到 data_size
        if data_size > V_train_pool.shape[0]:
            print(f"    [警告] 请求 {data_size} 但只有 {V_train_pool.shape[0]} 个训练样本")
            V_train = V_train_pool
        else:
            V_train = V_train_pool[:data_size]

        return V_train, V_val

    output_dir = os.path.join(AE_OUTPUT, "Sensor_Offset", "difficulty_ablation")

    loss_matrix, before_tl_values = run_difficulty_ablation(
        scenario_name="Sensor_Offset",
        difficulty_values=DIFFICULTY_VALUES,
        data_sizes=DATA_SIZES,
        pretrain_model_path=PRETRAIN_PTH,
        data_loader_func=data_loader_func,
        output_dir=output_dir,
        settings=SETTINGS,
        device=device,
        xlabel="Number of Offset Sensors",
        freeze_strategy="none",
    )

    # ============================================
    # 步骤3: 汇总打印
    # ============================================
    print(f"\n{'=' * 60}")
    print("[步骤3] 结果汇总")
    print(f"{'=' * 60}")

    print(f"\nBefore TL (pretrain直接推理):")
    for j, diff in enumerate(DIFFICULTY_VALUES):
        print(f"  offset_count={diff}: MSE = {before_tl_values[j]:.6f}")

    print(f"\nTL val_loss 矩阵 (行=data_size, 列=offset_count):")
    header = "data_size\t" + "\t".join([f"oc={d}" for d in DIFFICULTY_VALUES])
    print(header)
    for i, ds in enumerate(DATA_SIZES):
        row = f"{ds}\t\t" + "\t".join([f"{loss_matrix[i, j]:.6f}" for j in range(len(DIFFICULTY_VALUES))])
        print(row)

    print(f"\n[完成] 输出目录: {output_dir}")
    print(f"  CSV: difficulty_ablation_Sensor_Offset.csv")
    print(f"  图:  difficulty_surface_Sensor_Offset.png")


if __name__ == "__main__":
    main()
