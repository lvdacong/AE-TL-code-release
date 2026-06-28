"""
AE_run_dr_difficulty_ablation.py
=================================
Difficulty ablation for the Structural Repair scenario.
6 repair extents x 4 data sizes = 24 TL training runs.

Usage:
    cd script && python AE_run_dr_difficulty_ablation.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch

# ========================================
# 路径设置
# ========================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

AC_OUTPUT = os.path.join(SCRIPT_DIR, "AC_convert_and_extract_output")
AD_OUTPUT = os.path.join(SCRIPT_DIR, "AD_preprocess_datasets_output")
AE_OUTPUT = os.path.join(SCRIPT_DIR, "AE_model_train_and_detect_output")

# 仿真数据源目录
HEALTH_SOURCE = "C:/SHM_abaqus_data/health"
DR_SOURCE_TEMPLATE = "C:/SHM_abaqus_data/damage_repaired_{n}"
SD_SOURCE_TEMPLATE = "C:/SHM_abaqus_data/second_damage_{n}"

PRETRAIN_PTH = os.path.join(AE_OUTPUT, "Damage_Repaired", "pretrain", "autoencoder.pth")

DIFFICULTY_VALUES = [0, 4, 8, 12, 16, 20]
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
        "l2_sp_alpha": 0.005,
    },
}


# ========================================
# 数据收集（复用 SO 逻辑）
# ========================================

from AE_run_so_difficulty_ablation import collect_data


# ========================================
# 数据预处理
# ========================================

def preprocess_dr_data(n_repair: int) -> str:
    """
    对指定 n_repair 生成 damage_repaired 的预处理数据

    - n_repair=0: 使用 health 数据（无域偏移对照）
    - n_repair=N: 使用 damage_repaired_N 数据

    返回:
        data_dir: npz 文件所在目录
    """
    # 读取原始测点ID
    measures_csv = os.path.join(AC_OUTPUT, "measures_ID_original.csv")
    if not os.path.exists(measures_csv):
        raise FileNotFoundError(f"测点ID文件不存在: {measures_csv}")

    measures_df = pd.read_csv(measures_csv)
    col_name = measures_df.columns[0]
    v_ids = measures_df[col_name].dropna().astype(int).tolist()
    assert len(v_ids) == 252, f"测点数不等于252: {len(v_ids)}"

    if n_repair == 0:
        # difficulty=0: 使用 health 数据
        dir_name = "health_original_2000"
        source_dir = HEALTH_SOURCE
        max_folders = 2000
    else:
        # difficulty=N: 使用 damage_repaired_N 数据
        dir_name = f"damage_repaired_{n_repair}_original_500"
        source_dir = DR_SOURCE_TEMPLATE.format(n=n_repair)
        max_folders = 500

    data_dir = os.path.join(AD_OUTPUT, dir_name)
    npz_path = os.path.join(data_dir, "preprocessed_data_raw.npz")

    if os.path.exists(npz_path):
        d = np.load(npz_path)
        print(f"  [skip] {dir_name} already exists, shape={d['V'].shape}")
    else:
        print(f"  [generate] {dir_name} ...")
        if not os.path.isdir(source_dir):
            raise FileNotFoundError(f"Simulation data source not found: {source_dir}")
        V_data = collect_data(source_dir, max_folders=max_folders, v_ids=v_ids)
        os.makedirs(data_dir, exist_ok=True)
        np.savez(npz_path, V=V_data)
        print(f"  [done] shape={V_data.shape}")

    return data_dir


# ========================================
# 主函数
# ========================================

def main():
    print("=" * 60)
    print("Damage_Repaired Difficulty Ablation")
    print(f"  difficulty (n_repair): {DIFFICULTY_VALUES}")
    print(f"  data_sizes: {DATA_SIZES}")
    print(f"  total training runs: {len(DIFFICULTY_VALUES) * len(DATA_SIZES)}")
    print("=" * 60)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[device] {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # 验证 pretrain 模型
    if not os.path.exists(PRETRAIN_PTH):
        print(f"[error] Pretrain model not found: {PRETRAIN_PTH}")
        sys.exit(1)
    print(f"[pretrain] {PRETRAIN_PTH}")

    # ============================================
    # Step 1: Data preprocessing
    # ============================================
    print(f"\n{'=' * 60}")
    print("[Step 1] Data preprocessing")
    print(f"{'=' * 60}")

    data_dirs: dict[int, str] = {}
    for n_repair in DIFFICULTY_VALUES:
        print(f"\n[n_repair={n_repair}]")
        data_dir = preprocess_dr_data(n_repair)
        data_dirs[n_repair] = data_dir

    # ============================================
    # Step 2: Build data_loader_func and run ablation engine
    # ============================================
    print(f"\n{'=' * 60}")
    print("[Step 2] Running ablation experiments")
    print(f"{'=' * 60}")

    from AE_difficulty_ablation_auxiliary import run_difficulty_ablation

    # Preload all data into memory
    all_data: dict[int, np.ndarray] = {}

    val_samples = SETTINGS["training"]["val_samples"]

    for n_repair in DIFFICULTY_VALUES:
        data_dir = data_dirs[n_repair]
        npz_path = os.path.join(data_dir, "preprocessed_data_raw.npz")
        all_data[n_repair] = np.load(npz_path)["V"].astype(np.float32)
        print(f"  [load] n_repair={n_repair}: shape={all_data[n_repair].shape}")

    def data_loader_func(difficulty: int, data_size: int):
        """
        Return (V_train, V_val) for the ablation engine.

        V_train: first data_size samples from the training pool
        V_val:   last val_samples samples
        """
        V_all = all_data[difficulty]
        N = V_all.shape[0]

        # Fixed validation set = last val_samples samples
        V_train_pool = V_all[:N - val_samples]
        V_val = V_all[N - val_samples:]

        # Truncate training set to data_size
        if data_size > V_train_pool.shape[0]:
            print(f"    [warning] requested {data_size} but only {V_train_pool.shape[0]} training samples available")
            V_train = V_train_pool
        else:
            V_train = V_train_pool[:data_size]

        return V_train, V_val

    output_dir = os.path.join(AE_OUTPUT, "Damage_Repaired", "difficulty_ablation")

    loss_matrix, before_tl_values = run_difficulty_ablation(
        scenario_name="Damage_Repaired",
        difficulty_values=DIFFICULTY_VALUES,
        data_sizes=DATA_SIZES,
        pretrain_model_path=PRETRAIN_PTH,
        data_loader_func=data_loader_func,
        output_dir=output_dir,
        settings=SETTINGS,
        device=device,
        xlabel="Number of Repaired Elements",
        freeze_strategy="none",
    )

    # ============================================
    # Step 3: Summary
    # ============================================
    print(f"\n{'=' * 60}")
    print("[Step 3] Results summary")
    print(f"{'=' * 60}")

    print(f"\nBefore TL (pretrain direct inference):")
    for j, diff in enumerate(DIFFICULTY_VALUES):
        print(f"  n_repair={diff}: MSE = {before_tl_values[j]:.6f}")

    print(f"\nTL val_loss matrix (rows=data_size, cols=n_repair):")
    header = "data_size\t" + "\t".join([f"nr={d}" for d in DIFFICULTY_VALUES])
    print(header)
    for i, ds in enumerate(DATA_SIZES):
        row = f"{ds}\t\t" + "\t".join([f"{loss_matrix[i, j]:.6f}" for j in range(len(DIFFICULTY_VALUES))])
        print(row)

    print(f"\n[done] Output directory: {output_dir}")
    print(f"  CSV: difficulty_ablation_Damage_Repaired.csv")
    print(f"  Fig: difficulty_surface_Damage_Repaired.png")


if __name__ == "__main__":
    main()
