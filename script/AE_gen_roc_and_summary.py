"""
AE_gen_roc_and_summary.py
==========================
Regenerate ROC curves and summary bar chart from saved model weights.
Outputs roc_comparison.png per scenario and summary_bar_chart.png.

Usage:
    cd script && python AE_gen_roc_and_summary.py
"""

from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from AE_train_model_auxiliary import (
    Autoencoder, load_data_from_path, apply_style, FIG_DPI, PLOT_STYLE
)
from AE_model_train_and_detect_auxiliary import (
    compute_anomaly_scores, compute_detection_metrics
)

# -- 路径 --
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_OUT    = os.path.join(SCRIPT_DIR, 'AE_model_train_and_detect_output')
BASE_PRE    = os.path.join(SCRIPT_DIR, 'AD_preprocess_datasets_output')
PRETRAIN_PTH = os.path.join(BASE_OUT, 'Damage_Repaired', 'pretrain', 'autoencoder.pth')

# -- 模型架构 --
AE_CONFIG = dict(encoder_dims=[768, 384, 192], latent_dim=192,
                 decoder_dims=[192, 384, 768], dropout=0.0, activation='relu')
VAL_SAMPLES = 100   # must match tl_comparison SETTINGS["training"]["val_samples"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -- 色彩方案 (Morandi palette, unified across ROC + bar charts) --
ROC_COLORS = {'TL': '#7BA7BC', 'OldModel': '#C97A6C', 'FromScratch': '#999999'}
ROC_LS     = {'TL': '-',       'OldModel': '--',       'FromScratch': ':'}
BAR_COLORS = {'TL': '#7BA7BC', 'OldModel': '#C97A6C', 'FromScratch': '#AAAAAA'}
DISPLAY_LABELS = {
    'TL': 'TL',
    'OldModel': 'Unadapted source',
    'FromScratch': 'Scratch-trained',
}

# -- 场景配置 --
SCENARIOS = [
    dict(name='Damage_Repaired',  label='Structural\nRepair',
         control='damage_repaired_12_original_500',  test='second_damage_12_original_100'),
    dict(name='Sensor_Offset',  label='Sensor\nRelocation',
         control='health_offset_count_1_2000',   test='first_damage_offset_count_1_100'),
    dict(name='Sensor_Drift',   label='Sensor\nDrift',
         control='health_drift_nonuniform_2000',    test='first_damage_drift_nonuniform_100'),
]


# -- 工具函数 --
def load_ae(pth: str, D: int = 768) -> Autoencoder:
    m = Autoencoder(D, AE_CONFIG['encoder_dims'], AE_CONFIG['latent_dim'],
                    AE_CONFIG['decoder_dims'], AE_CONFIG['dropout'],
                    AE_CONFIG['activation']).to(device)
    m.load_state_dict(torch.load(pth, map_location=device))
    m.eval()
    return m


def load_npz_last(folder: str, n: int) -> np.ndarray:
    """读取预处理 npz，取最后 n 行作为 val 集（与 load_data_from_path 一致）"""
    path = os.path.join(BASE_PRE, folder, 'preprocessed_data_raw.npz')
    V = np.load(path)['V'].astype(np.float32)
    return V[-n:]


def load_npz_all(folder: str) -> np.ndarray:
    path = os.path.join(BASE_PRE, folder, 'preprocessed_data_raw.npz')
    return np.load(path)['V'].astype(np.float32)


# -- ROC 生成（同时生成 mean 和 max） --
TRAIN_SAMPLES = 400  # must match tl_comparison config

def regenerate_roc(scenario: dict, output_subdir: str = 'tl_comparison',
                   force_overwrite: bool = False) -> None:
    name     = scenario['name']
    out_dir  = os.path.join(BASE_OUT, name, output_subdir)
    if not os.path.isdir(out_dir):
        print(f"  [skip] {out_dir} does not exist")
        return

    # Unify V_control sample count with V_test so that ROC/AUC/TPR metrics
    # are computed on a comparable number of healthy samples across scenarios.
    # Original logic produced 100 samples for DR but 1400+ for SO/SD, and
    # the DR fallback overlapped with V_val.
    V_full = load_npz_all(scenario['control'])
    V_test = load_npz_all(scenario['test'])
    N = V_full.shape[0]
    n_control = V_test.shape[0]
    pure_start = TRAIN_SAMPLES
    pure_end = N - VAL_SAMPLES
    if pure_end - pure_start >= n_control:
        V_ctrl = V_full[pure_start:pure_start + n_control]
        ctrl_src = f"holdout[{pure_start}:{pure_start + n_control}]"
    else:
        # Data-limited fallback: use V_val (held-out, never seen gradients).
        V_ctrl = V_full[N - VAL_SAMPLES:N - VAL_SAMPLES + n_control]
        ctrl_src = f"V_val[{N - VAL_SAMPLES}:{N - VAL_SAMPLES + n_control}]"
    print(f"  Test: {V_test.shape}, Control: {V_ctrl.shape} ({ctrl_src})")
    D      = V_ctrl.shape[1]

    models = {
        'TL':          load_ae(os.path.join(out_dir, 'TL', 'autoencoder.pth'), D),
        'OldModel':    load_ae(PRETRAIN_PTH, D),
        'FromScratch': load_ae(os.path.join(out_dir, 'fromscratch', 'autoencoder.pth'), D),
    }

    # 收集两种评分方式的 ROC 数据
    all_roc_data = {}  # method -> {mname: (fprs, tprs, auc)}
    for method, csv_filename, score_label in [
        ('mean', 'detection_metrics.csv', 'Mean-Channel MAE'),
        ('max',  'detection_metrics_max.csv', 'Max-Channel MAE'),
    ]:
        roc_data = {}
        rows = []
        for mname, model in models.items():
            sd = compute_anomaly_scores(V_test, model, device, method=method)
            sc = compute_anomaly_scores(V_ctrl, model, device, method=method)
            m  = compute_detection_metrics(sd, sc)
            roc_data[mname] = (m['fprs'], m['tprs'], m['auc'])
            rows.append({
                'Model': mname,
                'AUC': round(m['auc'], 4),
                'AUC_CI_lo': round(m['auc_ci_lo'], 4),
                'AUC_CI_hi': round(m['auc_ci_hi'], 4),
                'TPR@FPR5%': round(m['tpr_at_fpr'], 4),
                'Best_F1': round(m['best_f1'], 4)
            })
            print(f"  {mname} ({method}): AUC={m['auc']:.4f} [{m['auc_ci_lo']:.4f}, {m['auc_ci_hi']:.4f}]")

        all_roc_data[method] = roc_data

        # CSV 写入策略：默认仅在不存在时写入；force_overwrite=True 时覆盖
        csv_path = os.path.join(out_dir, csv_filename)
        if force_overwrite or not os.path.exists(csv_path):
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            print(f"  [保存] {os.path.relpath(csv_path, SCRIPT_DIR)}")

    # ROC 绘图：DR 是确定 mean/max 选择的首发场景，保留双面板对比
    # SO/SD 等后续场景已沿用 DR 选定的 max-channel，只输出 max 单面板
    roc_style = PLOT_STYLE.copy()
    roc_style.update({
        "font.size": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 14,
    })
    apply_style(roc_style)
    model_order = ['TL', 'OldModel', 'FromScratch']

    is_dr_scenario = (name == 'Damage_Repaired')

    if is_dr_scenario:
        # DR：双面板 (a) Mean + (b) Max（首次对比，建立选择依据）
        # figsize 高度 6.8 让图例紧贴图框上沿（原 7.3 留白过多）
        fig, axes = plt.subplots(1, 2, figsize=(14, 6.8), constrained_layout=True)
        panel_labels = [
            ('mean', '(a) Mean-Channel MAE'),
            ('max', '(b) Max-Channel MAE'),
        ]
        handles_for_legend = []
        for ax, (method, panel_title) in zip(axes, panel_labels):
            roc_data = all_roc_data[method]
            for mname in model_order:
                fprs, tprs, auc = roc_data[mname]
                line, = ax.plot(fprs, tprs, label=DISPLAY_LABELS[mname],
                        color=ROC_COLORS[mname], lw=1.5, linestyle=ROC_LS[mname])
                if method == 'mean':
                    handles_for_legend.append(line)
            ax.plot([0, 1], [0, 1], color='#cccccc', lw=0.8, linestyle='--')
            ax.axvline(x=0.05, color='#999999', lw=0.8, linestyle='--')
            ax.set_xlabel("False Positive Rate")
            ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
            ax.set_aspect('equal', adjustable='box')
            ax.tick_params(direction='in')
            ax.grid(False)
            ax.set_title(panel_title, fontsize=20, pad=10)
        axes[0].set_ylabel("True Positive Rate")
        axes[1].set_ylabel("")
        fig.legend(handles_for_legend, [DISPLAY_LABELS[m] for m in model_order],
                   loc="outside upper center",
                   ncol=3, frameon=False, fontsize=14)
        merged_out = os.path.join(out_dir, 'roc_comparison_merged.png')
        fig.savefig(merged_out, dpi=FIG_DPI)
        plt.close(fig)
        print(f"  [保存] {os.path.relpath(merged_out, SCRIPT_DIR)}")
    else:
        # SO/SD：单面板 max-channel only
        # figsize 高度 6.8 让图例紧贴图框上沿（原 7.3 留白过多）
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.8), constrained_layout=True)
        roc_data = all_roc_data['max']
        for mname in model_order:
            fprs, tprs, auc = roc_data[mname]
            ax.plot(fprs, tprs, label=DISPLAY_LABELS[mname],
                    color=ROC_COLORS[mname], lw=1.5, linestyle=ROC_LS[mname])
        ax.plot([0, 1], [0, 1], color='#cccccc', lw=0.8, linestyle='--')
        ax.axvline(x=0.05, color='#999999', lw=0.8, linestyle='--')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(direction='in')
        ax.grid(False)

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels,
                   loc="outside upper center",
                   ncol=3, frameon=False, fontsize=14)
        max_out = os.path.join(out_dir, 'roc_comparison_max.png')
        fig.savefig(max_out, dpi=FIG_DPI)
        plt.close(fig)
        print(f"  [保存] {os.path.relpath(max_out, SCRIPT_DIR)}")

    apply_style(PLOT_STYLE)


# -- 汇总条形图（单行：max-channel only，已在 DR 内确定为最终评分）--
def plot_summary_bar(output_path: str) -> None:
    """
    1x3 条形图：max-channel MAE（已在 DR 场景内确定为最终评分方案）
    3 个子图对应 AUC / TPR@FPR5% / Best F1，
    横轴为三个场景，三组柱分别为 TL / OldModel / FromScratch。
    """
    metrics_cols  = ['AUC', 'TPR@FPR5%', 'Best_F1']
    metric_labels = ['AUC', 'TPR at FPR = 5%', 'Optimal $F_1$']
    model_order   = ['TL', 'OldModel', 'FromScratch']

    # 读取各场景的 max-channel CSV
    all_data = {}  # scenario_name -> df
    for s in SCENARIOS:
        csv = os.path.join(BASE_OUT, s['name'], 'tl_comparison', 'detection_metrics_max.csv')
        all_data[s['name']] = pd.read_csv(csv).set_index('Model')

    x        = np.arange(len(SCENARIOS))
    width    = 0.25
    x_labels = [s['label'] for s in SCENARIOS]

    # 整体字号放大：轴标签 22, 刻度 18, 图例 14（图例 < 刻度）
    summary_style = PLOT_STYLE.copy()
    summary_style.update({
        "font.size": 22,
        "axes.labelsize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 14,
    })
    apply_style(summary_style)
    # figsize 高度 6.2 预留顶部外部图例空间
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.2), constrained_layout=True)

    for col_idx, (col, ylabel) in enumerate(zip(metrics_cols, metric_labels)):
        ax = axes[col_idx]
        for i, model in enumerate(model_order):
            vals = [all_data[s['name']].loc[model, col] for s in SCENARIOS]
            ax.bar(x + (i - 1) * width, vals, width,
                   label=DISPLAY_LABELS[model], color=BAR_COLORS[model],
                   edgecolor='none')

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.tick_params(direction='in', labelsize=16)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # 图例在整图外部上方横排
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="outside upper center",
               ncol=3, frameon=False, fontsize=14)
    apply_style(PLOT_STYLE)

    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[保存] 汇总条形图: {os.path.relpath(output_path, SCRIPT_DIR)}")


# -- 主程序 --
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--subdir', default='tl_comparison',
                        help='Output subdir under each scenario (tl_comparison or tl_comparison_standard)')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing detection_metrics CSVs')
    parser.add_argument('--skip-summary', action='store_true',
                        help='Skip summary bar chart (only regenerate per-scenario ROC and CSVs)')
    args = parser.parse_args()

    print(f"=== 重新生成 ROC 对比图 (subdir={args.subdir}, force={args.force}) ===")
    for s in SCENARIOS:
        print(f"\n--- {s['name']} ---")
        regenerate_roc(s, output_subdir=args.subdir, force_overwrite=args.force)

    if args.skip_summary:
        print("\n[--skip-summary] 跳过汇总条形图")
    elif args.subdir == 'tl_comparison':
        print("\n=== 生成汇总条形图 ===")
        summary_out = os.path.join(BASE_OUT, 'summary_bar_chart.png')
        plot_summary_bar(summary_out)
    else:
        print(f"\n[subdir={args.subdir}] 跳过汇总条形图（仅 tl_comparison 用于论文）")

    print("\n全部完成。")
