# -*- coding: utf-8 -*-
"""Audit whether AE-main scenarios are visually solved by channel-sample heatmaps.

This script does not retrain any model. It reads the already preprocessed
simulation matrices and makes one-coordinate heatmaps: sample/load-case rows
stacked by segment, 252 sensor channels on the x axis, and one shared color
scale per scenario.
"""
from __future__ import annotations

import csv
import html
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "script" / "AD_preprocess_datasets_output"
MODEL_ROOT = ROOT / "script" / "AE_model_train_and_detect_output"
OUT_ROOT = ROOT / "script" / "AF_timespace_heatmap_audit_output"
FIG_DIR = OUT_ROOT / "figures"
HTML_ROOT = OUT_ROOT / "html_report"
HTML_FIG_DIR = HTML_ROOT / "figures"
HTML_INDEX = HTML_ROOT / "index.html"
HTML_REPORT_MD = HTML_ROOT / "report.md"
REPORT_MD = OUT_ROOT / "AE-main时空热图基础方法复审.md"
SUMMARY_CSV = OUT_ROOT / "AE-main时空热图基础方法复审_简单统计结果.csv"


@dataclass(frozen=True)
class Segment:
    label: str
    dataset: str
    role: str


@dataclass(frozen=True)
class Marker:
    channels: tuple[int, ...]
    label: str
    color: str
    linestyle: str = "-"


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    segments: tuple[Segment, ...]
    target_healthy: str
    target_damage: str
    tl_metric_csv: str | None = None
    markers: tuple[Marker, ...] = ()


DAMAGE_REGION_1 = Marker(
    (208,),
    "结构损伤 region_1: ch208 / elem120344",
    "#d62728",
)
REPAIR_REGION = Marker(
    (123,),
    "结构修补 repair_N*: ch123 / elem72048",
    "#ff7f0e",
)
TWO_CIRCLE_DAMAGE = Marker(
    (218, 219),
    "双圆损伤: ch218/219 / elem126920/128846",
    "#9467bd",
    "--",
)
OFFSET_COUNT_1 = Marker(
    (9,),
    "测点偏移 count=1: ch9, 3389->3470",
    "#1f77b4",
)
OFFSET_COUNT_5 = Marker(
    (9, 55, 99, 132, 171),
    "测点偏移 count=5: ch9/55/99/132/171",
    "#1f77b4",
)
DRIFT_HOT_CHANNELS = Marker(
    (13, 16, 37, 167, 193),
    "非均匀漂移高响应通道: ch13/16/37/167/193",
    "#2ca02c",
    ":",
)


SCENARIOS = [
    Scenario(
        key="00_baseline_first_damage",
        title="源域基线：健康 -> 初始损伤",
        segments=(
            Segment("source healthy", "health_original_2000", "source"),
            Segment("first damage", "first_damage_original_100", "damage"),
            Segment("two-circle damage", "two_circle_damage_original_100", "damage_alt"),
        ),
        target_healthy="health_original_2000",
        target_damage="first_damage_original_100",
        tl_metric_csv="baseline_validation/baseline_detection_metrics_max.csv",
        markers=(DAMAGE_REGION_1, TWO_CIRCLE_DAMAGE),
    ),
    Scenario(
        key="01_damage_repaired",
        title="场景1：结构修补域偏移 -> 二次损伤",
        segments=(
            Segment("source healthy", "health_original_2000", "source"),
            Segment("repaired healthy N12", "damage_repaired_12_original_500", "target"),
            Segment("repaired + new damage", "second_damage_12_original_100", "damage"),
        ),
        target_healthy="damage_repaired_12_original_500",
        target_damage="second_damage_12_original_100",
        tl_metric_csv="Damage_Repaired/tl_comparison/detection_metrics_max.csv",
        markers=(REPAIR_REGION, DAMAGE_REGION_1),
    ),
    Scenario(
        key="02_sensor_offset",
        title="场景2：传感器迁移域偏移 -> 损伤",
        segments=(
            Segment("source healthy", "health_original_2000", "source"),
            Segment("offset healthy count=1", "health_offset_count_1_2000", "target"),
            Segment("offset + first damage", "first_damage_offset_count_1_100", "damage"),
        ),
        target_healthy="health_offset_count_1_2000",
        target_damage="first_damage_offset_count_1_100",
        tl_metric_csv="Sensor_Offset/tl_comparison/detection_metrics_max.csv",
        markers=(DAMAGE_REGION_1, OFFSET_COUNT_1),
    ),
    Scenario(
        key="03_sensor_drift",
        title="场景3：非均匀传感器漂移 -> 损伤",
        segments=(
            Segment("source healthy", "health_original_2000", "source"),
            Segment("drifted healthy", "health_drift_nonuniform_2000", "target"),
            Segment("drifted + first damage", "first_damage_drift_nonuniform_100", "damage"),
        ),
        target_healthy="health_drift_nonuniform_2000",
        target_damage="first_damage_drift_nonuniform_100",
        tl_metric_csv="Sensor_Drift/tl_comparison/detection_metrics_max.csv",
        markers=(DAMAGE_REGION_1, DRIFT_HOT_CHANNELS),
    ),
    Scenario(
        key="04_repair_offset_drift",
        title="组合偏移：修补 + 迁移 + 非均匀漂移 -> 二次损伤",
        segments=(
            Segment("source healthy", "health_original_2000", "source"),
            Segment(
                "repaired + offset + drift healthy",
                "damage_repaired_12_offset_count_1_drift_nonuniform_500",
                "target",
            ),
            Segment(
                "repaired + offset + drift + damage",
                "second_damage_12_offset_count_1_drift_nonuniform_100",
                "damage",
            ),
        ),
        target_healthy="damage_repaired_12_offset_count_1_drift_nonuniform_500",
        target_damage="second_damage_12_offset_count_1_drift_nonuniform_100",
        markers=(REPAIR_REGION, DAMAGE_REGION_1, OFFSET_COUNT_1, DRIFT_HOT_CHANNELS),
    ),
]

SWEEPS = [
    Scenario(
        key="10_sweep_repair_size",
        title="难度消融：修补区域大小 N=4/8/12/16/20 的健康域热图",
        segments=(
            Segment("source healthy", "health_original_2000", "source"),
            Segment("repaired N4", "damage_repaired_4_original_500", "target"),
            Segment("repaired N8", "damage_repaired_8_original_500", "target"),
            Segment("repaired N12", "damage_repaired_12_original_500", "target"),
            Segment("repaired N16", "damage_repaired_16_original_500", "target"),
            Segment("repaired N20", "damage_repaired_20_original_500", "target"),
        ),
        target_healthy="damage_repaired_12_original_500",
        target_damage="second_damage_12_original_100",
        markers=(REPAIR_REGION,),
    ),
    Scenario(
        key="11_sweep_offset_count",
        title="难度消融：迁移测点数量 count=0..5 的健康域热图",
        segments=(
            Segment("offset count=0", "health_offset_count_0_2000", "target"),
            Segment("offset count=1", "health_offset_count_1_2000", "target"),
            Segment("offset count=2", "health_offset_count_2_2000", "target"),
            Segment("offset count=3", "health_offset_count_3_2000", "target"),
            Segment("offset count=4", "health_offset_count_4_2000", "target"),
            Segment("offset count=5", "health_offset_count_5_2000", "target"),
        ),
        target_healthy="health_offset_count_1_2000",
        target_damage="first_damage_offset_count_1_100",
        markers=(OFFSET_COUNT_5,),
    ),
    Scenario(
        key="12_sweep_drift_type",
        title="难度消融：健康、全局漂移、非均匀漂移热图",
        segments=(
            Segment("source healthy", "health_original_2000", "source"),
            Segment("drift all healthy", "health_drift_all_2000", "target"),
            Segment("drift nonuniform healthy", "health_drift_nonuniform_2000", "target"),
            Segment("drift nonuniform + damage", "first_damage_drift_nonuniform_100", "damage"),
        ),
        target_healthy="health_drift_nonuniform_2000",
        target_damage="first_damage_drift_nonuniform_100",
        markers=(DAMAGE_REGION_1, DRIFT_HOT_CHANNELS),
    ),
]


def setup_fonts() -> None:
    for path in [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(path)).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 220


def load_v(dataset: str) -> np.ndarray:
    path = DATA_ROOT / dataset / "preprocessed_data_raw.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path)
    return data["V"].astype(np.float32)


def concat_segments(scenario: Scenario) -> tuple[np.ndarray, list[tuple[int, str, str]]]:
    mats = [load_v(seg.dataset) for seg in scenario.segments]
    if len({m.shape[1] for m in mats}) != 1:
        raise ValueError(f"{scenario.key} has inconsistent channel count")
    bounds: list[tuple[int, str, str]] = []
    cursor = 0
    for seg, mat in zip(scenario.segments, mats):
        cursor += mat.shape[0]
        bounds.append((cursor, seg.label, seg.role))
    return np.vstack(mats), bounds


def draw_channel_markers(ax: plt.Axes, markers: tuple[Marker, ...]) -> None:
    """Draw change-location labels outside the heatmap area."""
    if not markers:
        return
    xaxis_transform = ax.get_xaxis_transform()
    levels = [-0.16, -0.25, -0.34, -0.43]
    for idx, marker in enumerate(markers):
        center = float(np.mean(marker.channels))
        text_x = center
        ha = "center"
        if center < 28:
            text_x = center + 7
            ha = "left"
        elif center > 224:
            text_x = center - 7
            ha = "right"
        for channel in marker.channels:
            ax.plot(
                channel,
                -0.012,
                marker="^",
                markersize=5,
                color=marker.color,
                transform=xaxis_transform,
                clip_on=False,
                zorder=4,
            )
        ax.annotate(
            marker.label,
            xy=(center, -0.012),
            xycoords=xaxis_transform,
            xytext=(text_x, levels[idx % len(levels)]),
            textcoords=xaxis_transform,
            ha=ha,
            va="top",
            fontsize=8,
            color=marker.color,
            arrowprops=dict(
                arrowstyle="-|>",
                color=marker.color,
                lw=1.0,
                shrinkA=0,
                shrinkB=1,
                connectionstyle="angle3,angleA=0,angleB=90",
            ),
            annotation_clip=False,
        )


def plot_scenario(scenario: Scenario) -> Path:
    matrix, bounds = concat_segments(scenario)
    finite = matrix[np.isfinite(matrix)]
    vmin = float(np.nanmin(finite)) if finite.size else 0.0
    vmax = float(np.nanmax(finite)) if finite.size else 1.0
    n_rows, n_channels = matrix.shape
    fig_h = min(34.0, max(7.5, n_rows / 520.0))
    fig_w = 15.5

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=False)
    im = ax.imshow(
        np.ma.masked_invalid(matrix),
        aspect="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        extent=[0, n_channels - 1, n_rows, 0],
    )
    y_ticks = []
    y_labels = []
    start = 0
    for end, label, role in bounds:
        if start > 0:
            ax.axhline(start, color="white", lw=1.3)
        y_ticks.append((start + end) / 2)
        y_labels.append(f"{label}\n{end - start} rows")
        start = end
    draw_channel_markers(ax, scenario.markers)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_xlabel("")
    ax.set_ylabel("仿真样本/载荷工况行，按数据段直接拼接")
    ax.set_title(
        f"{scenario.title}\n完整矩阵热图：{n_rows} 行 x {n_channels} 通道；横轴为 0-based 通道，共用 min-max 色标",
        fontsize=14,
        fontweight="bold",
    )
    fig.colorbar(im, ax=ax, shrink=0.90, label=f"V，统一色标 [{vmin:.3g}, {vmax:.3g}]")
    fig.subplots_adjust(left=0.18, right=0.92, top=0.90, bottom=0.42)
    out = FIG_DIR / f"{scenario.key}.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def z_scores(train: np.ndarray, healthy: np.ndarray, damage: np.ndarray) -> dict[str, float]:
    mu = train.mean(axis=0)
    sd = train.std(axis=0) + 1e-9
    z_train = np.abs((train - mu) / sd)
    z_h = np.abs((healthy - mu) / sd)
    z_d = np.abs((damage - mu) / sd)
    y = np.r_[np.zeros(len(healthy), dtype=int), np.ones(len(damage), dtype=int)]
    out: dict[str, float] = {}
    score_specs = {
        "max": (z_train.max(axis=1), z_h.max(axis=1), z_d.max(axis=1)),
        "mean": (z_train.mean(axis=1), z_h.mean(axis=1), z_d.mean(axis=1)),
        "p99": (np.percentile(z_train, 99, axis=1), np.percentile(z_h, 99, axis=1), np.percentile(z_d, 99, axis=1)),
    }
    for name, (s_train, s_h, s_d) in score_specs.items():
        scores = np.r_[s_h, s_d]
        out[f"{name}_auc"] = float(roc_auc_score(y, scores))
        threshold = float(np.quantile(s_train, 0.95))
        out[f"{name}_threshold_q95"] = threshold
        out[f"{name}_far_at_train_q95"] = float(np.mean(s_h > threshold))
        out[f"{name}_tpr_at_train_q95"] = float(np.mean(s_d > threshold))
    return out


def read_best_model_auc(csv_rel: str | None) -> tuple[float | None, float | None, str]:
    if csv_rel is None:
        return None, None, ""
    path = MODEL_ROOT / csv_rel
    if not path.exists():
        return None, None, ""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, None, ""
    if "Model" in rows[0]:
        best = max(rows, key=lambda r: float(r["AUC"]))
        return float(best["AUC"]), float(best.get("TPR@FPR5%", "nan")), best["Model"]
    row = rows[0]
    return float(row["AUC"]), float(row.get("TPR@FPR5%", "nan")), row.get("Model", "Pretrain")


def audit_scenario(scenario: Scenario, fig_path: Path) -> dict[str, object]:
    source = load_v("health_original_2000")
    healthy = load_v(scenario.target_healthy)
    damage = load_v(scenario.target_damage)
    target_scores = z_scores(healthy, healthy, damage)
    source_scores = z_scores(source, healthy, damage)
    model_auc, model_tpr, model_name = read_best_model_auc(scenario.tl_metric_csv)
    visual_level = "不明显"
    if target_scores["max_auc"] >= 0.90:
        visual_level = "强"
    elif target_scores["max_auc"] >= 0.75:
        visual_level = "中等"
    conclusion = "不能只靠热图/简单统计"
    if target_scores["max_auc"] >= 0.95 and target_scores["max_tpr_at_train_q95"] >= 0.80:
        conclusion = "基础统计基本够用"
    elif target_scores["max_auc"] >= 0.85:
        conclusion = "可作初筛但不够稳"
    return {
        "scenario": scenario.key,
        "title": scenario.title,
        "target_healthy": scenario.target_healthy,
        "target_damage": scenario.target_damage,
        "heatmap": str(fig_path.relative_to(OUT_ROOT)),
        "target_z_max_auc": target_scores["max_auc"],
        "target_z_max_tpr_at_q95": target_scores["max_tpr_at_train_q95"],
        "target_z_mean_auc": target_scores["mean_auc"],
        "source_z_max_auc": source_scores["max_auc"],
        "source_z_max_far_at_q95": source_scores["max_far_at_train_q95"],
        "source_z_max_tpr_at_q95": source_scores["max_tpr_at_train_q95"],
        "best_existing_model": model_name,
        "best_existing_model_auc": model_auc,
        "best_existing_model_tpr_at_fpr5": model_tpr,
        "visual_level_by_simple_auc": visual_level,
        "conclusion": conclusion,
    }


def write_csv(rows: list[dict[str, object]]) -> None:
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt_float(value: object) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.3f}"


def write_report(rows: list[dict[str, object]], sweep_figs: list[Path]) -> None:
    lines = [
        "# AE-main 时空热图基础方法复审",
        "",
        "更新时间：2026-05-29",
        "",
        "## 一页结论",
        "",
        "**结论：AE-main 这批模拟研究不能像 ODiSi 那样靠原始时空热图或简单逐通道统计直接解决。** 原始矩阵是 `仿真样本/载荷工况 x 252 通道`，不是实测连续时间；热图能看出源域、修补域、迁移域、漂移域之间的分布变化，但目标域健康与目标域损伤之间的视觉差异很弱。用目标健康段拟合的逐通道 z-score max AUC 只有约 0.66-0.69，远低于已有 AE/TL 模型的 0.95-0.99。",
        "",
        "这说明 `AE-main` 的旧研究和当前 ODiSi 结论不同：ODiSi 的 BI/AI 分布肉眼可分，复杂模型必要性弱；而 AE-main 的船体 FEM 传感器阵问题里，损伤是少数通道上的弱局部偏离，简单热图只能做数据体检，不能替代健康流形/重构模型。",
        "",
        "## 简单统计对比",
        "",
        "| 场景 | 目标健康 vs 损伤 z-max AUC | z-max TPR@q95 | 已有模型最好 AUC | 判断 |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['title']} | {fmt_float(row['target_z_max_auc'])} | "
            f"{fmt_float(row['target_z_max_tpr_at_q95'])} | {fmt_float(row['best_existing_model_auc'])} | {row['conclusion']} |"
        )
    lines.extend(
        [
            "",
            "说明：z-max 是最基础的逐通道健康均值/标准差基线；阈值取训练健康分数 95% 分位。已有模型 AUC 读取自原论文输出的 max-channel detection metrics。",
            "",
            "## 图上定位说明",
            "",
            "- 分段标签已经移到左侧坐标轴外，不再覆盖热图数据。",
            "- 图中变化位置用热图外部的彩色三角和箭头标出，不在数据区内画竖线；通道编号均按热图横轴的 0-based 编号标注。",
            "- 红色：结构损伤 `region_1`，直接测点 `ch208 / elem120344`。",
            "- 橙色：结构修补区 `repair_N*`，直接测点 `ch123 / elem72048`。",
            "- 紫色虚线：双圆损伤的直接测点 `ch218 / elem126920` 与 `ch219 / elem128846`。",
            "- 蓝色：测点偏移位置；`count=1` 为 `ch9`，`count=5` 为 `ch9/55/99/132/171`。",
            "- 绿色点线：非均匀漂移的高响应通道。漂移是传感器/温漂变化，不是结构损伤位置。",
            "",
            "## 完整热图",
            "",
        ]
    )
    for row in rows:
        lines.extend(
            [
                f"### {row['title']}",
                "",
                f"![{row['scenario']}]({row['heatmap'].replace(chr(92), '/')})",
                "",
                f"- 简单 z-max AUC：{fmt_float(row['target_z_max_auc'])}；TPR@q95：{fmt_float(row['target_z_max_tpr_at_q95'])}。",
                f"- 源域健康阈值直接用到目标健康的误报率：{fmt_float(row['source_z_max_far_at_q95'])}。",
                f"- 判断：{row['conclusion']}。",
                "",
            ]
        )
    lines.extend(["## 难度消融热图", ""])
    for fig in sweep_figs:
        rel = fig.relative_to(OUT_ROOT).as_posix()
        lines.extend([f"![{fig.stem}]({rel})", ""])
    lines.extend(
        [
            "## 最终判断",
            "",
            "1. `baseline / first_damage`：完整热图看不出像 ODiSi 那样的强分界；简单 z-max AUC 约 0.685，而原 AE 源域模型 AUC 约 0.985。基础方法不够。",
            "2. `Damage_Repaired`：修补域本身是域偏移，目标健康和二次损伤的逐通道 z-score AUC 约 0.660；已有 TL AUC 约 0.953。基础热图不能替代模型。",
            "3. `Sensor_Offset`：迁移改变测点位置，热图能提示域变化，但目标健康与损伤仍不强分离；z-max AUC 约 0.685，已有 TL AUC 约 0.990。基础方法不够。",
            "4. `Sensor_Drift`：漂移本身在热图中更容易看出来，但损伤检测不是只识别漂移；z-max AUC 仍约 0.685，已有 TL AUC 约 0.993。基础方法不够。",
            "5. `Repair + Offset + Drift`：组合偏移下简单 z-max AUC 约 0.660，没有证据表明热图/简单统计足够。",
            "",
            "因此，`AE-main` 这条旧研究不应被简单归档为“热图就能解决”。它仍然有一个相对清楚的建模必要性：域偏移后需要用少量目标健康数据重建健康流形，损伤信号是局部弱异常，肉眼热图和逐通道 z-score 只能做前置审计，不能作为最终检测方法。",
        ]
    )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def markdown_to_html(md_text: str) -> str:
    try:
        import markdown  # type: ignore

        return markdown.markdown(md_text, extensions=["tables", "fenced_code", "toc"])
    except Exception:
        escaped = html.escape(md_text)
        return f"<pre>{escaped}</pre>"


def write_portable_html_report() -> None:
    HTML_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for fig in FIG_DIR.glob("*.png"):
        shutil.copy2(fig, HTML_FIG_DIR / fig.name)
    md_text = REPORT_MD.read_text(encoding="utf-8")
    HTML_REPORT_MD.write_text(md_text, encoding="utf-8")
    body = markdown_to_html(md_text)
    css = (
        'body{margin:0;background:#f6f7f9;color:#1f2933;'
        'font-family:"Microsoft YaHei","Segoe UI",Arial,sans-serif;line-height:1.72}'
        'main{max-width:1280px;margin:0 auto;padding:36px 32px 72px;background:#fff;'
        'min-height:100vh;box-shadow:0 0 0 1px #e5e7eb}'
        'h1{font-size:30px;line-height:1.25;margin:0 0 18px}'
        'h2{font-size:22px;margin-top:36px;padding-top:12px;border-top:1px solid #e5e7eb}'
        'p,li{font-size:16px}img{display:block;max-width:100%;height:auto;margin:18px auto 28px;'
        'border:1px solid #d9dee7}table{border-collapse:collapse;width:100%;margin:16px 0 24px;font-size:14px}'
        'th,td{border:1px solid #d8dee9;padding:7px 9px;vertical-align:top}'
        'th{background:#f1f5f9;text-align:left}code{background:#f1f5f9;padding:2px 5px;border-radius:4px}'
        '.notice{margin-bottom:24px;padding:12px 14px;background:#eef6ff;border:1px solid #bfdbfe;color:#1e3a5f}'
    )
    HTML_INDEX.write_text(
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>AE-main 时空热图复审</title>"
        f"<style>{css}</style></head><body><main>"
        '<div class="notice">固定可打开版本：HTML 和图片都在 '
        '<code>script/AF_timespace_heatmap_audit_output/html_report/</code> 下。</div>'
        f"{body}</main></body></html>",
        encoding="utf-8",
    )


def main() -> None:
    setup_fonts()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        fig = plot_scenario(scenario)
        rows.append(audit_scenario(scenario, fig))
    sweep_figs = [plot_scenario(scenario) for scenario in SWEEPS]
    write_csv(rows)
    write_report(rows, sweep_figs)
    write_portable_html_report()
    print(f"report={REPORT_MD}")
    print(f"summary={SUMMARY_CSV}")
    print(f"figures={FIG_DIR}")
    print(f"html={HTML_INDEX}")


if __name__ == "__main__":
    main()
