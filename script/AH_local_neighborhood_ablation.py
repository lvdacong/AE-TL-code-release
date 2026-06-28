# -*- coding: utf-8 -*-
"""Local-neighborhood distribution ablation for AE-main.

This script tests whether a small physical neighborhood around each measured
element is enough, instead of using the full 252-channel joint distribution.
Neighborhood rings are defined by FEM edge-adjacency shortest-path distance
between measured element IDs.
"""
from __future__ import annotations

import csv
import html
import pickle
import shutil
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "script" / "AD_preprocess_datasets_output"
AC_ROOT = ROOT / "script" / "AC_convert_and_extract_output"
OUT_ROOT = ROOT / "script" / "AH_local_neighborhood_ablation_output"
FIG_DIR = OUT_ROOT / "figures"
HTML_ROOT = OUT_ROOT / "html_report"
HTML_FIG_DIR = HTML_ROOT / "figures"
REPORT_MD = OUT_ROOT / "AE-main局部邻域尺度消融.md"
HTML_INDEX = HTML_ROOT / "index.html"
HTML_REPORT_MD = HTML_ROOT / "report.md"
KNOWN_CURVE_CSV = OUT_ROOT / "known_region_neighborhood_ablation.csv"
LOCAL_SCAN_CSV = OUT_ROOT / "all_center_local_neighborhood_scan.csv"
SCAN_SUMMARY_CSV = OUT_ROOT / "local_scan_rank_summary.csv"

MAX_SHELLS = 5
RANDOM_STATE = 7


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    healthy_dataset: str
    damage_dataset: str
    known_channels: tuple[int, ...]
    known_label: str


SCENARIOS = [
    Scenario(
        key="baseline_first_damage",
        title="源域基线：健康 vs 初始损伤",
        healthy_dataset="health_original_2000",
        damage_dataset="first_damage_original_100",
        known_channels=(208,),
        known_label="region_1 损伤，ch208 / elem120344",
    ),
    Scenario(
        key="baseline_two_circle_damage",
        title="源域基线：健康 vs 双圆损伤",
        healthy_dataset="health_original_2000",
        damage_dataset="two_circle_damage_original_100",
        known_channels=(218, 219),
        known_label="双圆损伤，ch218/ch219 / elem126920/128846",
    ),
    Scenario(
        key="damage_repaired_second_damage",
        title="修补域：修补健康 vs 二次损伤",
        healthy_dataset="damage_repaired_12_original_500",
        damage_dataset="second_damage_12_original_100",
        known_channels=(208,),
        known_label="二次损伤 region_1，ch208 / elem120344",
    ),
    Scenario(
        key="sensor_offset_first_damage",
        title="测点偏移域：偏移健康 vs 初始损伤",
        healthy_dataset="health_offset_count_1_2000",
        damage_dataset="first_damage_offset_count_1_100",
        known_channels=(208,),
        known_label="region_1 损伤，ch208 / elem120344",
    ),
    Scenario(
        key="sensor_drift_first_damage",
        title="漂移域：漂移健康 vs 初始损伤",
        healthy_dataset="health_drift_nonuniform_2000",
        damage_dataset="first_damage_drift_nonuniform_100",
        known_channels=(208,),
        known_label="region_1 损伤，ch208 / elem120344",
    ),
    Scenario(
        key="repair_offset_drift_second_damage",
        title="组合偏移域：组合健康 vs 二次损伤",
        healthy_dataset="damage_repaired_12_offset_count_1_drift_nonuniform_500",
        damage_dataset="second_damage_12_offset_count_1_drift_nonuniform_100",
        known_channels=(208,),
        known_label="二次损伤 region_1，ch208 / elem120344",
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
    return np.load(DATA_ROOT / dataset / "preprocessed_data_raw.npz")["V"].astype(np.float32)


def load_measure_ids() -> list[int]:
    with (AC_ROOT / "measures_ID_original.csv").open("r", encoding="utf-8-sig", newline="") as f:
        return [int(row["all_measures"]) for row in csv.DictReader(f)]


def load_adjacency() -> dict[int, set[int]]:
    with (AC_ROOT / "adjacency_graph.pkl").open("rb") as f:
        return pickle.load(f)


def measured_shells(
    seed_channels: tuple[int, ...],
    measure_ids: list[int],
    channel_by_elem: dict[int, int],
    adjacency: dict[int, set[int]],
    max_shells: int = MAX_SHELLS,
) -> list[dict[str, object]]:
    seed_elements = [measure_ids[ch] for ch in seed_channels]
    distance = {elem: 0 for elem in seed_elements}
    queue = deque(seed_elements)
    shells_by_distance: dict[int, list[int]] = defaultdict(list)
    for ch in seed_channels:
        shells_by_distance[0].append(ch)
    while queue and len(shells_by_distance) < max_shells + 1:
        elem = queue.popleft()
        for nb in adjacency.get(elem, set()):
            if nb in distance:
                continue
            distance[nb] = distance[elem] + 1
            if nb in channel_by_elem:
                shells_by_distance[distance[nb]].append(channel_by_elem[nb])
            queue.append(nb)
    cumulative: set[int] = set()
    rows = []
    for ring_index, fem_hop in enumerate(sorted(shells_by_distance)[: max_shells + 1]):
        cumulative.update(shells_by_distance[fem_hop])
        channels = tuple(sorted(cumulative))
        rows.append(
            {
                "ring_index": ring_index,
                "max_fem_hop": fem_hop,
                "n_channels": len(channels),
                "channels": channels,
            }
        )
    return rows


def c2st_auc(healthy: np.ndarray, damage: np.ndarray, channels: tuple[int, ...]) -> tuple[float, float]:
    x = np.vstack([healthy[:, channels], damage[:, channels]])
    y = np.r_[np.zeros(len(healthy), dtype=int), np.ones(len(damage), dtype=int)]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    estimator = make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    )
    proba = cross_val_predict(estimator, x, y, cv=cv, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    return float(roc_auc_score(y, proba)), float(balanced_accuracy_score(y, pred))


def mahalanobis_auc(healthy: np.ndarray, damage: np.ndarray, channels: tuple[int, ...]) -> tuple[float, float]:
    xh = healthy[:, channels]
    xd = damage[:, channels]
    scaler = StandardScaler()
    xh_s = scaler.fit_transform(xh)
    xd_s = scaler.transform(xd)
    model = LedoitWolf().fit(xh_s)
    diff_h = xh_s - model.location_
    diff_d = xd_s - model.location_
    precision = model.precision_
    score_h = np.einsum("ij,jk,ik->i", diff_h, precision, diff_h)
    score_d = np.einsum("ij,jk,ik->i", diff_d, precision, diff_d)
    y = np.r_[np.zeros(len(score_h), dtype=int), np.ones(len(score_d), dtype=int)]
    scores = np.r_[score_h, score_d]
    threshold = float(np.quantile(score_h, 0.95))
    tpr = float(np.mean(score_d > threshold))
    return float(roc_auc_score(y, scores)), tpr


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, digits: int = 3) -> str:
    value = float(value)
    if abs(value) < 0.001 and value != 0:
        return f"{value:.1e}"
    return f"{value:.{digits}f}"


def build_neighborhood_cache(
    measure_ids: list[int],
    adjacency: dict[int, set[int]],
) -> dict[tuple[int, ...], list[dict[str, object]]]:
    channel_by_elem = {elem: ch for ch, elem in enumerate(measure_ids)}
    cache: dict[tuple[int, ...], list[dict[str, object]]] = {}
    for ch in range(len(measure_ids)):
        cache[(ch,)] = measured_shells((ch,), measure_ids, channel_by_elem, adjacency, MAX_SHELLS)
    for scenario in SCENARIOS:
        cache[scenario.known_channels] = measured_shells(
            scenario.known_channels,
            measure_ids,
            channel_by_elem,
            adjacency,
            MAX_SHELLS,
        )
    return cache


def run_known_ablation(cache: dict[tuple[int, ...], list[dict[str, object]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        healthy = load_v(scenario.healthy_dataset)
        damage = load_v(scenario.damage_dataset)
        for spec in cache[scenario.known_channels]:
            channels = spec["channels"]
            c_auc, c_bacc = c2st_auc(healthy, damage, channels)
            m_auc, m_tpr = mahalanobis_auc(healthy, damage, channels)
            rows.append(
                {
                    "scenario": scenario.key,
                    "title": scenario.title,
                    "known_label": scenario.known_label,
                    "seed_channels": "/".join(str(c) for c in scenario.known_channels),
                    "ring_index": spec["ring_index"],
                    "max_fem_hop": spec["max_fem_hop"],
                    "n_channels": spec["n_channels"],
                    "channels": "/".join(str(c) for c in channels),
                    "local_c2st_auc": c_auc,
                    "local_c2st_balanced_accuracy": c_bacc,
                    "local_mahalanobis_auc": m_auc,
                    "local_mahalanobis_tpr_at_healthy_q95": m_tpr,
                }
            )
    return rows


def run_all_center_scan(cache: dict[tuple[int, ...], list[dict[str, object]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        print(f"[scan] {scenario.key}")
        healthy = load_v(scenario.healthy_dataset)
        damage = load_v(scenario.damage_dataset)
        for center_ch in range(252):
            for spec in cache[(center_ch,)]:
                channels = spec["channels"]
                c_auc, c_bacc = c2st_auc(healthy, damage, channels)
                m_auc, m_tpr = mahalanobis_auc(healthy, damage, channels)
                rows.append(
                    {
                        "scenario": scenario.key,
                        "center_channel": center_ch,
                        "center_is_known_damage": int(center_ch in scenario.known_channels),
                        "ring_index": spec["ring_index"],
                        "max_fem_hop": spec["max_fem_hop"],
                        "n_channels": spec["n_channels"],
                        "channels": "/".join(str(c) for c in channels),
                        "local_c2st_auc": c_auc,
                        "local_c2st_balanced_accuracy": c_bacc,
                        "local_mahalanobis_auc": m_auc,
                        "local_mahalanobis_tpr_at_healthy_q95": m_tpr,
                    }
                )
    return rows


def summarize_scan(scan_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        scenario_rows = [r for r in scan_rows if r["scenario"] == scenario.key]
        for ring_index in range(MAX_SHELLS + 1):
            ring_rows = [r for r in scenario_rows if int(r["ring_index"]) == ring_index]
            by_c2st = sorted(ring_rows, key=lambda r: float(r["local_c2st_auc"]), reverse=True)
            by_maha = sorted(ring_rows, key=lambda r: float(r["local_mahalanobis_auc"]), reverse=True)
            best_known_c2st = min(
                (idx + 1 for idx, row in enumerate(by_c2st) if int(row["center_channel"]) in scenario.known_channels),
                default=None,
            )
            best_known_maha = min(
                (idx + 1 for idx, row in enumerate(by_maha) if int(row["center_channel"]) in scenario.known_channels),
                default=None,
            )
            known_c2st = max(
                (float(row["local_c2st_auc"]) for row in ring_rows if int(row["center_channel"]) in scenario.known_channels),
                default=float("nan"),
            )
            known_maha = max(
                (float(row["local_mahalanobis_auc"]) for row in ring_rows if int(row["center_channel"]) in scenario.known_channels),
                default=float("nan"),
            )
            summary.append(
                {
                    "scenario": scenario.key,
                    "title": scenario.title,
                    "ring_index": ring_index,
                    "best_c2st_center": by_c2st[0]["center_channel"],
                    "best_c2st_auc": by_c2st[0]["local_c2st_auc"],
                    "best_mahalanobis_center": by_maha[0]["center_channel"],
                    "best_mahalanobis_auc": by_maha[0]["local_mahalanobis_auc"],
                    "known_damage_best_c2st_rank": best_known_c2st,
                    "known_damage_best_mahalanobis_rank": best_known_maha,
                    "known_damage_best_c2st_auc": known_c2st,
                    "known_damage_best_mahalanobis_auc": known_maha,
                }
            )
    return summary


def plot_known_ablation(rows: list[dict[str, object]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.3), sharey=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(SCENARIOS)))
    for color, scenario in zip(colors, SCENARIOS):
        scenario_rows = [r for r in rows if r["scenario"] == scenario.key]
        x = [int(r["n_channels"]) for r in scenario_rows]
        label = scenario.title.split("：", 1)[-1]
        axes[0].plot(x, [float(r["local_c2st_auc"]) for r in scenario_rows], marker="o", color=color, label=label)
        axes[1].plot(
            x,
            [float(r["local_mahalanobis_auc"]) for r in scenario_rows],
            marker="o",
            color=color,
            label=label,
        )
    for ax, title in zip(axes, ["监督局部 C2ST/LDA", "健康建模局部 Mahalanobis"]):
        ax.axhline(0.5, color="#666666", lw=1, linestyle="--")
        ax.axhline(0.9, color="#aa0000", lw=1, linestyle=":")
        ax.set_ylim(0.45, 1.03)
        ax.set_xlabel("累计邻域测点数量")
        ax.set_ylabel("AUC")
        ax.set_title(title)
        ax.grid(alpha=0.22)
    axes[0].legend(loc="lower right", frameon=False, fontsize=8)
    fig.suptitle("邻域尺度消融：从结构点单点到周围测点局部联合", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIG_DIR / "known_region_neighborhood_ablation_curve.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_scan_heatmap(scan_rows: list[dict[str, object]], metric: str = "local_c2st_auc") -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(15.5, 10.5), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, scenario in zip(axes, SCENARIOS):
        mat = np.full((MAX_SHELLS + 1, 252), np.nan)
        for row in scan_rows:
            if row["scenario"] == scenario.key:
                mat[int(row["ring_index"]), int(row["center_channel"])] = float(row[metric])
        im = ax.imshow(mat, aspect="auto", interpolation="nearest", vmin=0.5, vmax=1.0, cmap="magma")
        ax.set_title(scenario.title, fontsize=10)
        ax.set_ylabel("邻域圈层")
        for ch in scenario.known_channels:
            ax.axvline(ch, color="#00e5ff", lw=1.2, linestyle="--")
        ax.set_yticks(range(MAX_SHELLS + 1))
    axes[-1].set_xlabel("中心测点通道")
    axes[-2].set_xlabel("中心测点通道")
    fig.subplots_adjust(left=0.07, right=0.88, top=0.91, bottom=0.08, hspace=0.32, wspace=0.12)
    cax = fig.add_axes([0.90, 0.18, 0.018, 0.64])
    fig.colorbar(im, cax=cax, label=metric)
    title = "每个中心测点的局部邻域 C2ST AUC 扫描" if metric == "local_c2st_auc" else "每个中心测点的局部 Mahalanobis AUC 扫描"
    fig.suptitle(title, fontsize=15, fontweight="bold")
    out = FIG_DIR / f"all_center_scan_{metric}.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_known_rank(summary_rows: list[dict[str, object]]) -> Path:
    fig, ax = plt.subplots(figsize=(13.5, 6.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(SCENARIOS)))
    for color, scenario in zip(colors, SCENARIOS):
        rows = [r for r in summary_rows if r["scenario"] == scenario.key]
        x = [int(r["ring_index"]) for r in rows]
        y = [int(r["known_damage_best_c2st_rank"]) for r in rows]
        ax.plot(x, y, marker="o", color=color, label=scenario.title.split("：", 1)[-1])
    ax.invert_yaxis()
    ax.set_xlabel("邻域圈层 ring index")
    ax.set_ylabel("已知结构点在 252 个中心中的 C2ST AUC 排名（越小越好）")
    ax.set_title("结构点局部邻域是否在全测点扫描中靠前")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "known_center_rank_by_ring.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def write_report(
    known_rows: list[dict[str, object]],
    scan_summary: list[dict[str, object]],
    figures: list[Path],
) -> None:
    lines = [
        "# AE-main 局部邻域尺度消融",
        "",
        "更新时间：2026-05-29",
        "",
        "## 实验名称",
        "",
        "这个实验可以叫 **邻域尺度消融实验**，英文写作 `neighborhood-radius ablation` 或 `local-neighborhood ablation`。它的含义是：从结构变化测点本身开始，按 FEM 边邻接图一圈一圈加入周围已测通道，观察局部联合分布的识别能力如何变化。",
        "",
        "邻域定义：先用 `adjacency_graph.pkl` 的元素边邻接图计算测点到测点的最短 hop 距离；`ring0` 只含中心测点，`ring1` 加入最近一圈已测点，`ring2` 再加入下一圈，直到 `ring5`。由于 252 个测点稀疏，最近一圈已测点通常对应约 9 个 FEM element hop，而不是 1 个 element hop。",
        "",
        "## 一页结论",
        "",
        "**局部邻域联合分布比单点更有意义，而且不需要扩到全 252 通道。** 对已知结构变化位置做邻域尺度消融后，只加入最近一圈已测点，局部监督 C2ST/LDA 就从单点的约 0.73-0.83 提升到约 0.918-1.000；修补后二次损伤和组合二次损伤也从 0.751 提升到 0.956。ring2 以后基本进入平台期。",
        "",
        "**只用健康样本建模的局部 Mahalanobis 也很强。** 单点时 AUC 只有约 0.684-0.775，但 ring2 后多数场景达到 0.972-1.000，健康 95% 阈值下的损伤命中率达到 0.92-1.00。这说明真正有效的信息不是孤立单点，而是结构点附近少数测点之间的局部联合变化。",
        "",
        "**全测点扫描也支持这个定位。** ring0/ring1 时，已知结构变化点通常就是 252 个中心里排名第 1 的中心；ring2 以后，一些邻近中心也达到相近 AUC，因为它们的局部窗口已经和结构点窗口高度重叠。这不是反例，而是说明损伤信息在附近小区域内扩散。",
        "",
        "## 已知结构点邻域尺度曲线",
        "",
        "| 场景 | ring | FEM hop | 邻域通道数 | 通道集合 | C2ST AUC | Mahalanobis AUC | Mahalanobis TPR@q95 |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in known_rows:
        lines.append(
            f"| {row['title']} | {row['ring_index']} | {row['max_fem_hop']} | {row['n_channels']} | "
            f"{row['channels']} | {fmt(row['local_c2st_auc'])} | {fmt(row['local_mahalanobis_auc'])} | "
            f"{fmt(row['local_mahalanobis_tpr_at_healthy_q95'])} |"
        )
    lines.extend(
        [
            "",
            "## 全测点局部扫描摘要",
            "",
            "这个表检查：如果把 252 个测点都当作中心点扫描，已知结构变化点的局部邻域是否排在前面。排名越小越好。",
            "",
            "| 场景 | ring | C2ST 最强中心 | 最强 AUC | 已知结构点最好排名 | 已知结构点 AUC | Mahalanobis 最强中心 | 已知结构点 Mahalanobis 排名 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in scan_summary:
        lines.append(
            f"| {row['title']} | {row['ring_index']} | {row['best_c2st_center']} | {fmt(row['best_c2st_auc'])} | "
            f"{row['known_damage_best_c2st_rank']} | {fmt(row['known_damage_best_c2st_auc'])} | "
            f"{row['best_mahalanobis_center']} | {row['known_damage_best_mahalanobis_rank']} |"
        )
    lines.extend(["", "## 图", ""])
    for fig in figures:
        lines.extend([f"![{fig.stem}]({fig.relative_to(OUT_ROOT).as_posix()})", ""])
    lines.extend(
        [
            "## 最终判断",
            "",
            "1. 这个新增实验支持一个更细的结论：不必只在“单点”和“全 252 维联合”之间二选一，结构点周围的局部邻域联合分布是一个合理中间层。",
            "2. 对初始损伤、双圆损伤、修补后二次损伤和组合二次损伤，局部邻域尺度增加后 AUC 都快速提升，说明损伤信息在结构点附近有空间扩散，不只是一个孤立通道。",
            "3. 最关键的工程结论是：单点不够稳，但“结构点 + 最近一圈/两圈测点”的局部联合窗口已经很强；这比全 252 维联合更可解释，也更像一个可以写进论文的方法模块。",
            "4. 如果论文要强调统计路线，可以表述为：先做结构/传感器拓扑约束的局部邻域尺度消融，再选择足够小但 AUC 稳定的局部联合窗口，而不是直接上全局高维模型。",
        ]
    )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def markdown_to_html(md_text: str) -> str:
    try:
        import markdown  # type: ignore

        return markdown.markdown(md_text, extensions=["tables", "fenced_code", "toc"])
    except Exception:
        return f"<pre>{html.escape(md_text)}</pre>"


def write_html_report() -> None:
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
        "<title>AE-main 局部邻域尺度消融</title>"
        f"<style>{css}</style></head><body><main>"
        '<div class="notice">固定可打开版本：HTML 和图片都在 '
        '<code>script/AH_local_neighborhood_ablation_output/html_report/</code> 下。</div>'
        f"{body}</main></body></html>",
        encoding="utf-8",
    )


def main() -> None:
    setup_fonts()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    measure_ids = load_measure_ids()
    adjacency = load_adjacency()
    print("[neighborhood] building measured-ring cache")
    cache = build_neighborhood_cache(measure_ids, adjacency)
    print("[known] running known-region neighborhood ablation")
    known_rows = run_known_ablation(cache)
    write_csv(KNOWN_CURVE_CSV, known_rows)
    print("[scan] running all-center local scan")
    scan_rows = run_all_center_scan(cache)
    write_csv(LOCAL_SCAN_CSV, scan_rows)
    scan_summary = summarize_scan(scan_rows)
    write_csv(SCAN_SUMMARY_CSV, scan_summary)
    figures = [
        plot_known_ablation(known_rows),
        plot_scan_heatmap(scan_rows, "local_c2st_auc"),
        plot_scan_heatmap(scan_rows, "local_mahalanobis_auc"),
        plot_known_rank(scan_summary),
    ]
    write_report(known_rows, scan_summary, figures)
    write_html_report()
    print(f"report={REPORT_MD}")
    print(f"html={HTML_INDEX}")
    print(f"known_csv={KNOWN_CURVE_CSV}")
    print(f"scan_csv={LOCAL_SCAN_CSV}")
    print(f"summary_csv={SCAN_SUMMARY_CSV}")
    print(f"figures={FIG_DIR}")


if __name__ == "__main__":
    main()
