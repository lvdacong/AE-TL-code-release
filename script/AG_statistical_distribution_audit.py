# -*- coding: utf-8 -*-
"""Statistical audit for AE-main damage-vs-healthy distributions.

The goal is to separate three questions:
1. Are healthy and damaged samples from a different joint distribution?
2. Can a simple statistical detector identify single damaged samples?
3. Is the known structural-change channel alone sufficient?
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
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, permutation_test_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "script" / "AD_preprocess_datasets_output"
OUT_ROOT = ROOT / "script" / "AG_statistical_distribution_audit_output"
FIG_DIR = OUT_ROOT / "figures"
HTML_ROOT = OUT_ROOT / "html_report"
HTML_FIG_DIR = HTML_ROOT / "figures"
REPORT_MD = OUT_ROOT / "AE-main损伤前后统计分布复审.md"
HTML_INDEX = HTML_ROOT / "index.html"
HTML_REPORT_MD = HTML_ROOT / "report.md"
JOINT_CSV = OUT_ROOT / "joint_distribution_results.csv"
SINGLE_CSV = OUT_ROOT / "known_single_channel_summary.csv"
SINGLE_DETAIL_CSV = OUT_ROOT / "known_single_channel_detail.csv"
TOP_CSV = OUT_ROOT / "top_univariate_channels.csv"


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
    path = DATA_ROOT / dataset / "preprocessed_data_raw.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)["V"].astype(np.float32)


def abs_z_scores(healthy: np.ndarray, damage: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = healthy.mean(axis=0)
    sd = healthy.std(axis=0) + 1e-9
    return np.abs((healthy - mu) / sd), np.abs((damage - mu) / sd)


def threshold_metrics(score_h: np.ndarray, score_d: np.ndarray) -> tuple[float, float, float]:
    labels = np.r_[np.zeros(len(score_h), dtype=int), np.ones(len(score_d), dtype=int)]
    scores = np.r_[score_h, score_d]
    auc = float(roc_auc_score(labels, scores))
    threshold = float(np.quantile(score_h, 0.95))
    tpr = float(np.mean(score_d > threshold))
    return auc, threshold, tpr


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        original_rank = n - rank + 1
        running = min(running, p[idx] * n / original_rank)
        adjusted[idx] = running
    return np.clip(adjusted, 0.0, 1.0)


def scan_univariate(scenario: Scenario, healthy: np.ndarray, damage: np.ndarray) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    z_h, z_d = abs_z_scores(healthy, damage)
    labels = np.r_[np.zeros(len(healthy), dtype=int), np.ones(len(damage), dtype=int)]
    p_values = []
    rows = []
    for channel in range(healthy.shape[1]):
        raw_scores = np.r_[healthy[:, channel], damage[:, channel]]
        raw_auc = float(roc_auc_score(labels, raw_scores))
        directionless_auc = max(raw_auc, 1.0 - raw_auc)
        absz_auc, absz_threshold, absz_tpr = threshold_metrics(z_h[:, channel], z_d[:, channel])
        ks = stats.ks_2samp(healthy[:, channel], damage[:, channel], alternative="two-sided", mode="auto")
        mwu = stats.mannwhitneyu(healthy[:, channel], damage[:, channel], alternative="two-sided")
        pooled = np.sqrt((healthy[:, channel].var(ddof=1) + damage[:, channel].var(ddof=1)) / 2)
        cohen_d = float((damage[:, channel].mean() - healthy[:, channel].mean()) / (pooled + 1e-12))
        row = {
            "scenario": scenario.key,
            "channel": channel,
            "raw_auc_directionless": directionless_auc,
            "raw_auc_signed_damage_high": raw_auc,
            "absz_auc": absz_auc,
            "absz_tpr_at_healthy_q95": absz_tpr,
            "ks_p": float(ks.pvalue),
            "mannwhitney_p": float(mwu.pvalue),
            "cohen_d_damage_minus_healthy": cohen_d,
        }
        rows.append(row)
        p_values.append(row["mannwhitney_p"])
    adjusted = bh_fdr(np.array(p_values, dtype=float))
    for row, adj in zip(rows, adjusted):
        row["mannwhitney_fdr_q"] = float(adj)
    known_rows = [row | {"known_label": scenario.known_label} for row in rows if row["channel"] in scenario.known_channels]
    return rows, known_rows


def run_joint_c2st(scenario: Scenario, healthy: np.ndarray, damage: np.ndarray) -> dict[str, object]:
    data = np.vstack([healthy, damage])
    labels = np.r_[np.zeros(len(healthy), dtype=int), np.ones(len(damage), dtype=int)]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    estimator = make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    )
    proba = cross_val_predict(estimator, data, labels, cv=cv, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = float(roc_auc_score(labels, proba))
    bacc = float(balanced_accuracy_score(labels, pred))
    score, _, p_value = permutation_test_score(
        estimator,
        data,
        labels,
        scoring="roc_auc",
        cv=cv,
        n_permutations=199,
        n_jobs=-1,
        random_state=7,
    )
    return {
        "scenario": scenario.key,
        "title": scenario.title,
        "healthy_dataset": scenario.healthy_dataset,
        "damage_dataset": scenario.damage_dataset,
        "n_healthy": len(healthy),
        "n_damage": len(damage),
        "joint_c2st_lda_auc_oof": auc,
        "joint_c2st_lda_balanced_accuracy_oof": bacc,
        "joint_c2st_lda_auc_cv_mean": float(score),
        "joint_c2st_permutation_p": float(p_value),
    }


def summarize_known_channels(scenario: Scenario, healthy: np.ndarray, damage: np.ndarray) -> dict[str, object]:
    z_h, z_d = abs_z_scores(healthy, damage)
    score_h = z_h[:, scenario.known_channels].max(axis=1)
    score_d = z_d[:, scenario.known_channels].max(axis=1)
    known_auc, known_thr, known_tpr = threshold_metrics(score_h, score_d)
    all_auc, _, all_tpr = threshold_metrics(z_h.max(axis=1), z_d.max(axis=1))
    return {
        "scenario": scenario.key,
        "known_channels": "/".join(str(c) for c in scenario.known_channels),
        "known_label": scenario.known_label,
        "known_absz_auc": known_auc,
        "known_absz_threshold_q95": known_thr,
        "known_absz_tpr_at_healthy_q95": known_tpr,
        "all_channel_zmax_auc": all_auc,
        "all_channel_zmax_tpr_at_healthy_q95": all_tpr,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, digits: int = 3) -> str:
    if value is None or value == "":
        return ""
    value = float(value)
    if abs(value) < 0.001 and value != 0:
        return f"{value:.1e}"
    return f"{value:.{digits}f}"


def plot_auc_summary(joint_rows: list[dict[str, object]], known_rows: list[dict[str, object]]) -> Path:
    labels = [row["title"].split("：", 1)[-1] for row in joint_rows]
    joint_auc = [float(row["joint_c2st_lda_auc_oof"]) for row in joint_rows]
    known_auc = [float(row["known_absz_auc"]) for row in known_rows]
    zmax_auc = [float(row["all_channel_zmax_auc"]) for row in known_rows]
    x = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    ax.bar(x - width, joint_auc, width, label="联合分布 C2ST/LDA AUC", color="#355c9a")
    ax.bar(x, zmax_auc, width, label="全部通道 z-max AUC", color="#6aa84f")
    ax.bar(x + width, known_auc, width, label="已知结构通道 abs-z AUC", color="#e69138")
    ax.axhline(0.5, color="#666666", lw=1, linestyle="--")
    ax.axhline(0.9, color="#aa0000", lw=1, linestyle=":")
    ax.set_ylim(0.45, 1.03)
    ax.set_ylabel("AUC")
    ax.set_title("联合分布统计分类 vs 单通道/简单异常分数")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    out = FIG_DIR / "joint_vs_single_auc.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_known_channel_distributions(single_rows: list[dict[str, object]]) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2), sharey=False)
    axes = axes.ravel()
    for ax, scenario in zip(axes, SCENARIOS):
        healthy = load_v(scenario.healthy_dataset)
        damage = load_v(scenario.damage_dataset)
        z_h, z_d = abs_z_scores(healthy, damage)
        score_h = z_h[:, scenario.known_channels].max(axis=1)
        score_d = z_d[:, scenario.known_channels].max(axis=1)
        ax.hist(score_h, bins=34, density=True, alpha=0.62, label="healthy", color="#4c78a8")
        ax.hist(score_d, bins=24, density=True, alpha=0.62, label="damage", color="#f58518")
        threshold = np.quantile(score_h, 0.95)
        ax.axvline(threshold, color="#333333", lw=1.1, linestyle="--", label="healthy q95")
        row = next(r for r in single_rows if r["scenario"] == scenario.key)
        ax.set_title(f"{scenario.title}\n{scenario.known_label}\nAUC={fmt(row['known_absz_auc'])}, TPR@q95={fmt(row['known_absz_tpr_at_healthy_q95'])}", fontsize=10)
        ax.set_xlabel("known-channel max abs-z")
        ax.set_ylabel("density")
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("已知结构变化通道的单点/少数点分布：显著但仍有重叠", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIG_DIR / "known_channel_distributions.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def write_report(
    joint_rows: list[dict[str, object]],
    known_summary_rows: list[dict[str, object]],
    known_detail_rows: list[dict[str, object]],
    top_rows: list[dict[str, object]],
    figures: list[Path],
) -> None:
    lines = [
        "# AE-main 损伤前后统计分布复审",
        "",
        "更新时间：2026-05-29",
        "",
        "## 一页结论",
        "",
        "**联合分布层面：损伤前后确实不是同一个分布。** 用 C2ST（classifier two-sample test，分类器两样本检验）加收缩 LDA 进行 5 折交叉验证，所有损伤对比的联合分布 AUC 都在 0.956 以上，置换检验 p 值达到本次 199 次置换的下限 `p=0.005`。因此，如果问题改成“给定健康/损伤样本，纯统计方法能否证明并识别两类分布不同”，答案是能。",
        "",
        "**单点结构通道层面：也显著不同，但不能说已经稳定直接分开。** 已知损伤通道的 KS/Mann-Whitney p 值很小，说明单点分布确有变化；但已知结构通道的单样本 AUC 主要在 0.684-0.774，健康 95% 阈值下的损伤命中率只有 0.40-0.47。它适合做“定位证据”和“批量统计显著性”，不适合作为稳健的单样本检测器。",
        "",
        "**因此要分清两个层级：** 若只写统计验证，AE-main 可以用联合分布统计分类器证明损伤前后可分；若要做工程检测，单点或 z-max 仍不够，至少需要联合统计判别器，最好再讨论无损伤标签时如何从健康流形构造异常分数。",
        "",
        "## 联合分布检验",
        "",
        "方法：收缩 LDA 作为统计判别器，5 折交叉验证得到 out-of-fold AUC；再用 199 次标签置换做 C2ST 显著性检验。这里 p=0.005 是置换次数下的最小可分辨 p 值。",
        "",
        "| 场景 | 健康/损伤样本数 | 联合 C2ST AUC | Balanced Acc. | 置换 p 值 | 判断 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in joint_rows:
        verdict = "联合分布显著不同，可统计识别" if float(row["joint_c2st_lda_auc_oof"]) >= 0.90 else "差异偏弱"
        lines.append(
            f"| {row['title']} | {row['n_healthy']}/{row['n_damage']} | "
            f"{fmt(row['joint_c2st_lda_auc_oof'])} | {fmt(row['joint_c2st_lda_balanced_accuracy_oof'])} | "
            f"{fmt(row['joint_c2st_permutation_p'])} | {verdict} |"
        )
    lines.extend(
        [
            "",
            "## 单点/简单异常分数",
            "",
            "已知结构通道分数定义：只取已知结构变化通道上的 `abs((x - healthy_mean) / healthy_std)`；如果一个位置有两个直接测点，则取两者最大值。TPR@q95 表示阈值固定为健康样本 95% 分位时，损伤样本被命中的比例。",
            "",
            "| 场景 | 已知通道 | 已知通道 AUC | 已知通道 TPR@q95 | 全通道 z-max AUC | 全通道 z-max TPR@q95 | 判断 |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in known_summary_rows:
        verdict = "显著但重叠明显，不能单独稳健检测"
        if float(row["known_absz_auc"]) >= 0.90 and float(row["known_absz_tpr_at_healthy_q95"]) >= 0.80:
            verdict = "单点基本可直接检测"
        lines.append(
            f"| {next(s.title for s in SCENARIOS if s.key == row['scenario'])} | {row['known_channels']} | "
            f"{fmt(row['known_absz_auc'])} | {fmt(row['known_absz_tpr_at_healthy_q95'])} | "
            f"{fmt(row['all_channel_zmax_auc'])} | {fmt(row['all_channel_zmax_tpr_at_healthy_q95'])} | {verdict} |"
        )
    lines.extend(
        [
            "",
            "## 已知结构通道的显著性",
            "",
            "| 场景 | 通道 | raw AUC(方向无关) | abs-z AUC | TPR@q95 | KS p | MWU p | FDR q |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in known_detail_rows:
        lines.append(
            f"| {next(s.title for s in SCENARIOS if s.key == row['scenario'])} | {row['channel']} | "
            f"{fmt(row['raw_auc_directionless'])} | {fmt(row['absz_auc'])} | "
            f"{fmt(row['absz_tpr_at_healthy_q95'])} | {fmt(row['ks_p'])} | "
            f"{fmt(row['mannwhitney_p'])} | {fmt(row['mannwhitney_fdr_q'])} |"
        )
    best_rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        scenario_rows = [row for row in top_rows if row["scenario"] == scenario.key]
        best_rows.append(max(scenario_rows, key=lambda r: float(r["absz_auc"])))
    lines.extend(
        [
            "",
            "## 全通道单变量扫描",
            "",
            "这个表回答“如果不预先告诉结构位置，只扫 252 个通道，最强单通道是不是结构变化点”。结果显示，最强通道基本就是已知结构变化通道，但强度仍只到中等 AUC。",
            "",
            "| 场景 | 最强单通道 | 是否已知结构点 | abs-z AUC | TPR@q95 | FDR q |",
            "| --- | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in best_rows:
        scenario = next(s for s in SCENARIOS if s.key == row["scenario"])
        is_known = "是" if int(row["channel"]) in scenario.known_channels else "否"
        lines.append(
            f"| {scenario.title} | {row['channel']} | {is_known} | "
            f"{fmt(row['absz_auc'])} | {fmt(row['absz_tpr_at_healthy_q95'])} | {fmt(row['mannwhitney_fdr_q'])} |"
        )
    lines.extend(
        [
            "",
            "## 图",
            "",
        ]
    )
    for fig in figures:
        lines.extend([f"![{fig.stem}]({fig.relative_to(OUT_ROOT).as_posix()})", ""])
    lines.extend(
        [
            "## 解释与最终判断",
            "",
            "1. “损伤前后联合分布不同”这个命题成立。C2ST/LDA 的 AUC 很高，说明 252 通道之间的联合模式确实包含损伤信息。",
            "2. “单个结构变化通道就能直接解决”这个命题只成立一半。它在统计检验上显著，而且通常就是最强单通道；但分布仍有重叠，阈值按健康 95% 分位控制误报时，只能命中约 40%-47% 的损伤样本。",
            "3. “不需要联合分布”不建议这样下结论。更准确的说法是：单点能提供强定位证据，联合统计判别器能提供高识别能力；如果论文目标是检测，联合统计方法比肉眼热图和单点阈值更合理。",
            "4. 和深度 AE/TL 的关系：这次证明了不一定非要深度模型才能识别，传统统计判别器已经很强；但它用了健康/损伤标签。若实验设定只能拿到少量目标健康样本、没有目标损伤标签，那么仍需要异常检测/健康流形建模，而不能直接套监督 LDA。",
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
        "<title>AE-main 损伤前后统计分布复审</title>"
        f"<style>{css}</style></head><body><main>"
        '<div class="notice">固定可打开版本：HTML 和图片都在 '
        '<code>script/AG_statistical_distribution_audit_output/html_report/</code> 下。</div>'
        f"{body}</main></body></html>",
        encoding="utf-8",
    )


def main() -> None:
    setup_fonts()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    joint_rows: list[dict[str, object]] = []
    known_summary_rows: list[dict[str, object]] = []
    known_detail_rows: list[dict[str, object]] = []
    top_rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        healthy = load_v(scenario.healthy_dataset)
        damage = load_v(scenario.damage_dataset)
        joint_rows.append(run_joint_c2st(scenario, healthy, damage))
        scan_rows, known_rows = scan_univariate(scenario, healthy, damage)
        known_detail_rows.extend(known_rows)
        top_rows.extend(sorted(scan_rows, key=lambda r: float(r["absz_auc"]), reverse=True)[:10])
        known_summary_rows.append(summarize_known_channels(scenario, healthy, damage))
    write_csv(JOINT_CSV, joint_rows)
    write_csv(SINGLE_CSV, known_summary_rows)
    write_csv(SINGLE_DETAIL_CSV, known_detail_rows)
    write_csv(TOP_CSV, top_rows)
    figures = [
        plot_auc_summary(joint_rows, known_summary_rows),
        plot_known_channel_distributions(known_summary_rows),
    ]
    write_report(joint_rows, known_summary_rows, known_detail_rows, top_rows, figures)
    write_html_report()
    print(f"report={REPORT_MD}")
    print(f"html={HTML_INDEX}")
    print(f"joint_csv={JOINT_CSV}")
    print(f"single_csv={SINGLE_CSV}")
    print(f"single_detail_csv={SINGLE_DETAIL_CSV}")
    print(f"top_csv={TOP_CSV}")
    print(f"figures={FIG_DIR}")


if __name__ == "__main__":
    main()
