"""
AE_gen_paper_macros.py
======================
Scan all relevant CSV outputs and generate LaTeX macro + table-body files
for paper_TL/. This is the SOURCE-OF-TRUTH bridge between code outputs and
the paper text.

Workflow:
    Phase B (re-run experiments)
        ↓
    python AE_gen_paper_macros.py     ← THIS SCRIPT
        ↓
    pdflatex paper_TL_en.tex / xelatex paper_TL_cn.tex

Outputs (all under paper_TL/generated/, never edit by hand):
    paper_macros.tex                  - all named numeric macros
    tab_detection_metrics_body.tex    - main 9-row max-channel table
    tab_dr_mean_vs_max_body.tex       - DR-only mean-vs-max comparison
    tab_sequential_detection_body.tex - sequential adaptation AUC by ordering

Macro naming convention:
    \\<sc><model><metric>{<mode>}
    sc      : dr | so | sd
    model   : TL | Old | FS
    metric  : AUC | TPR | Fone | valloss
    mode    : max | mean   (omitted for valloss)
    examples: \\drTLAUCmax, \\soOldAUCmean, \\sdTLvalloss
"""
from __future__ import annotations

import os
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime
from pathlib import Path

import json

import numpy as np
import pandas as pd

# ========================================
# Paths
# ========================================
SCRIPT_DIR = Path(__file__).parent
AE_OUT = SCRIPT_DIR / "AE_model_train_and_detect_output"
PAPER_DIR = SCRIPT_DIR.parent / "paper_TL"
GEN_DIR = PAPER_DIR / "generated"
GEN_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = {
    "dr": ("Damage_Repaired", "Structural Repair", "结构修复"),
    "so": ("Sensor_Offset", "Sensor Relocation", "传感器迁移"),
    "sd": ("Sensor_Drift", "Sensor Drift", "传感器漂移"),
}

MODELS = [("TL", "TL"), ("Old", "OldModel"), ("FS", "FromScratch")]

# Display labels for table rows — keep lookup keys (m_full) distinct from display names
MODEL_DISPLAY = {
    "en": {"TL": "TL", "OldModel": "Unadapted source model", "FromScratch": "Scratch-trained baseline"},
    "cn": {"TL": "TL", "OldModel": "未适应源域模型", "FromScratch": "随机初始化基线"},
}


# ========================================
# Formatting helpers
# ========================================
def fmt_decimal(v: float, places: int) -> str:
    quantum = Decimal("1").scaleb(-places)
    return str(Decimal(str(v)).quantize(quantum, rounding=ROUND_HALF_UP))


def fmt_auc(v: float) -> str:    return fmt_decimal(v, 3)
def fmt_ci(lo: float, hi: float) -> str: return f"[{fmt_decimal(lo, 2)}, {fmt_decimal(hi, 2)}]"
def fmt_tpr(v: float) -> str:    return fmt_decimal(v, 2)
def fmt_loss(v: float) -> str:   return fmt_decimal(v, 5)
def fmt_loss4(v: float) -> str:  return fmt_decimal(v, 4)
def fmt_loss_short(v: float) -> str:
    """E.g. 0.001229 -> 0.00123, 0.024 -> 0.024."""
    if v >= 0.01: return f"{v:.3f}"
    return f"{v:.5f}"
def fmt_pct(v: float) -> str:    return f"{v:.0f}"
def fmt_pct1(v: float) -> str:   return f"{v:.1f}"
def fmt_ratio(v: float) -> str:  return f"{(int(v * 10 + 0.5)) / 10:.1f}"


# ========================================
# Data loaders
# ========================================
def load_detection(scenario_folder: str, mode: str) -> pd.DataFrame:
    """mode = 'max' or 'mean' (mean uses no suffix)."""
    suffix = "_max" if mode == "max" else ""
    csv = AE_OUT / scenario_folder / "tl_comparison" / f"detection_metrics{suffix}.csv"
    return pd.read_csv(csv).set_index("Model")


def load_final_val_loss(scenario_folder: str, model_sub: str) -> float:
    """Read training_losses.csv and return final epoch val_loss."""
    csv = AE_OUT / scenario_folder / "tl_comparison" / model_sub / "training_losses.csv"
    return float(pd.read_csv(csv).iloc[-1]["val_loss"])


def load_difficulty(scenario_folder: str) -> pd.DataFrame:
    csv = AE_OUT / scenario_folder / "difficulty_ablation" / f"difficulty_ablation_{scenario_folder}.csv"
    return pd.read_csv(csv)


def load_adam_uniform_so(mode: str = "max") -> pd.DataFrame:
    """SO Adam-Uniform detection metrics for DiffLR ablation."""
    suffix = "_max" if mode == "max" else ""
    csv = AE_OUT / "Sensor_Offset" / "tl_comparison_adam_uniform" / f"detection_metrics{suffix}.csv"
    return pd.read_csv(csv).set_index("Model")


def load_channel_ablation(scenario_folder: str) -> pd.DataFrame:
    """Load per-channel ablation results: difficulty × channel × {Pre-Shift,Post-Shift,N=50..400}."""
    csv = AE_OUT / scenario_folder / "difficulty_ablation" / "channel_ablation_results.csv"
    return pd.read_csv(csv)


def load_recovery_phase(phase: int) -> pd.DataFrame:
    """Load recovery study phase comparison CSV."""
    csv = AE_OUT / "Damage_Repaired" / "recovery_study" / f"phase{phase}" / f"phase{phase}_comparison.csv"
    return pd.read_csv(csv).set_index("experiment")


def load_standard_sgd_so(mode: str = "max") -> pd.DataFrame:
    """SO Standard-SGD detection metrics for DiffLR ablation."""
    suffix = "_max" if mode == "max" else ""
    csv = AE_OUT / "Sensor_Offset" / "tl_comparison_standard" / f"detection_metrics{suffix}.csv"
    return pd.read_csv(csv).set_index("Model")


# ========================================
# Generate paper_macros.tex
# ========================================
def gen_macros() -> None:
    L = []
    L.append(f"% Auto-generated by AE_gen_paper_macros.py at {datetime.now():%Y-%m-%d %H:%M:%S}")
    L.append("% DO NOT EDIT MANUALLY. Re-run AE_gen_paper_macros.py after any CSV change.")
    L.append("% Only macros actually referenced in paper/cover-letter are generated.")
    L.append("")

    # ----- TL max-channel AUC (all scenarios); TPR + F1 only for SO (§6.2 ablation) -----
    L.append("% ============ TL detection metrics (max-channel) ============")
    for sc_short, (folder, _, _) in SCENARIOS.items():
        df = load_detection(folder, "max")
        L.append(rf"\newcommand{{\{sc_short}TLAUCmax}}{{{fmt_auc(df.loc['TL', 'AUC'])}}}")
        if sc_short == "so":
            L.append(rf"\newcommand{{\{sc_short}TLTPRmax}}{{{fmt_tpr(df.loc['TL', 'TPR@FPR5%'])}}}")
            L.append(rf"\newcommand{{\{sc_short}TLFonemax}}{{{fmt_auc(df.loc['TL', 'Best_F1'])}}}")
    L.append("")

    # ----- DR scratch-trained baseline AUC (referenced in §5.1 honest disclosure) -----
    _dr_max_df = load_detection("Damage_Repaired", "max")
    L.append(r"% ============ DR FromScratch AUC (for §5.1 honest disclosure) ============")
    L.append(rf"\newcommand{{\drScratchAUCmax}}{{{fmt_auc(_dr_max_df.loc['FromScratch', 'AUC'])}}}")
    L.append(rf"\newcommand{{\drSourceAUCmax}}{{{fmt_auc(_dr_max_df.loc['OldModel', 'AUC'])}}}")
    L.append("")

    # ----- Val losses (compute only, for fold ratio below — no macros emitted) -----
    val_losses = {}
    for sc_short, (folder, _, _) in SCENARIOS.items():
        for sub_short, sub in [("TL", "TL"), ("FS", "fromscratch")]:
            val_losses[(sc_short, sub_short)] = load_final_val_loss(folder, sub)

    # ----- SO DiffLR ablation (6 macros: AUC + TPR + F1 for Adam-Uniform + SGD) -----
    L.append("% ============ SO DiffLR ablation (Adam-Uniform + SGD) ============")
    df_au = load_adam_uniform_so("max")
    L.append(rf"\newcommand{{\soAdamUnifAUCmax}}{{{fmt_auc(df_au.loc['TL', 'AUC'])}}}")
    L.append(rf"\newcommand{{\soAdamUnifTPRmax}}{{{fmt_tpr(df_au.loc['TL', 'TPR@FPR5%'])}}}")
    L.append(rf"\newcommand{{\soAdamUnifFonemax}}{{{fmt_auc(df_au.loc['TL', 'Best_F1'])}}}")
    df_sgd_so = load_standard_sgd_so("max")
    L.append(rf"\newcommand{{\soStdSGDAUCmax}}{{{fmt_auc(df_sgd_so.loc['TL', 'AUC'])}}}")
    L.append(rf"\newcommand{{\soStdSGDTPRmax}}{{{fmt_tpr(df_sgd_so.loc['TL', 'TPR@FPR5%'])}}}")
    L.append(rf"\newcommand{{\soStdSGDFonemax}}{{{fmt_auc(df_sgd_so.loc['TL', 'Best_F1'])}}}")
    L.append("")

    # ----- Fold ratio range (2 macros: foldRatioMin, foldRatioMax) -----
    L.append("% ============ Reconstruction error fold ratio ============")
    ratios_fs = [val_losses[(sc, "FS")] / val_losses[(sc, "TL")] for sc in SCENARIOS]
    L.append(rf"\newcommand{{\foldRatioMin}}{{{fmt_ratio(min(ratios_fs))}}}")
    L.append(rf"\newcommand{{\foldRatioMax}}{{{fmt_ratio(max(ratios_fs))}}}")
    L.append("")

    # ----- Fixed configuration constants (hand-specified, single source of truth) -----
    L.append("% ============ Fine-tuning hyperparameters ============")
    L.append(r"\newcommand{\encLr}{10^{-5}}")
    L.append(r"\newcommand{\decLr}{10^{-3}}")
    L.append(r"\newcommand{\noiseAugStd}{0.005}")
    L.append(r"\newcommand{\augSigmaRatio}{0.01}")
    L.append(r"\newcommand{\numTgtSamples}{400}")
    L.append(r"\newcommand{\numTgtValSamples}{100}")
    L.append(r"\newcommand{\numChannels}{252}")
    L.append(r"\newcommand{\driftRatio}{0.10}")
    L.append(r"\newcommand{\numSeqEpochs}{200}")
    L.append(r"\newcommand{\numSeqSamples}{400}")
    L.append(r"\newcommand{\prelimSGDChanRecovery}{93}")
    L.append("")

    # ----- Source-domain pre-training and baseline test sample counts -----
    L.append("% ============ Source-domain / baseline sample counts ============")
    L.append(r"\newcommand{\numSrcSamples}{2000}")
    L.append(r"\newcommand{\numSrcTrainSamples}{1800}")
    L.append(r"\newcommand{\numSrcValSamples}{200}")
    L.append(r"\newcommand{\numPreEpochs}{2000}")
    L.append(r"\newcommand{\numFTEpochs}{1000}")
    L.append(r"\newcommand{\preLrInit}{3 \times 10^{-4}}")
    L.append(r"\newcommand{\preLrFinal}{10^{-6}}")
    L.append(r"\newcommand{\trainBatchSize}{256}")
    L.append(r"\newcommand{\numBaselineTestHealthy}{200}")
    L.append(r"\newcommand{\numBaselineTestDamage}{100}")
    L.append("")

    # ----- Optimizer study measured outcomes (§5.2.1 preliminary SGD vs Adam) -----
    L.append("% ============ Optimizer study measured outcomes ============")
    L.append(r"\newcommand{\optStudyEpochs}{500}")
    L.append(r"\newcommand{\optAdamSpikeLoss}{0.07}")
    L.append(r"\newcommand{\optPreBaseline}{0.00186}")
    L.append(r"\newcommand{\optSpikeRatio}{35}")
    L.append(r"\newcommand{\optSpikeOscEpochs}{200}")
    L.append(r"\newcommand{\optSGDFinalLoss}{0.00125}")
    L.append(r"\newcommand{\optAdamFinalLoss}{0.00149}")
    L.append(r"\newcommand{\optAdamWarmFinalLoss}{0.00147}")
    L.append(r"\newcommand{\optSGDRecoveryPct}{0.3}")
    L.append(r"\newcommand{\optAdamOvershootMin}{14}")
    L.append(r"\newcommand{\optAdamOvershootMax}{15}")
    L.append("")

    # ----- LR tuning / freeze-study measured outcomes (§5.2.2-5.2.3) -----
    # Numeric values in §5.2.2 prose were removed per §五.3 (caption/table/text
    # keeps each figure once); the lr_tuning table body already carries the
    # quantitative rows. Only the macros still referenced in prose remain.
    L.append("% ============ LR tuning / freeze study measured outcomes ============")
    L.append(r"\newcommand{\lrBaselineChan}{86}")
    L.append(r"\newcommand{\freezeMaxDeltaPct}{0.1}")
    L.append("")

    # ----- SO correlation disruption contribution range (§6.3) -----
    L.append("% ============ SO correlation disruption contribution range ============")
    L.append(r"\newcommand{\soCorrDispMin}{5.9}")
    L.append(r"\newcommand{\soCorrDispMax}{6.9}")
    L.append("")

    out = GEN_DIR / "paper_macros.tex"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[Saved] {out} ({len(L)} lines)")


# ========================================
# Generate tab_detection_metrics_body.tex
# ========================================
def gen_table_detection(language: str = "en") -> None:
    """language = 'en' or 'cn' (only changes scenario row labels).
    Bolds the empirical best value in each column within each scenario."""
    rows = []
    for sc_short, (folder, en_lbl, cn_lbl) in SCENARIOS.items():
        sc_label = (en_lbl, cn_lbl)[1 if language == "cn" else 0]
        df = load_detection(folder, "max")
        scenario_rows = []
        for m_short, m_full in MODELS:
            row = df.loc[m_full]
            scenario_rows.append({
                "m_full": m_full,
                "auc_val": row["AUC"], "tpr_val": row["TPR@FPR5%"], "f1_val": row["Best_F1"],
                "auc": fmt_auc(row["AUC"]),
                "ci": fmt_ci(row["AUC_CI_lo"], row["AUC_CI_hi"]),
                "tpr": fmt_tpr(row["TPR@FPR5%"]),
                "f1": fmt_auc(row["Best_F1"]),
            })
        best_auc = max(range(len(scenario_rows)), key=lambda k: scenario_rows[k]["auc_val"])
        best_tpr = max(range(len(scenario_rows)), key=lambda k: scenario_rows[k]["tpr_val"])
        best_f1 = max(range(len(scenario_rows)), key=lambda k: scenario_rows[k]["f1_val"])
        for i, r in enumerate(scenario_rows):
            multirow = rf"\multirow{{3}}{{*}}{{{sc_label}}}" if i == 0 else ""
            auc = rf"\textbf{{{r['auc']}}}" if i == best_auc else r["auc"]
            tpr = rf"\textbf{{{r['tpr']}}}" if i == best_tpr else r["tpr"]
            f1 = rf"\textbf{{{r['f1']}}}" if i == best_f1 else r["f1"]
            m_label = MODEL_DISPLAY[language][r["m_full"]]
            rows.append(rf"{multirow:30s} & {m_label} & {auc} {r['ci']} & {tpr} & {f1} \\")
        if sc_short != "sd":
            rows.append(r"\addlinespace")

    if language == "en":
        header = r"Scenario & Model & AUC [95\% CI] & TPR@FPR5\% & F1 \\"
    else:
        header = r"场景 & 模型 & AUC [95\% CI] & TPR@FPR5\% & F1 \\"

    body = "\n".join([
        f"% Auto-generated by AE_gen_paper_macros.py at {datetime.now():%Y-%m-%d %H:%M:%S}",
        "% DO NOT EDIT MANUALLY.",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llccc}",
        r"\toprule",
        header,
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular*}",
    ])
    out = GEN_DIR / f"tab_detection_metrics_body_{language}.tex"
    out.write_text(body + "\n", encoding="utf-8")
    print(f"[Saved] {out}")


# ========================================
# Generate per-scenario detection tables (SO + SD)
# ========================================
def gen_table_scenario_detection(sc_short: str, language: str = "en") -> None:
    """Generate a per-scenario detection metrics table (3 models x 4 cols)."""
    folder, en_lbl, cn_lbl = SCENARIOS[sc_short]
    df = load_detection(folder, "max")

    if language == "en":
        header = r"Model & AUC [95\% CI] & TPR@FPR5\% & F1 \\"
    else:
        header = r"模型 & AUC [95\% CI] & TPR@FPR5\% & F1 \\"

    scenario_rows = []
    for m_short, m_full in MODELS:
        row = df.loc[m_full]
        scenario_rows.append({
            "m_full": m_full,
            "auc_val": row["AUC"], "tpr_val": row["TPR@FPR5%"], "f1_val": row["Best_F1"],
            "auc": fmt_auc(row["AUC"]),
            "ci": fmt_ci(row["AUC_CI_lo"], row["AUC_CI_hi"]),
            "tpr": fmt_tpr(row["TPR@FPR5%"]),
            "f1": fmt_auc(row["Best_F1"]),
        })
    best_auc = max(range(len(scenario_rows)), key=lambda k: scenario_rows[k]["auc_val"])
    best_tpr = max(range(len(scenario_rows)), key=lambda k: scenario_rows[k]["tpr_val"])
    best_f1 = max(range(len(scenario_rows)), key=lambda k: scenario_rows[k]["f1_val"])
    rows = []
    for i, r in enumerate(scenario_rows):
        auc = rf"\textbf{{{r['auc']}}}" if i == best_auc else r["auc"]
        tpr = rf"\textbf{{{r['tpr']}}}" if i == best_tpr else r["tpr"]
        f1 = rf"\textbf{{{r['f1']}}}" if i == best_f1 else r["f1"]
        m_label = MODEL_DISPLAY[language][r["m_full"]]
        rows.append(rf"{m_label} & {auc} {r['ci']} & {tpr} & {f1} \\")

    body = "\n".join([
        f"% Auto-generated by AE_gen_paper_macros.py at {datetime.now():%Y-%m-%d %H:%M:%S}",
        "% DO NOT EDIT MANUALLY.",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccc}",
        r"\toprule",
        header,
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular*}",
    ])
    out = GEN_DIR / f"tab_{sc_short}_detection_body_{language}.tex"
    out.write_text(body + "\n", encoding="utf-8")
    print(f"[Saved] {out}")


# ========================================
# Generate tab_dr_mean_vs_max_body.tex
# ========================================
def gen_table_dr_mean_vs_max(language: str = "en") -> None:
    df_mean = load_detection("Damage_Repaired", "mean")
    df_max = load_detection("Damage_Repaired", "max")
    rows = []
    for mode_label, df in [
        (("Mean-channel", "均值通道"), df_mean),
        (("Max-channel", "最大通道"), df_max),
    ]:
        label = mode_label[1 if language == "cn" else 0]
        mode_rows = []
        for m_short, m_full in MODELS:
            row = df.loc[m_full]
            mode_rows.append({
                "m_full": m_full,
                "auc_val": row["AUC"], "tpr_val": row["TPR@FPR5%"], "f1_val": row["Best_F1"],
                "auc": fmt_auc(row["AUC"]),
                "ci": fmt_ci(row["AUC_CI_lo"], row["AUC_CI_hi"]),
                "tpr": fmt_tpr(row["TPR@FPR5%"]),
                "f1": fmt_auc(row["Best_F1"]),
            })
        best_auc = max(range(len(mode_rows)), key=lambda k: mode_rows[k]["auc_val"])
        best_tpr = max(range(len(mode_rows)), key=lambda k: mode_rows[k]["tpr_val"])
        best_f1 = max(range(len(mode_rows)), key=lambda k: mode_rows[k]["f1_val"])
        for i, r in enumerate(mode_rows):
            multirow = rf"\multirow{{3}}{{*}}{{{label}}}" if i == 0 else ""
            auc = rf"\textbf{{{r['auc']}}}" if i == best_auc else r["auc"]
            tpr = rf"\textbf{{{r['tpr']}}}" if i == best_tpr else r["tpr"]
            f1 = rf"\textbf{{{r['f1']}}}" if i == best_f1 else r["f1"]
            m_label = MODEL_DISPLAY[language][r["m_full"]]
            rows.append(rf"  {multirow:25s} & {m_label} & {auc} {r['ci']} & {tpr} & {f1} \\")
        if mode_label[0] == "Mean-channel":
            rows.append(r"\midrule")

    if language == "en":
        header = r"Scoring & Model & AUC [95\% CI] & TPR@FPR5\% & F1 \\"
    else:
        header = r"评分 & 模型 & AUC [95\% CI] & TPR@FPR5\% & F1 \\"

    body = "\n".join([
        f"% Auto-generated by AE_gen_paper_macros.py at {datetime.now():%Y-%m-%d %H:%M:%S}",
        "% DO NOT EDIT MANUALLY.",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llccc}",
        r"\toprule",
        header,
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular*}",
    ])
    out = GEN_DIR / f"tab_dr_mean_vs_max_body_{language}.tex"
    out.write_text(body + "\n", encoding="utf-8")
    print(f"[Saved] {out}")


# ========================================
# Generate tab_lr_tuning_body.tex
# ========================================
def gen_table_lr_tuning(language: str = "en") -> None:
    """Generate the LR tuning table body from recovery study CSVs.

    All rows use SGD for 1000 epochs with cosine T_max=1000; the table
    isolates learning rate and regularisation as the only variables.
    """
    p1 = load_recovery_phase(1)
    p3 = load_recovery_phase(3)
    p4 = load_recovery_phase(4)
    p5 = load_recovery_phase(5)

    if language == "cn":
        labels = ["原始配置", r"\textbf{+学习率（采纳）}",
                  r"数据回放 1$\times$", r"增强 5$\times$", r"L2-SP ($\alpha$=1e-3)"]
        verdicts = ["基线", r"\textbf{采纳}", "有害", "无增益", "无增益"]
        header = r"实验 & lr & 验证损失 & 通道 123 恢复率 (\%) & 判定 \\"
    else:
        labels = ["Original config", r"\textbf{+LR (adopted)}",
                  r"Data replay 1$\times$", r"Augment 5$\times$", r"L2-SP ($\alpha$=1e-3)"]
        verdicts = ["baseline", r"\textbf{adopted}", "harmful", "no gain", "no gain"]
        header = r"Experiment & lr & Val Loss & Channel 123 recovery (\%) & Verdict \\"

    # Row data: (label, lr_str, val_loss, ch123_rec%, verdict, bold)
    rows_data = [
        (labels[0], r"$3\!\times\!10^{-4}$",
         p1.loc["1A_sgd_1000ep_cosT1000", "final_val_loss"],
         p1.loc["1A_sgd_1000ep_cosT1000", "ch123_recovery_pct"], verdicts[0], False),
        (labels[1], r"$\mathbf{1\!\times\!10^{-3}}$",
         p1.loc["1B_sgd_1000ep_lr1e3", "final_val_loss"],
         p1.loc["1B_sgd_1000ep_lr1e3", "ch123_recovery_pct"], verdicts[1], True),
        (labels[2], r"$1\!\times\!10^{-3}$",
         p3.loc["3A_sgd_lr1e3_replay1x", "final_val_loss"],
         p3.loc["3A_sgd_lr1e3_replay1x", "ch123_recovery_pct"], verdicts[2], False),
        (labels[3], r"$1\!\times\!10^{-3}$",
         p4.loc["4A_sgd_lr1e3_aug5x_s001", "final_val_loss"],
         p4.loc["4A_sgd_lr1e3_aug5x_s001", "ch123_recovery_pct"], verdicts[3], False),
        (labels[4], r"$1\!\times\!10^{-3}$",
         p5.loc["5_l2sp_a1e-03", "final_val_loss"],
         p5.loc["5_l2sp_a1e-03", "ch123_recovery_pct"], verdicts[4], False),
    ]

    tex_rows = []
    for label, lr_s, vl, rec, verd, bold in rows_data:
        vl_s = f"{vl:.5f}"
        rec_s = f"{rec:.0f}"
        if bold:
            vl_s = rf"\textbf{{{vl_s}}}"
            rec_s = rf"\textbf{{{rec_s}}}"
        tex_rows.append(f"{label} & {lr_s} & {vl_s} & {rec_s} & {verd} \\\\")

    body = "\n".join([
        f"% Auto-generated by AE_gen_paper_macros.py at {datetime.now():%Y-%m-%d %H:%M:%S}",
        "% DO NOT EDIT MANUALLY. All rows use 1000 epochs with cosine T_max=1000.",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccl}",
        r"\toprule",
        header,
        r"\midrule",
        *tex_rows,
        r"\bottomrule",
        r"\end{tabular*}",
    ])
    out = GEN_DIR / f"tab_lr_tuning_body_{language}.tex"
    out.write_text(body + "\n", encoding="utf-8")
    print(f"[Saved] {out}")


# ========================================
# Generate tab_sequential_detection_body.tex
# ========================================
def gen_table_sequential_detection(language: str = "en") -> None:
    """Sequential adaptation: max-channel AUC grouped by final cumulative state,
    with direct TL baselines under the same stage budget for comparison."""
    seq_json = AE_OUT / "Sequential_Adaptation" / "sequential_results.json"
    ana_json = AE_OUT / "Sequential_Adaptation" / "sequential_analysis.json"
    if not (seq_json.exists() and ana_json.exists()):
        print("[SKIP] sequential data not found")
        return

    with open(seq_json) as f:
        seq_data = json.load(f)
    with open(ana_json) as f:
        ana_data = json.load(f)

    EVAL_DOMAINS = ["Source", "DR", "SO", "SD"]
    FINAL_ORDER = ["DR", "SO", "SD"]
    FINAL_LABEL = {
        "en": {"DR": "CDR", "SO": "CSO", "SD": "CSD"},
        "cn": {"DR": "CDR", "SO": "CSO", "SD": "CSD"},
    }
    DISPLAY_DOMAIN = {"Source": "Source", "DR": "CDR", "SO": "CSO", "SD": "CSD"}
    DIRECT_LABEL = {"en": "Direct TL", "cn": "直接TL"}

    def display_order(ordering: str) -> str:
        return "→".join(DISPLAY_DOMAIN.get(part, part) for part in ordering.split("→"))

    # Group orderings by final domain
    by_final: dict[str, list[tuple[str, dict]]] = {d: [] for d in FINAL_ORDER}
    for key, val in seq_data.items():
        last = val["sequence"][-1]
        by_final[last].append((key, val))

    rows: list[str] = []
    for fi, final_dom in enumerate(FINAL_ORDER):
        label = FINAL_LABEL[language][final_dom]
        entries = by_final[final_dom]
        n_rows = len(entries) + 1  # +1 for direct TL baseline

        # Direct TL baseline row
        direct = ana_data["direct_tl"][final_dom]
        multirow = rf"\multirow{{{n_rows}}}{{*}}{{{label}}}"
        aucs = []
        for ed in EVAL_DOMAINS:
            v = direct["detection_max"][ed]["auc"]
            s = fmt_auc(v)
            if ed == final_dom:
                s = rf"\textbf{{{s}}}"
            aucs.append(s)
        rows.append(
            rf"{multirow} & {DIRECT_LABEL[language]} & {' & '.join(aucs)} \\"
        )
        rows.append(r"\cline{2-6}")

        # Sequential orderings
        for j, (key, val) in enumerate(entries):
            ordering_str = display_order(key)
            aucs = []
            for ed in EVAL_DOMAINS:
                v = val["detection_max"][ed]["auc"]
                s = fmt_auc(v)
                if ed == final_dom:
                    s = rf"\textbf{{{s}}}"
                aucs.append(s)
            rows.append(
                rf"{'':30s} & {ordering_str} & {' & '.join(aucs)} \\"
            )

        if fi < len(FINAL_ORDER) - 1:
            rows.append(r"\midrule")

    if language == "en":
        header = r"Final cumulative state & Ordering & Source & CDR & CSO & CSD \\"
    else:
        header = r"最后微调的累计状态 & 排列顺序 & Source & CDR & CSO & CSD \\"

    body = "\n".join([
        f"% Auto-generated by AE_gen_paper_macros.py at {datetime.now():%Y-%m-%d %H:%M:%S}",
        "% DO NOT EDIT MANUALLY.",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llcccc}",
        r"\toprule",
        header,
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular*}",
    ])
    out = GEN_DIR / f"tab_sequential_detection_body_{language}.tex"
    out.write_text(body + "\n", encoding="utf-8")
    print(f"[Saved] {out}")


# ========================================
# Generate tab_training_comparison_{scenario}_body.tex
# ========================================
def gen_table_training_comparison(language: str = "en") -> None:
    """Per-scenario 3-row table: Pre-train / TL / From-scratch final val loss."""
    pretrain_csv = AE_OUT / "Damage_Repaired" / "pretrain" / "training_losses.csv"
    pretrain_val = float(pd.read_csv(pretrain_csv).iloc[-1]["val_loss"])

    MODEL_COLS = [
        ("pretrain", {"en": "Source-domain validation reference", "cn": "源域验证损失参考"}),
        ("TL", {"en": "Transfer learning", "cn": "迁移学习"}),
        ("fromscratch", {"en": "Random initialization", "cn": "随机初始化"}),
    ]

    row_label = {"en": "Final Val Loss (MSE)", "cn": "最终验证损失 (MSE)"}

    for sc_short, (folder, _, _) in SCENARIOS.items():
        col_headers = [col[1][language] for col in MODEL_COLS]
        header = " & " + " & ".join(col_headers) + r" \\"

        vals = []
        for sub, _ in MODEL_COLS:
            if sub == "pretrain":
                vals.append(pretrain_val)
            else:
                csv = AE_OUT / folder / "tl_comparison" / sub / "training_losses.csv"
                vals.append(float(pd.read_csv(csv).iloc[-1]["val_loss"]))
        data_row = row_label[language] + " & " + " & ".join(fmt_loss(v) for v in vals) + r" \\"

        ncol = len(MODEL_COLS) + 1
        body = "\n".join([
            f"% Auto-generated by AE_gen_paper_macros.py at {datetime.now():%Y-%m-%d %H:%M:%S}",
            "% DO NOT EDIT MANUALLY.",
            r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l" + "c" * len(MODEL_COLS) + "}",
            r"\toprule",
            header,
            r"\midrule",
            data_row,
            r"\bottomrule",
            r"\end{tabular*}",
        ])
        out = GEN_DIR / f"tab_training_comparison_{sc_short}_body_{language}.tex"
        out.write_text(body + "\n", encoding="utf-8")
        print(f"[Saved] {out}")


# ========================================
# Generate tab_difficulty_ablation_{scenario}_body.tex
# ========================================
def _fmt_sci(v: float) -> str:
    """Format a float as LaTeX scientific notation with 2 significant digits.
    E.g. 0.001340 -> $1.34{\\times}10^{-3}$"""
    if v == 0:
        return "$0$"
    exp = int(np.floor(np.log10(abs(v))))
    mantissa = v / (10 ** exp)
    return rf"${mantissa:.2f}{{\times}}10^{{{exp}}}$"


# Difficulty column label per scenario
_DIFF_LABEL = {
    "dr": r"$N_r$",
    "so": r"$N_o$",
    "sd": r"$r$",
}

# How to display difficulty values per scenario (avoid floating-point artefacts)
_DIFF_FMT = {
    "dr": lambda v: str(int(v)),      # 0, 4, 8, ...
    "so": lambda v: str(int(v)),      # 0, 1, 2, ...
    "sd": lambda v: f"{v:.2f}",       # 0.00, 0.05, ...
}

DATA_SIZES = [50, 100, 200, 400]


def gen_table_difficulty_ablation(language: str = "en") -> None:
    """Per-scenario difficulty x data_size val-loss table."""
    ncol = len(DATA_SIZES) + 1  # difficulty col + data_size cols
    col_span = f"{{2-{ncol}}}"

    if language == "en":
        size_header = r"& \multicolumn{" + str(len(DATA_SIZES)) + r"}{c}{$N_{\text{tgt}}$} \\"
    else:
        size_header = r"& \multicolumn{" + str(len(DATA_SIZES)) + r"}{c}{$N_{\text{tgt}}$} \\"

    cmidrule = rf"\cmidrule(lr){col_span}"
    size_row = " & ".join(str(s) for s in DATA_SIZES) + r" \\"

    for sc_short, (folder, _, _) in SCENARIOS.items():
        df = load_difficulty(folder)
        difficulties = sorted(df["difficulty"].unique())
        diff_label = _DIFF_LABEL[sc_short]
        diff_fmt = _DIFF_FMT[sc_short]

        sub_header = f"{diff_label} & {size_row}"

        tex_rows = []
        for d in difficulties:
            sub = df[df["difficulty"] == d].set_index("data_size")
            cells = [diff_fmt(d)]
            for s in DATA_SIZES:
                cells.append(_fmt_sci(sub.loc[s, "val_loss"]))
            tex_rows.append(" & ".join(cells) + r" \\")

        body = "\n".join([
            f"% Auto-generated by AE_gen_paper_macros.py at {datetime.now():%Y-%m-%d %H:%M:%S}",
            "% DO NOT EDIT MANUALLY.",
            r"\small",
            r"\begin{tabular}{l" + "c" * len(DATA_SIZES) + "}",
            r"\toprule",
            size_header,
            cmidrule,
            sub_header,
            r"\midrule",
            *tex_rows,
            r"\bottomrule",
            r"\end{tabular}",
        ])
        out = GEN_DIR / f"tab_difficulty_ablation_{sc_short}_body_{language}.tex"
        out.write_text(body + "\n", encoding="utf-8")
        print(f"[Saved] {out}")


# ========================================
# Generate tab_channel_focus_{scenario}_body.tex
# ========================================

# Header labels per language and scenario
_CH_FOCUS_HEADER = {
    "en": {
        "dr": r"$N_r$ & Ch.\ & Pre-Shift & Post-Shift & TL@400 & Recovery (\%) \\",
        "so": r"$N_o$ & Ch.\ & Pre-Shift & Post-Shift & TL@400 & Recovery (\%) \\",
        "sd": r"$r$ & Ch.\ & Pre-Shift & Post-Shift & TL@400 & Recovery (\%) \\",
    },
    "cn": {
        "dr": r"$N_r$ & 通道 & 偏移前 & 偏移后 & TL@400 & 恢复率 (\%) \\",
        "so": r"$N_o$ & 通道 & 偏移前 & 偏移后 & TL@400 & 恢复率 (\%) \\",
        "sd": r"$r$ & 通道 & 偏移前 & 偏移后 & TL@400 & 恢复率 (\%) \\",
    },
}


def gen_table_channel_focus(language: str = "en") -> None:
    """Per-scenario table: peak-affected channel at each non-zero difficulty."""
    for sc_short, (folder, _, _) in SCENARIOS.items():
        df = load_channel_ablation(folder)
        difficulties = sorted(df["difficulty"].unique())
        diff_fmt = _DIFF_FMT[sc_short]
        header = _CH_FOCUS_HEADER[language][sc_short]

        tex_rows = []
        for d in difficulties:
            if d == 0:
                continue  # skip zero-shift control
            sub = df[df["difficulty"] == d].copy()
            sub["delta"] = sub["Post-Shift"] - sub["Pre-Shift"]
            peak = sub.nlargest(1, "delta").iloc[0]
            ch = int(peak["channel"])
            pre = peak["Pre-Shift"]
            post = peak["Post-Shift"]
            at400 = peak["400"]
            rec = (post - at400) / (post - pre) * 100 if post != pre else 0.0
            tex_rows.append(
                f"{diff_fmt(d)} & {ch} & {fmt_loss4(pre)} & {fmt_loss4(post)} "
                f"& {fmt_loss4(at400)} & {fmt_pct(rec)} \\\\"
            )

        body = "\n".join([
            f"% Auto-generated by AE_gen_paper_macros.py at {datetime.now():%Y-%m-%d %H:%M:%S}",
            "% DO NOT EDIT MANUALLY.",
            r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}cccccc}",
            r"\toprule",
            header,
            r"\midrule",
            *tex_rows,
            r"\bottomrule",
            r"\end{tabular*}",
        ])
        out = GEN_DIR / f"tab_channel_focus_{sc_short}_body_{language}.tex"
        out.write_text(body + "\n", encoding="utf-8")
        print(f"[Saved] {out}")


# ========================================
# Main
# ========================================
def main() -> None:
    print("=" * 60)
    print("AE_gen_paper_macros.py")
    print(f"  Source: {AE_OUT}")
    print(f"  Output: {GEN_DIR}")
    print("=" * 60)

    gen_macros()
    gen_table_detection("en")
    gen_table_detection("cn")
    gen_table_scenario_detection("so", "en")
    gen_table_scenario_detection("so", "cn")
    gen_table_scenario_detection("sd", "en")
    gen_table_scenario_detection("sd", "cn")
    gen_table_dr_mean_vs_max("en")
    gen_table_dr_mean_vs_max("cn")
    gen_table_lr_tuning("en")
    gen_table_lr_tuning("cn")
    gen_table_sequential_detection("en")
    gen_table_sequential_detection("cn")
    gen_table_training_comparison("en")
    gen_table_training_comparison("cn")
    gen_table_channel_focus("en")
    gen_table_channel_focus("cn")

    print()
    print("[Done] All paper macros and table bodies regenerated.")
    print("       Now compile paper_TL_en.tex / paper_TL_cn.tex.")


if __name__ == "__main__":
    main()
