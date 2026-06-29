# AE-TL code release / AE-TL 代码发布包

This repository contains the core code package for the AE-TL ship structural damage detection study.

本仓库为 AE-TL 船舶结构损伤检测研究的核心代码发布包。

## Contents / 内容

- `script/`: finite-element data generation hooks, preprocessing, AE training, transfer-learning comparisons, ablations, evaluation, plotting, and configuration scripts.
- `script/TL_settings.jsonc`: main experiment configuration.
- `script/TL_settings_doc.md`: configuration field notes.
- `requirements.txt`: Python package requirements for non-Abaqus analysis scripts.
- `CODE_AVAILABILITY.md`: scope, data boundary, and reproduction notes.

## Boundary / 边界

Manuscript files, generated LaTeX tables, Chinese review-package materials, PDFs, review-stage result CSV files, full finite-element model files, raw simulation outputs, trained checkpoints, generated figures, and large intermediate arrays are intentionally not included because of size and sharing constraints.

本文手稿、生成的 LaTeX 表格、中文审阅包材料、PDF、审阅阶段结果 CSV、完整有限元模型文件、原始仿真输出、训练检查点、生成图件和大型中间数组因体量与共享条件限制不纳入本仓库。

## Reproduction / 复现

Most analysis scripts use the packages listed in `requirements.txt`. Abaqus-dependent scripts require a local Abaqus Python environment and the original finite-element model files. After the requested finite-element data, preprocessed arrays, and checkpoints are placed under the expected output folders, the configured workflow can regenerate the study outputs.

大多数分析脚本使用 `requirements.txt` 中列出的 Python 依赖；依赖 Abaqus 的脚本需要本机 Abaqus Python 环境及原始有限元模型文件。将按请求获取的有限元数据、预处理数组和检查点放回对应输出目录后，可按配置流程重新生成研究输出。

## Command Map For Review / 审阅复现命令索引

Run commands from the repository root unless the command explicitly changes into `script/`. Abaqus commands require the original FE model files and a licensed local Abaqus installation; the other commands use the Python packages in `requirements.txt`.

除特别说明进入 `script/` 目录的命令外，以下命令均从仓库根目录执行。Abaqus 命令需要原始有限元模型文件和本机授权 Abaqus；其他命令使用 `requirements.txt` 中的 Python 依赖。

```powershell
python -m pip install -r requirements.txt

# Finite-element model and simulation hooks
cd script
python A0_generate_repair_regions.py
abaqus cae noGUI=AA_create_damage_cae.py
abaqus cae noGUI=AB_generate_simulation_data.py
python AB_generate_simulation_data_postprocess_auxiliary.py

# Sensor layout conversion and preprocessing
python AC_convert_and_extract.py
python AD_preprocess_datasets.py

# Core AE training and scenario evaluation
python AE_run_pretrain.py
python AE_run_dr_tl_comparison.py
python AE_run_so_sd_tl_comparison.py
python AE_run_tl_comparison_standard.py
python AE_run_baseline_validation.py
python AE_run_optimizer_study.py
python AE_run_so_difflr_ablation.py
python AE_run_sequential_adaptation.py
python AE_run_sequential_analysis.py

# Figure and summary regeneration
python AE_gen_roc_and_summary.py
python AE_gen_scenario_figures.py
python AE_gen_revision_comparison_figures.py
python AE_gen_lr_tuning_figure.py
python AE_gen_difflr_comparison.py
python AE_gen_difflr_roc.py
python AE_gen_baseline_3d_figures.py
```

## Configuration And Random Seeds / 配置与随机种子

The main configuration is `script/TL_settings.jsonc`; field notes are in `script/TL_settings_doc.md`. Random seeds are fixed inside the experiment scripts and helper modules, primarily with seed `42` for model training/evaluation and the documented region seed `72048` for repair-region generation.

主配置文件为 `script/TL_settings.jsonc`，字段说明见 `script/TL_settings_doc.md`。随机种子固定在实验脚本和辅助模块中：模型训练与评估主要使用 `42`，修补区域生成使用文档化的区域种子 `72048`。
