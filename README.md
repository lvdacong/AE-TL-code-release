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
