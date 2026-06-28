# Code availability / 代码可用性说明

This repository contains the core code package for the AE-TL ship structural damage detection study.

本仓库为 AE-TL 船舶结构损伤检测研究的核心代码包。

## Contents / 内容

- `script/`: data generation, preprocessing, AE training, transfer-learning comparison, ablation, evaluation, plotting, and configuration scripts.
- `script/TL_settings.jsonc`: main experiment configuration.
- `script/TL_settings_doc.md`: configuration field notes.
- `requirements.txt`: Python package requirements for the non-Abaqus analysis scripts.

Manuscript files, generated LaTeX tables, Chinese review-package materials, PDFs, and review-stage result CSV files are intentionally not included here.

本文手稿、生成的 LaTeX 表格、中文审阅包材料、PDF 和审阅阶段结果 CSV 均不纳入本仓库。

## Data boundary / 数据边界

The full finite-element model files, raw simulation outputs, trained checkpoints, generated figures, and large intermediate arrays are not included because of file size and sharing constraints. They can be requested from the corresponding author under reasonable research-use conditions.

由于体量和共享条件限制，完整有限元模型文件、原始仿真输出、训练检查点、生成图件和大型中间数组未纳入本仓库；可在合理研究用途条件下向通讯作者请求获取。

## Environment / 运行环境

Most analysis scripts use standard Python scientific packages listed in `requirements.txt`. Abaqus-dependent scripts require a local Abaqus Python environment and the original finite-element model files.

大多数分析脚本使用 `requirements.txt` 中列出的 Python 科学计算包。依赖 Abaqus 的脚本需要本机 Abaqus Python 环境及原始有限元模型文件。

## Reproduction note / 复现说明

The scripts keep the same relative layout as the working repository. When the requested finite-element data, preprocessed arrays, and checkpoints are placed under the expected output folders, the code can regenerate the study outputs from the configured workflow.

脚本保留工作仓库中的相对路径约定。将按请求获取的有限元数据、预处理数组和检查点放回对应输出目录后，可按配置流程重新生成研究输出。
