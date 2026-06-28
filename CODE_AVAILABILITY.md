# Code availability / 代码可用性说明

This repository contains the code package supporting the AE-TL ship structural damage detection revision package.

本仓库为 AE-TL 船舶结构损伤检测修订稿配套代码包。

## Contents / 内容

- `script/`: data generation, preprocessing, AE training, transfer-learning comparison, ablation, plotting, and table-generation scripts.
- `用于查看的中文版/复核脚本/`: review-stage verification scripts for non-neural baselines, noise sensitivity, and the independent sensor-relocation test.
- `用于查看的中文版/*.csv`: small verification result tables used in the Chinese review package.
- `用于查看的中文版/独立偏移测试raw/run_config.json`: configuration evidence for the independent sensor-relocation test.
- `paper_TL/generated/`: generated LaTeX table bodies and numeric macros.

## Data boundary / 数据边界

The full finite-element model files, raw simulation outputs, trained checkpoints, and large intermediate arrays are not included because of file size and sharing constraints. They can be requested from the corresponding author under reasonable research-use conditions.

由于体量和共享条件限制，完整有限元模型文件、原始仿真输出、训练检查点和大型中间数组未纳入本仓库；可在合理研究用途条件下向通讯作者请求获取。

## Environment / 运行环境

Most analysis scripts use standard Python scientific packages listed in `requirements.txt`. Abaqus-dependent scripts require a local Abaqus Python environment and the original finite-element model files.

大多数分析脚本使用 `requirements.txt` 中列出的 Python 科学计算包。依赖 Abaqus 的脚本需要本机 Abaqus Python 环境及原始有限元模型文件。

## Reproduction note / 复现说明

The review verification scripts keep the same relative layout as the working repository. When the requested data arrays and checkpoints are placed under the expected output folders, the scripts can regenerate the review CSV files.

复核脚本保留工作仓库中的相对路径约定。将按请求获取的数据数组和检查点放回对应输出目录后，可重新生成审阅包中的 CSV 结果。
