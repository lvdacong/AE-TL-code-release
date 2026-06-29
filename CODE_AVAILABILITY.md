# Code Availability

This repository contains the core code package for the AE-TL ship structural damage detection study.

## Included

- `script/`: data generation hooks, preprocessing, AE training, transfer-learning comparison, ablation, evaluation, plotting, table-generation, and configuration scripts.
- `script/review_checks/`: review-stage verification scripts for the independent relocation test, non-neural baselines, and noise-sensitivity CSV outputs used in the revision tables and figures.
- `script/TL_settings.jsonc`: main experiment configuration.
- `script/TL_settings_doc.md`: configuration field notes.
- `requirements.txt`: Python package requirements for the non-Abaqus analysis scripts.
- `README.md`: command map, exact synthesis-output commands, and known exclusions.

## Not Included

The following materials are intentionally excluded because of file size, sharing constraints, or manuscript-review boundary:

- full finite-element model files;
- raw simulation outputs;
- trained checkpoints;
- generated figures;
- generated LaTeX tables;
- manuscript and review materials;
- review-stage result CSV files;
- large intermediate arrays.

These excluded materials can be requested from the corresponding author under reasonable research-use conditions where sharing restrictions permit.

## Reproduction Note

The scripts keep the same relative layout as the working repository. When the requested finite-element data, preprocessed arrays, checkpoints, and review CSV outputs are placed under the expected folders, the code can regenerate the reported study outputs from the configured workflow. README.md identifies the commands and expected outputs for the independent relocation test and Tables 15-17 / Figures 26-28.

Abaqus-dependent scripts require a licensed local Abaqus installation and the original finite-element model files. The remaining analysis scripts use the Python packages listed in `requirements.txt`.
