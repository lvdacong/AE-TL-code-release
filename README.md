# AE-TL Core Code Release

This repository contains the core implementation used by the final revision of the AE-TL ship structural damage detection study. It is a code-only release: raw data, finite-element models, checkpoints, generated figures, result tables, and review materials are not included.

## Scope

The retained code covers:

- finite-element scenario construction and simulation hooks;
- stress extraction and dataset preprocessing;
- source autoencoder pre-training and healthy-only target recalibration;
- Local Structural Stiffening, Sensor Relocation, and Temperature-conditioned Sensor Response;
- shift-severity and target-data-budget sensitivity;
- transfer-method, representation, and target-budget comparisons;
- composite nonlinear and sequential multi-domain evaluation;
- the principal evaluation and plotting routines.

Reviewer-only checks, exploratory branches, interactive viewers, duplicate plotting utilities, generated outputs, and compact result snapshots have been removed from this public package.

## Environment

Install the non-Abaqus dependencies with:

```powershell
python -m pip install -r requirements.txt
```

The finite-element scripts require a licensed Abaqus installation and the original model files. The default `C:/SHM_abaqus_models` and `C:/SHM_abaqus_data` locations are generic placeholders from the research workflow; change them for your environment before running the data-generation stage.

## Configuration

- `script/TL_settings.jsonc`: finite-element model definitions, Local Structural Stiffening cases, Sensor Relocation cases, and dataset sizes.
- `script/offset_targets.json`: fixed sensor-relocation mapping used by the final manuscript.
- `script/camera_position.json`: optional camera configuration for structural rendering.
- `script/AE_temperature_sensor_drift.py`: frozen Temperature-conditioned Sensor Response model.
- `script/TL_settings_doc.md`: concise configuration and data-boundary notes.

## Core workflow

Run commands from `script/` unless noted otherwise.

### 1. Finite-element data and preprocessing

```powershell
python A0_generate_repair_regions.py
abaqus cae noGUI=AA_create_damage_cae.py
abaqus cae noGUI=AB_generate_simulation_data.py
python AC_convert_and_extract.py
python AD_preprocess_datasets.py
```

The three Abaqus wrapper/helper files used by this stage are retained because they are part of the execution path.

### 2. Source model and three target shifts

```powershell
python AE_run_pretrain.py
python AE_run_baseline_validation.py
python AE_run_dr_tl_comparison.py
python AE_run_so_sd_tl_comparison.py
```

`AE_run_so_sd_tl_comparison.py` runs the Sensor Relocation and Temperature-conditioned Sensor Response comparisons. The temperature-conditioned branch can also be run directly with `AE_run_temperature_sd_tl_comparison.py`.

### 3. Core sensitivity and comparison experiments

```powershell
python AE_run_dr_difficulty_ablation.py
python AE_run_so_difficulty_ablation.py
python AE_run_sd_difficulty_ablation.py
python AE_run_channel_ablation_all.py
python AE_run_transfer_method_ablation.py
python AE_run_representation_dimension_ablation.py
python AE_run_target_sample_budget_ablation.py
```

### 4. Composite and sequential evaluation

```powershell
python AE_run_composite_csd_comparison.py
python AE_gen_sequential_cumulative_data.py
python AE_run_sequential_adaptation.py
python AE_run_sequential_analysis.py
```

### 5. Principal synthesis routines

```powershell
python AD_gen_sensor_drift_figure.py
python AE_gen_roc_and_summary.py
python AE_gen_comparison_figures.py
```

These scripts write generated artifacts under ignored output directories. No generated result is tracked in this repository.

## Reproduction boundary

The scripts retain the working repository's relative data layout. Reproduction therefore requires the corresponding finite-element model, simulation data, preprocessed arrays, and source checkpoint to be placed in the expected external/output locations. This repository documents and exposes the computational workflow but does not redistribute restricted or large research artifacts.
