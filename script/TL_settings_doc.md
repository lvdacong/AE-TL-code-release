# Core configuration notes

`TL_settings.jsonc` is the public configuration used by the finite-element construction, simulation, conversion, and preprocessing stages. It is JSON with comments.

## Sections

- `region_by_elem_ID`: fixed element groups for the local section-loss state and nested Local Structural Stiffening patches.
- `FEM_models`: healthy, stiffened, damaged, and baseline finite-element variants.
- `simulation_counts`: requested sample counts for the finite-element runs.
- `cases`: the Local Structural Stiffening and Sensor Relocation preprocessing configurations.

The Temperature-conditioned Sensor Response is not represented by the former multiplicative-drift configuration. Its frozen measurement model is implemented in `AE_temperature_sensor_drift.py`, with the sensitivity grid executed by `AE_run_sd_difficulty_ablation.py`.

## External paths

The finite-element scripts retain generic defaults under:

```text
C:/SHM_abaqus_models
C:/SHM_abaqus_data
```

Change these defaults for your system, or import the relevant stage and call `run_from_config` with your own configuration. No personal workstation path is required.

## Generated artifacts

Data arrays, checkpoints, metrics, figures, logs, and other generated artifacts are written to ignored output directories and are not part of the public repository.
