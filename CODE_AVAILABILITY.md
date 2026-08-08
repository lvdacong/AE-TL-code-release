# Code Availability

This repository is the code-only public package for the final AE-TL manuscript revision.

## Included

- core finite-element scenario and simulation hooks;
- stress conversion and preprocessing code;
- autoencoder pre-training, target recalibration, and evaluation code;
- the three final target-shift implementations;
- the main sensitivity, transfer-method, representation, composite-state, and sequential-adaptation experiments;
- principal plotting routines and minimal configuration files.

## Not included

- finite-element model files or licensed software;
- raw simulation data or preprocessed arrays;
- model checkpoints;
- generated CSV, JSON, image, PDF, or LaTeX results;
- manuscript, response-letter, or reviewer materials;
- review-only verification scripts and result snapshots;
- exploratory ablations, interactive tools, and duplicate plotting branches.

The exclusions prevent redistribution of large or restricted research artifacts and keep the public repository limited to the computational core. Required research inputs may be requested from the corresponding author where sharing restrictions permit.

## Reproduction note

The code preserves the relative directory conventions of the research workflow. Users must provide the external finite-element inputs and regenerate intermediate datasets and checkpoints before running the downstream experiments. The exact command groups are listed in `README.md`.

No credentials, access tokens, private repository addresses, personal machine paths, or generated research results are intentionally included in this release.
