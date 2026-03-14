# ALIGNN Mainline Pipeline

This document defines the maintained ALIGNN screening pipeline in `CBM-MOF`.
It exists to keep the canonical entrypoints, result directories, and legacy
boundaries explicit after the `src/alignn` refactor.

## Scope

The maintained mainline covers four phases:

1. model preparation and deployment artifacts
2. uncertainty calibration and library screening
3. top-candidate validation by GCMC/Widom
4. process-level validation by pure-component fitting and breakthrough

The `src/alignn` root now keeps only canonical Python entrypoints. Historical
experiments and abandoned scripts are preserved under `src/alignn/legacy/`.
Shell and SLURM wrappers that still serve the maintained pipeline live under
`src/alignn/scripts/`.

## Canonical Directory Layout

For one model directory, the maintained layout is:

```text
results/alignn/model_epXXX/
├── deployment/
├── uq/
├── full_library_inference/
├── top_candidates/
└── bkt_candidates/
```

The shared path helper is:

- `src/alignn/common/paths.py`

The canonical target definitions and screening constants live in:

- `src/alignn/common/constants.py`

## Phase 1. Deployment and UQ

### 1. Extract split embeddings

- Entry point: `src/alignn/extract_split_embeddings.py`
- Wrapper: `src/alignn/scripts/run_extract_embeddings.sh`
- Inputs:
  - checkpoint weights
  - meta checkpoint
  - processed ALIGNN dataset
- Outputs:
  - `deployment/{train,val,test}_latent_features.npz`
  - `deployment/{train,val,test}_predictions.csv`
  - `deployment/{train,val,test}_groundtruth.csv`

### 2. Calibrate uncertainty

- Entry point: `src/alignn/calibrate_uq.py`
- Optional wrapper: `src/alignn/scripts/run_uq.sh`
- Internal modules:
  - `src/alignn/uq/core.py`
  - `src/alignn/uq/io.py`
  - `src/alignn/uq/calibration.py`
  - `src/alignn/uq/plots.py`
  - `src/alignn/uq/consistency.py`
- Authoritative outputs:
  - `uq/uncertainty_trees.pkl`
  - `uq/uq_calibration.json`
  - `uq/k_sensitivity_sweep.json`
  - `uq/lsv_thresholds.json`
  - `uq/latent_space_pca_by_targets.png`

`calibrate_uq.py` is the only maintained producer of `lsv_thresholds.json`.
Downstream code must not hard-code independent UQ thresholds.
The current refactor default is `--recommended-pct 85`. Any future change to
the formal UQ cutoff should be made in `calibrate_uq.py` and then propagated by
rerunning the downstream screening pipeline.

### 3. Full-library inference

- Entry point: `src/alignn/full_library_inference.py`
- Wrapper: `src/alignn/scripts/run_full_library_inference.sh`
- Outputs:
  - `full_library_inference/batches/*.npz`
  - `full_library_inference/batches/*_predictions.csv`

### 4. Apply UQ to the inferred library

- Entry point: `src/alignn/apply_uq_to_library.py`
- Internal module: `src/alignn/uq/apply.py`
- Outputs:
  - `full_library_inference/full_library_uq.csv`

Consistency must be checked against `uq/lsv_thresholds.json` via:

- `src/alignn/uq/consistency.py`
- `tests/test_alignn_pipeline_consistency.py`

## Phase 2. Screening

### 1. Compute API metrics

- Entry point: `src/alignn/compute_api_metrics.py`
- Internal module: `src/alignn/screening/metrics.py`
- Outputs:
  - `full_library_inference/full_library_with_api.csv`

This step computes metrics only. It does not apply the UQ threshold.

### 2. Screen the library

- Entry point: `src/alignn/screen_library.py`
- Outputs:
  - `full_library_inference/full_library_screened.csv`

This step applies:

1. the current canonical UQ cutoff from `lsv_thresholds.json`
2. the methane uptake floor

### 3. Stability filter and top-candidate selection

- Entry points:
  - `src/alignn/filter_stable_candidates.py`
  - `src/alignn/select_top_candidates.py`
- Outputs:
  - filtered candidate CSVs
  - top-candidate CIF collection

## Phase 3. Validation

### 1. Top-candidate GCMC / Widom validation

- Entry points:
  - `src/alignn/submit_gcmc_validation.py`
  - `src/alignn/parse_validation_results.py`
  - `src/alignn/select_final_top10.py`

### 2. Pure-component and process validation

- Entry points:
  - `src/alignn/submit_pure_component_gcmc.py`
  - `src/alignn/parse_pure_component_results.py`
  - `src/alignn/fit_pure_component_isotherms.py`
  - `src/alignn/compute_iast_selectivity.py`
  - `src/alignn/run_breakthrough.py`
- Shared process helper:
  - `src/alignn/process/curve_cache.py`

## Plotting Boundary

Publication figures are not maintained in `src/alignn`.
They belong to `src/figures`, which consumes the canonical CSV/JSON outputs from
the mainline pipeline.

Examples:

- `src/figures/fig_model_comparison.py`
- `src/figures/fig_uq_validation.py`
- `src/figures/fig_database_analysis.py`
- `src/figures/fig_process_validation.py`

## Wrappers vs Legacy

Maintained wrappers live in:

- `src/alignn/scripts/`

Historical experiments, abandoned exploratory scripts, and retired wrappers live in:

- `src/alignn/legacy/`

Nothing in `src/alignn/legacy/` should be treated as a source of truth for the
current screening pipeline.

## Minimal Maintained Pipeline

The maintained end-to-end sequence is:

1. `extract_split_embeddings.py`
2. `calibrate_uq.py`
3. `full_library_inference.py`
4. `apply_uq_to_library.py`
5. `compute_api_metrics.py`
6. `screen_library.py`
7. `filter_stable_candidates.py`
8. `select_top_candidates.py`
9. `submit_gcmc_validation.py`
10. `parse_validation_results.py`
11. `select_final_top10.py`
12. `submit_pure_component_gcmc.py`
13. `parse_pure_component_results.py`
14. `fit_pure_component_isotherms.py`
15. `compute_iast_selectivity.py`
16. `run_breakthrough.py`

The corresponding consistency smoke checks are currently covered by:

- `tests/test_alignn_pipeline_consistency.py`
- `tests/test_gcmc_dependency_paths.py`
- `tests/test_parse_atc_cu_pure_component.py`
- `tests/test_bkt_dependency_paths.py`
- `tests/test_bkt_output_layout.py`
