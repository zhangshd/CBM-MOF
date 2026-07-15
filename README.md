# CBM-MOF

CBM-MOF is a research workflow repository for screening metal-organic
frameworks for coalbed methane upgrading. The repository combines structure
data preparation, graph neural network prediction, uncertainty-aware candidate
screening, molecular simulation validation, process-simulation interfaces, and
analysis figure generation.

The code is organized as a script-driven workflow. Most commands are intended
to be run from the repository root, and result directories are part of the
workflow contract.

## Functional Modules

```text
src/
├── alignn/                  # ALIGNN training, inference, screening, UQ, validation
│   ├── common/              # Shared paths and target definitions
│   ├── process/             # SuperPSA parameter conversion and result parsing
│   ├── screening/           # Ranking and separation metric calculations
│   ├── scripts/             # Shell and SLURM wrappers for maintained runs
│   ├── training/            # Training helpers
│   ├── uq/                  # Latent-space uncertainty calibration and application
│   └── validation/          # Validation analysis utilities
├── analysis/                # Standalone analysis utilities
├── cgcnn/                   # CGCNN model implementation
├── data/                    # Data transformation helpers
├── figures/                 # Figure and table generation scripts
├── gcmc/                    # RASPATOOLS submodule for GCMC and Widom workflows
├── jupyter/                 # Local notebooks and scratch analysis, ignored by git
├── ml/                      # Classical ML models and SHAP utilities
├── moftransformer/          # MOFTransformer implementation
├── SuperPSA/                # SuperPSA submodule for process simulations
└── mof_structure_visualizer.py
```

Repository-level directories:

```text
configs/                    # RASPA, Widom, force-field, and ML config files
data/                       # Raw, processed, and model-ready datasets
docs/                       # Pipeline notes and workflow documentation
logs/                       # Local run logs
results/                    # Model outputs, screening results, validation data, figures
slurm_logs/                 # Cluster job stdout/stderr logs
```

## Submodules

Initialize active submodules after cloning:

```bash
git submodule update --init --recursive src/gcmc src/SuperPSA
```

Submodule responsibilities:

- `src/gcmc`: RASPA-side utilities for mixture GCMC, pure-component GCMC, and
  Widom insertion workflows.
- `src/SuperPSA`: pressure-swing adsorption process simulation and optimization.

The parent repository pins each submodule to an exact gitlink. For `src/gcmc`,
the `cbm` branch is used only by `git submodule update --remote`; normal
initialization continues to check out the committed gitlink.

## ALIGNN Workflow

The `src/alignn` module is the primary screening workflow. It supports:

- model training and evaluation
- split embedding extraction
- latent-space uncertainty calibration
- full-library inference
- API-style separation metric calculation
- candidate filtering and ranking
- GCMC/Widom validation job preparation
- pure-component isotherm parsing and fitting
- IAST selectivity calculation
- SuperPSA parameter and result handling

Typical maintained entry points:

```bash
python src/alignn/train_alignn.py --help
python src/alignn/evaluate_alignn.py --help
python src/alignn/extract_split_embeddings.py --help
python src/alignn/calibrate_uq.py --help
python src/alignn/full_library_inference.py --help
python src/alignn/apply_uq_to_library.py --help
python src/alignn/compute_api_metrics.py --help
python src/alignn/screen_library.py --help
python src/alignn/filter_stable_candidates.py --help
python src/alignn/select_exp_top_candidates.py --help
```

Validation and process-oriented entry points:

```bash
python src/alignn/submit_gcmc_validation.py --help
python src/alignn/run_new_top10_pipeline.py --help
python src/alignn/visualize_gcmc_validation.py --help
python src/alignn/submit_pure_component_gcmc.py --help
python src/alignn/parse_pure_component_results.py --help
python src/alignn/fit_pure_component_isotherms.py --help
python src/alignn/fit_extended_dsl.py --help
python src/alignn/compute_iast_selectivity.py --help
python src/alignn/compute_iast_multicomp.py --help
```

Process helper entry points:

```bash
python src/alignn/process/convert_params_for_superpsa.py --help
python src/alignn/process/generate_process_config.py --help
python src/alignn/process/parse_nsga2_results.py --help
python src/alignn/process/select_knee_points.py --help
python src/alignn/process/select_optimization_candidates.py --help
```

For one model directory, the active ALIGNN result layout is:

```text
results/alignn/model_epXXX/
├── deployment/
├── uq/
├── full_library_inference/
├── top_candidates/
└── process_candidates/
```

Shared path and target definitions live in:

- `src/alignn/common/paths.py`
- `src/alignn/common/constants.py`

## Molecular Simulation Workflow

GCMC and Widom workflows are driven through `src/gcmc` and the configuration
files under `configs/`.

Common configuration files:

- `configs/custom_force_field.json`
- `configs/custom_simulation.json`
- `configs/custom_simulation_analysis.json`
- `configs/custom_widom_component.json`
- `configs/custom_widom_simulation.json`

Typical ALIGNN-side simulation orchestration:

```bash
python src/alignn/submit_gcmc_validation.py --test
python src/alignn/submit_pure_component_gcmc.py --help
python src/alignn/parse_pure_component_results.py --help
```

## Process Simulation Workflow

SuperPSA-facing utilities live under `src/alignn/process`. They convert fitted
adsorption parameters into process-simulation inputs and parse optimization
outputs back into CSV tables for ranking and plotting.

Key files:

- `src/alignn/process/convert_params_for_superpsa.py`
- `src/alignn/process/generate_process_config.py`
- `src/alignn/process/parse_nsga2_results.py`
- `src/alignn/process/select_knee_points.py`
- `src/alignn/process/select_optimization_candidates.py`

## Figure And Table Generation

Figure scripts live under `src/figures` and consume existing CSV, JSON, and
model-output files from `results/`.

Common entry points:

```bash
python src/figures/generate_all.py
python src/figures/fig_model_comparison.py
python src/figures/fig_uq_validation.py
python src/figures/fig_database_analysis.py
python src/figures/fig_top100_validation.py
python src/figures/fig_process_pairplot.py
python src/figures/fig_psa_pareto.py
python src/figures/generate_table1.py
```

Shared figure utilities:

- `src/figures/style.py`
- `src/figures/data_loader.py`
- `src/figures/annotation_layout.py`

## Data And Results

Data directories:

- `data/raw`: source datasets
- `data/processed`: processed CIFs, descriptors, and merged tables
- `data/cbm_mof`: project-specific CBM-MOF data assets
- `data/alignn*`: ALIGNN-ready datasets and transformed variants

Result directories:

- `results/alignn`: ALIGNN predictions, UQ outputs, screening results,
  validation data, and process-candidate files
- `results/figures`: generated figure assets
- `results/summary`: compact JSON summaries
- `results/ml_models*`: classical ML outputs
- `results/cgcnn_models*`: CGCNN outputs
- `results/moftransformer_models*`: MOFTransformer outputs

## Environment Notes

This repository does not define a single universal environment file. The active
workflow uses a Python scientific stack plus model- and simulator-specific
packages.

Expected Python packages include:

- numpy, pandas, scipy
- matplotlib, seaborn
- scikit-learn
- torch
- ALIGNN and JARVIS-related dependencies for graph neural network workflows
- simulator-specific dependencies for RASPA/GCMC and SuperPSA workflows

Use each script's `--help` output as the command-line source of truth.

## Operating Conventions

- Run scripts from the repository root unless the script documents otherwise.
- Keep generated notebooks and scratch exploration under `src/jupyter/`.
- Use `src/alignn/common/paths.py` for model-directory path handling.
- Use `src/alignn/common/constants.py` for target names and screening constants.
- Store generated outputs under `results/` rather than under `src/`.
