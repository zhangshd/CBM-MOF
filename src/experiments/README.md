# src/experiments — CBM-MOF Screening Pipeline

End-to-end Python scripts for screening Metal-Organic Frameworks (MOFs) for
low-concentration coalbed methane (CBM) upgrading (CH₄/N₂ = 20:80).
Each script corresponds to one workflow stage and can be run independently or in sequence.

## Setup

```bash
conda activate mofmthnn
cd /home/zhangsd/repos/CBM-MOF
```

## Pipeline Overview

```
exp01 → exp02 → exp03a → exp03b → exp04 → exp05
     → exp06 → exp06b
     → exp07 → exp08 → exp09
```

## Scripts

| Script | Stage | Purpose | Key Inputs | Key Outputs |
|--------|-------|---------|-----------|-------------|
| `exp01_integrate_cifs.py` | Data prep | Integrate CIFs from ARC-MOF, CoREMOF2024, MOSAEC-DB | External archives | `data/processed/integrated_cifs/`, `file_code_map.json` |
| `exp02_textural_screening.py` | Screening | Merge textural features; apply PLD > 3 Å, GSA > 100 m²/g filter | MOF-HTS feature batch dirs | `data/processed/textural_screened/textural_screened_list.txt`, 2 figures |
| `exp03a_clustering_analysis.py` | Clustering | K-Means (k=22), UMAP dimensionality reduction, stratified sampling (20k/1k/1k) | Screened CSV | Split CSVs, CIF symlinks, 1 figure |
| `exp03b_gcmc_batch_submission.py` | GCMC | RASPA3 GCMC + Widom insertion for ATC-Cu benchmark; validation plot | ATC-Cu CIFs | SLURM jobs, 1 validation figure |
| `exp04_prepare_graph_grid_data.py` | Feature prep | Batch SLURM jobs for MOFTransformer/CGCNN graph+grid preparation | Split CIFs | `src/moftransformer/data/round2/graphs_grids/` |
| `exp05_make_training_data.py` | Training data | Merge GCMC results → per-task train/val/test CSVs | RASPA3 outputs | Training CSVs, 2 figures |
| `exp06_training.py` | Training | Submit CGCNN / MOFTransformer training jobs to SLURM | Training CSVs | SLURM training jobs |
| `exp06b_training_results.py` | Evaluation | Aggregate R², MAE, MAPE across models; parity and bar charts | Model prediction files | Comparison xlsx, 2 figures |
| `exp07_inference_ml.py` | Inference | Full-library ML inference + PSA/VSA API score calculation | ML prediction CSV | Enhanced predictions CSV, violin + parity figures |
| `exp08_screening_ml.py` | Selection | Screen top-100 PSA/VSA candidates, integrate GCMC validation results, best-per-cluster selection | Enhanced CSV, GCMC results | Filtered CIFs, final selection CSV, 5 figures |
| `exp09_top_mofs_isotherm.py` | Characterisation | Pure-component isotherm GCMC, Langmuir fitting, breakthrough simulation | Final CIFs | GCMC results, fit parameters, density figures |

## Test Mode

Every script accepts `--test` to route **all** outputs to `results/test_run/` and
replace SLURM submissions with `[DRY-RUN]` echo messages. Production files are
never touched.

```bash
python src/experiments/exp08_screening_ml.py --test
```

`results/test_run/` is git-ignored.

## Shared Utilities (`utils.py`)

| Symbol | Description |
|--------|-------------|
| `REPO_ROOT` | Absolute path to repository root |
| `NATURE_COLORS` | 8-colour Nature-journal palette |
| `add_test_arg(parser)` | Attach `--test` flag to an `ArgumentParser` |
| `resolve_output_dir(test, subdir)` | Route to `results/` or `results/test_run/` |
| `resolve_data_dir(test, subdir)` | Route to `data/` or `results/test_run/data/` |
| `sbatch_submit(script, test, cwd)` | Real `sbatch` or dry-run echo |
| `setup_matplotlib()` | Headless Agg backend + Nature figure rcParams |
| `apply_nature_axes(ax)` | Remove top/right spines, add subtle grid |
| `savefig(fig, path, close)` | Save PNG at 300 dpi |
