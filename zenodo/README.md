# CBM-MOF Reproduction Archive

This record contains the trained inference artifacts and the compact set of processed data used for the manuscript "Process-Aware High-Throughput Discovery of Metal-Organic Frameworks for Upgrading Low-Concentration Coalbed Methane."

## Files

- `cbm_mof_models.tar.gz`: final ALIGNN inference checkpoint, model metadata and normalization statistics, and latent-space uncertainty trees.
- `cbm_mof_key_data.tar.gz`: data splits, model predictions, candidate tables, composition-sensitivity results, GCMC validation summaries, process-optimization tables, and figure source tables.
- `SHA256SUMS`: SHA-256 checksums for both archives and this README.

Paths inside each archive are relative to the CBM-MOF repository root. Public database structures, raw simulation trajectories, temporary files, logs, repeated checkpoints, and credentials are excluded.

## Model artifacts

| Path | Description |
|---|---|
| `models/inference_checkpoint.pt` | Final ALIGNN inference state. |
| `models/model_metadata.pt` | Model configuration, target normalization statistics, and training metadata. |
| `models/uncertainty_trees.pkl` | Fitted latent-space nearest-neighbor trees used for LSV diagnostics. |

The model predicts six mixture-uptake tasks and two heat-of-adsorption tasks. Physical-space uptake is reported in mol/kg and heat of adsorption in kJ/mol. The model-input `id_prop.csv` files use the per-task transformations defined by `data/alignn/transform_config.json`; the `deployment/*_groundtruth.csv` and `deployment/*_predictions.csv` files are in physical units.

## Key data groups

| Group | Main contents |
|---|---|
| Data splits and evaluation | `data/alignn/{train,val,test}/id_prop.csv`, target-transform metadata, and physical-space train/validation/test labels and predictions under `deployment/`. |
| Library screening | Full-library predictions before and after stability/metal filtering, screened candidate unions, and screening-funnel statistics. |
| Composition sensitivity | Per-candidate 20:80 versus 50:50 results and rank-change summaries. |
| Molecular simulation | Explicit mixed-gas GCMC validation tables, adsorption-heat uncertainties, pure-component isotherm inputs, and fitted isotherm parameters. |
| Process evaluation | PSA/VSA candidate lists, Pareto evaluations, material ranking, and selected knee points. |
| Figure source tables | Figure 2 UMAP coordinates and sample splits, model/UQ summary tables, Cluster-8 summaries, and compact validated-candidate tables for Figures 7 and 8. |

## Data dictionary and units

| Field pattern | Meaning | Unit or encoding |
|---|---|---|
| `AdsCH4_*`, `AdsN2_*` | Predicted or GCMC mixture uptake at the stated pressure | mol/kg in physical-space tables |
| `QstCH4`, `QstN2` | Heat of adsorption | kJ/mol |
| `*_gcmc_error` | Block uncertainty reported by the GCMC calculation | Same unit as the associated value |
| `PSA_WC_*`, `VSA_WC_*` | Working capacity between the screening pressures | mol/kg |
| `PSA_alpha_*`, `VSA_alpha_*` | Equilibrium selectivity | Dimensionless |
| `PSA_API_*`, `VSA_API_*` | Adsorbent performance indicator | Dimensionless |
| `*_lsv_norm`, `lsv_norm_composite` | Normalized latent-space variance diagnostic | Dimensionless |
| `PLD_A` | Pore-limiting diameter, derived from Zeo++ field `Df` | Angstrom |
| `purity`, `recovery` | Process product purity and methane recovery | Fraction |
| `productivity_mol_kg_h` | Process productivity | mol/(kg h) |
| `energy_kWh_ton` | Specific process energy | kWh/tonne CH4 |
| `Pressure[bar]`, `Temperature[K]` | Isotherm state point | bar, K |

Boolean fields are encoded as `True`/`False` or `0`/`1` according to the source table. Column names containing `2080` and `5050` refer to CH4:N2 feed compositions of 20:80 and 50:50, respectively. `release_data/alignn_test_metrics.json` and `release_data/uq_calibration.json` are path-sanitized copies of the test-set and UQ summaries. `release_data/figure7_validated_candidate_clusters.csv` and `release_data/figure8_pld_qst_source.csv` are compact derived tables generated during packaging from the archived validation set and the corresponding source descriptors.

## Minimum checks

```bash
sha256sum -c SHA256SUMS
tar -tzf cbm_mof_models.tar.gz
tar -tzf cbm_mof_key_data.tar.gz
```

To inspect the checkpoint metadata without running inference:

```bash
python -c "import torch; x=torch.load('models/model_metadata.pt', map_location='cpu', weights_only=False); print(sorted(x)); print(sorted(x['norm_stats']))"
```

After extracting the key-data archive into a clone of the public CBM-MOF code repository, the revised composition-sensitivity figure can be regenerated with:

```bash
python src/figures/fig_rank_change_histogram.py --output-dir reproduced_figures
```

The full training and inference workflow is documented in the public CBM-MOF code repository. Full model inference requires MOF structure files from the public source databases and the software environment described below. RASPA and process-model outputs in this archive are processed summaries; raw trajectories are intentionally omitted to keep the record compact.

## Inference environment

The archived ALIGNN inference path was configured with Python 3.10, NumPy 1.26.4, PyTorch 2.1.0 with CUDA 12.1, DGL 2.2.1 with CUDA 12.1, ALIGNN 2025.4.1, and JARVIS-Tools 2026.1.10. NumPy should remain below version 2 for this ALIGNN/JARVIS combination because its graph-construction utilities rely on the NumPy 1.x array interface. CPU installations may use matching CPU builds of PyTorch and DGL.
