# CBM-MOF R1 Reproduction Archive

This record contains the trained inference artifacts and the compact set of processed data used for the manuscript "Process-Aware High-Throughput Discovery of Metal-Organic Frameworks for Upgrading Low-Concentration Coalbed Methane."

## Files

- `cbm_mof_models_r1.tar.gz`: epoch-150 ALIGNN checkpoint, training metadata and normalization checkpoint, and latent-space uncertainty trees.
- `cbm_mof_key_data_r1.tar.gz`: data splits, full-library predictions, candidate tables, composition-sensitivity results, explicit GCMC/Widom summaries, process-optimization tables, and figure source tables.
- `SHA256SUMS`: SHA-256 checksums for both archives and this README.

Paths inside each archive are relative to the CBM-MOF repository root. Public database structures, raw simulation trajectories, temporary files, logs, repeated checkpoints, and credentials are excluded.

## Model artifacts

| Path | Description |
|---|---|
| `results/alignn/500ep_symlog_1e-3_ddp2g/checkpoint_epoch0150.pt` | ALIGNN epoch-150 inference state. |
| `results/alignn/500ep_symlog_1e-3_ddp2g/best_model.pt` | Model configuration, target normalization statistics, and training metadata. |
| `results/alignn/model_ep150/uq/uncertainty_trees.pkl` | Fitted latent-space nearest-neighbor trees used for LSV diagnostics. |

The adsorption targets are mixture uptake in mol/kg and heat of adsorption in kJ/mol. The target order and transforms are defined by `data/alignn/targets.txt` and `data/alignn/transform_config.json` in the key-data archive.

## Key data groups

| Group | Main contents |
|---|---|
| Data splits | `data/alignn/{train,val,test}/id_prop.csv` and target-transform metadata. |
| Library screening | `full_library_with_api.csv`, screened candidate unions, and screening-funnel statistics. |
| Composition sensitivity | Per-candidate 20:80 versus 50:50 results and rank-change summaries. |
| Molecular simulation | Explicit mixed-gas GCMC and Widom comparison tables, isotherm inputs, and fitted isotherm parameters. |
| Process evaluation | PSA/VSA candidate lists, Pareto evaluations, material ranking, and selected knee points. |
| Figure source tables | Cluster-8 structural summaries and the tabular inputs used for revised figures. |

## Minimum checks

```bash
sha256sum -c SHA256SUMS
tar -tzf cbm_mof_models_r1.tar.gz
tar -tzf cbm_mof_key_data_r1.tar.gz
```

To inspect the checkpoint metadata without running inference:

```bash
python -c "import torch; x=torch.load('results/alignn/500ep_symlog_1e-3_ddp2g/best_model.pt', map_location='cpu', weights_only=False); print(sorted(x)); print(sorted(x['norm_stats']))"
```

Full model inference requires the software environment documented in the public CBM-MOF code repository. RASPA and process-model outputs in this archive are processed summaries; raw trajectories are intentionally omitted to keep the record compact.

## Inference environment

The archived ALIGNN inference path was configured with Python 3.10, NumPy 1.26.4, PyTorch 2.1.0 with CUDA 12.1, DGL 2.2.1 with CUDA 12.1, ALIGNN 2025.4.1, and JARVIS-Tools 2026.1.10. NumPy should remain below version 2 for this ALIGNN/JARVIS combination because its graph-construction utilities rely on the NumPy 1.x array interface. CPU installations may use matching CPU builds of PyTorch and DGL.
