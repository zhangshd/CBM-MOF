"""
quick_ood_eval.py
=================
Phase 0: Quick OOD evaluation of ALIGNN checkpoints against GCMC ground truth.

For each checkpoint, runs inference on the 199 GCMC-validated MOFs,
then computes per-property per-group (PSA/VSA) R² metrics (16 cells total).

Two modes:
  1) Evaluate: run inference + compute metrics, save per-checkpoint JSON
  2) Merge: aggregate all per-checkpoint JSONs into summary CSV + markdown

Usage (evaluate, single checkpoint):
    python src/alignn/quick_ood_eval.py eval \
        --checkpoint results/alignn/500ep_symlog_1e-3_ddp2g/checkpoint_epoch0090.pt

Usage (merge all results):
    python src/alignn/quick_ood_eval.py merge \
        --output-dir results/alignn/model_selection

SLURM (array mode):
    sbatch src/alignn/run_batch_ood_eval.sh
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

# Suppress tqdm from ALIGNN internals in SLURM
if not sys.stdout.isatty() or os.environ.get("SLURM_JOB_ID"):
    os.environ["TQDM_DISABLE"] = "1"

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore", category=UserWarning)

# ── Constants ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path("/home/zhangsd/repos/CBM-MOF")

UPTAKE_COLS = ["AdsCH4_10kPa", "AdsCH4_100kPa", "AdsCH4_1000kPa",
               "AdsN2_10kPa",  "AdsN2_100kPa",  "AdsN2_1000kPa"]
QST_COLS    = ["QstCH4", "QstN2"]
TARGET_COLS = UPTAKE_COLS + QST_COLS
N_TARGETS   = len(TARGET_COLS)

# GCMC ground truth column names (must match gcmc_vs_ml_comparison.csv)
GCMC_COL_MAP = {
    "AdsCH4_10kPa":   "gcmc_AdsCH4_10kPa",
    "AdsCH4_100kPa":  "gcmc_AdsCH4_100kPa",
    "AdsCH4_1000kPa": "gcmc_AdsCH4_1000kPa",
    "AdsN2_10kPa":    "gcmc_AdsN2_10kPa",
    "AdsN2_100kPa":   "gcmc_AdsN2_100kPa",
    "AdsN2_1000kPa":  "gcmc_AdsN2_1000kPa",
    "QstCH4":         "QstCH4_gcmc",
    "QstN2":          "QstN2_gcmc",
}


def compute_per_group_r2(
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    group_col: str,
) -> dict:
    """
    Compute R² for each TARGET_COL within a group (PSA or VSA).

    Args:
        pred_df: DataFrame with mof_id + 8 predicted target columns
        gt_df:   DataFrame with mof_id + psa_rank + vsa_rank + GCMC columns
        group_col: 'psa_rank' or 'vsa_rank'

    Returns:
        dict of {property: R²} for MOFs in this group
    """
    mask = gt_df[group_col].notna()
    group_ids = gt_df.loc[mask, "mof_id"].tolist()

    pred_sub = pred_df[pred_df["mof_id"].isin(group_ids)].set_index("mof_id")
    gt_sub = gt_df[gt_df["mof_id"].isin(group_ids)].set_index("mof_id")

    common_ids = pred_sub.index.intersection(gt_sub.index)
    pred_sub = pred_sub.loc[common_ids]
    gt_sub = gt_sub.loc[common_ids]

    results = {}
    for prop in TARGET_COLS:
        gcmc_col = GCMC_COL_MAP[prop]
        y_true = gt_sub[gcmc_col].values
        y_pred = pred_sub[prop].values
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        if valid.sum() < 3:
            results[prop] = float("nan")
        else:
            results[prop] = float(r2_score(y_true[valid], y_pred[valid]))
    return results


def evaluate_checkpoint(
    ckpt_path: Path,
    meta_ckpt_path: Path,
    xform_cfg: dict,
    mol_ids: list,
    cif_dir: Path,
    gt_df: pd.DataFrame,
    max_atoms: int,
    batch_size: int,
    graph_cache_path: Path = None,
) -> dict:
    """
    Run inference for one checkpoint on the 199 GCMC MOFs and compute metrics.
    """
    import torch
    import torch.nn as nn

    # Import from full_library_inference (same package)
    from full_library_inference import (
        load_model,
        find_embedding_layer,
        invert_targets,
        build_batch_graphs,
        run_inference_with_embeddings,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract epoch from checkpoint filename
    ckpt_name = ckpt_path.stem
    if ckpt_name == "best_model":
        # best_model.pt is the ep276 checkpoint
        epoch = 276
    else:
        epoch_str = ckpt_name.replace("checkpoint_epoch", "").lstrip("0") or "0"
        epoch = int(epoch_str)

    print(f"\n{'=' * 65}")
    print(f"Evaluating checkpoint: {ckpt_name} (epoch {epoch})")
    print(f"  Device: {device}")
    print(f"  Time  : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 65}")

    # Load model
    model, norm_mean, norm_std, cfg = load_model(ckpt_path, meta_ckpt_path, device)
    hook_layer, _ = find_embedding_layer(model)

    # Build graphs (reuses cache if available — graphs are model-independent)
    valid_ids, graphs, records, failed_ids = build_batch_graphs(
        mol_ids, cif_dir, max_atoms, graph_cache_path,
    )

    if not valid_ids:
        print("  ERROR: No valid MOFs — skipping this checkpoint.")
        return None

    print(f"  Valid MOFs: {len(valid_ids)}, Failed: {len(failed_ids)}")

    # Build dataset and run inference
    from alignn.graphs import StructureDataset
    df_records = pd.DataFrame(records)
    dataset = StructureDataset(
        df_records,
        graphs,
        target="target",
        atom_features="cgcnn",
        line_graph=True,
        id_tag="jid",
    )

    preds_xform, _ = run_inference_with_embeddings(
        model, dataset, norm_mean, norm_std, hook_layer, device,
        batch_size, len(valid_ids),
    )

    # Invert transforms → physical space
    preds_orig = invert_targets(preds_xform, xform_cfg)

    # Build prediction DataFrame
    pred_df = pd.DataFrame(preds_orig, columns=TARGET_COLS)
    pred_df.insert(0, "mof_id", valid_ids)

    # Compute per-group R²
    psa_r2 = compute_per_group_r2(pred_df, gt_df, "psa_rank")
    vsa_r2 = compute_per_group_r2(pred_df, gt_df, "vsa_rank")

    # Assemble 16 cells
    cells = []
    for prop in TARGET_COLS:
        cells.append({"group": "PSA", "property": prop, "r2": psa_r2[prop]})
        cells.append({"group": "VSA", "property": prop, "r2": vsa_r2[prop]})

    all_r2 = [c["r2"] for c in cells if np.isfinite(c["r2"])]
    min_r2 = min(all_r2) if all_r2 else float("nan")
    mean_r2 = float(np.mean(all_r2)) if all_r2 else float("nan")

    weakest = min(cells, key=lambda c: c["r2"] if np.isfinite(c["r2"]) else float("inf"))
    weakest_str = f"{weakest['group']}_{weakest['property']}"

    # Global (all 199) R² per property
    global_r2 = {}
    pred_aligned = pred_df.set_index("mof_id")
    gt_aligned = gt_df.set_index("mof_id")
    common = pred_aligned.index.intersection(gt_aligned.index)
    for prop in TARGET_COLS:
        gcmc_col = GCMC_COL_MAP[prop]
        y_true = gt_aligned.loc[common, gcmc_col].values
        y_pred = pred_aligned.loc[common, prop].values
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        if valid.sum() >= 3:
            global_r2[prop] = float(r2_score(y_true[valid], y_pred[valid]))
        else:
            global_r2[prop] = float("nan")

    result = {
        "epoch": epoch,
        "checkpoint": ckpt_name,
        "min_r2": min_r2,
        "mean_r2": mean_r2,
        "weakest_cell": weakest_str,
        "per_cell": cells,
        "global_r2": global_r2,
        "n_valid": len(valid_ids),
        "n_failed": len(failed_ids),
    }

    print(f"\n  Results for epoch {epoch}:")
    print(f"    min(R²)  = {min_r2:.4f}  ({weakest_str})")
    print(f"    mean(R²) = {mean_r2:.4f}")
    print(f"    Global mean R² = {np.mean(list(global_r2.values())):.4f}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return result


def cmd_eval(args):
    """Evaluate a single checkpoint and save result as JSON."""
    import torch

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint)
    meta_ckpt_path = Path(args.meta_checkpoint)
    cif_dir = Path(args.cif_dir)

    print("=" * 65)
    print("quick_ood_eval.py — Phase 0 OOD Screening (eval mode)")
    print(f"  Checkpoint    : {ckpt_path.name}")
    print(f"  GCMC CSV      : {args.gcmc_csv}")
    print(f"  Output dir    : {output_dir}")
    print(f"  Start time    : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # Load GCMC ground truth
    gt_df = pd.read_csv(args.gcmc_csv)
    mol_ids = gt_df["mof_id"].tolist()
    print(f"\nLoaded GCMC ground truth: {len(mol_ids)} MOFs")

    # Load transform config
    with open(args.xform_config) as f:
        xform_cfg = json.load(f)

    # Graph cache (shared across checkpoints — graphs are model-independent)
    cache_path = None if args.no_cache else (output_dir / "ood_graph_cache.pkl")

    result = evaluate_checkpoint(
        ckpt_path=ckpt_path,
        meta_ckpt_path=meta_ckpt_path,
        xform_cfg=xform_cfg,
        mol_ids=mol_ids,
        cif_dir=cif_dir,
        gt_df=gt_df,
        max_atoms=args.max_atoms,
        batch_size=args.batch_size,
        graph_cache_path=cache_path,
    )

    if result is not None:
        # Save per-checkpoint JSON
        json_path = output_dir / f"ood_ep{result['epoch']:04d}.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved: {json_path}")
    else:
        print("\nERROR: No result produced.")
        sys.exit(1)


def cmd_build_cache(args):
    """Pre-build graph cache for the 199 GCMC MOFs (run once before SLURM array)."""
    from full_library_inference import build_batch_graphs

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "ood_graph_cache.pkl"

    if cache_path.exists():
        print(f"Graph cache already exists: {cache_path}")
        return

    gt_df = pd.read_csv(args.gcmc_csv)
    mol_ids = gt_df["mof_id"].tolist()
    cif_dir = Path(args.cif_dir)

    print(f"Building graph cache for {len(mol_ids)} MOFs...")
    valid_ids, graphs, records, failed_ids = build_batch_graphs(
        mol_ids, cif_dir, args.max_atoms, cache_path,
    )
    print(f"Done: {len(valid_ids)} valid, {len(failed_ids)} failed")
    print(f"Cache saved: {cache_path}")


def cmd_merge(args):
    """Merge all per-checkpoint JSON files into summary CSV + markdown."""
    output_dir = Path(args.output_dir)

    # Load all JSON result files
    json_files = sorted(output_dir.glob("ood_ep*.json"))
    if not json_files:
        print(f"ERROR: No ood_ep*.json files found in {output_dir}")
        sys.exit(1)

    results = []
    for jf in json_files:
        with open(jf) as f:
            results.append(json.load(f))

    print(f"Loaded {len(results)} checkpoint results")

    # ── Detailed CSV: one row per checkpoint × property × group ──────────────
    rows = []
    for r in results:
        for cell in r["per_cell"]:
            rows.append({
                "epoch": r["epoch"],
                "checkpoint": r["checkpoint"],
                "group": cell["group"],
                "property": cell["property"],
                "r2": cell["r2"],
            })
    detail_df = pd.DataFrame(rows)
    detail_path = output_dir / "ood_screening_results.csv"
    detail_df.to_csv(detail_path, index=False)
    print(f"Saved: {detail_path}")

    # ── Summary CSV: one row per checkpoint ──────────────────────────────────
    summary_rows = []
    for r in results:
        row = {
            "epoch": r["epoch"],
            "min_r2": r["min_r2"],
            "mean_r2": r["mean_r2"],
            "weakest_cell": r["weakest_cell"],
            "n_valid": r["n_valid"],
        }
        for prop, val in r["global_r2"].items():
            row[f"global_{prop}_r2"] = val
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("min_r2", ascending=False)
    summary_path = output_dir / "ood_screening_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    # ── Summary markdown ─────────────────────────────────────────────────────
    md_lines = [
        "# OOD Screening Summary — Phase 0 Model Selection",
        "",
        f"**Date**: {time.strftime('%Y-%m-%d')}",
        f"**Ground truth**: 199 GCMC-validated MOFs (8 properties × PSA/VSA = 16 cells)",
        f"**Metric**: min(R² across 16 cells) — higher is better",
        f"**Checkpoints evaluated**: {len(results)}",
        "",
        "## Ranking (sorted by min R²)",
        "",
        "| Rank | Epoch | min(R²) | mean(R²) | Weakest Cell | Global mean R² |",
        "|------|-------|---------|----------|--------------|----------------|",
    ]

    for i, (_, row) in enumerate(summary_df.iterrows(), 1):
        global_cols = [c for c in summary_df.columns if c.startswith("global_")]
        global_mean = np.mean([row[c] for c in global_cols])
        md_lines.append(
            f"| {i} | ep{int(row['epoch']):03d} | {row['min_r2']:.4f} | "
            f"{row['mean_r2']:.4f} | {row['weakest_cell']} | {global_mean:.4f} |"
        )

    md_lines.extend([
        "",
        "## Per-Cell R² Detail (sorted by min R², R² < 0.6 in bold)",
        "",
    ])

    # Build header row
    header = "| Epoch |"
    sep = "|-------|"
    for group in ["PSA", "VSA"]:
        for prop in TARGET_COLS:
            short = prop.replace("Ads", "").replace("_", " ")
            header += f" {group[:1]}-{short} |"
            sep += "--------|"
    md_lines.append(header)
    md_lines.append(sep)

    for r in sorted(results, key=lambda x: -x["min_r2"]):
        row_str = f"| ep{r['epoch']:03d} |"
        for cell in r["per_cell"]:
            v = cell["r2"]
            if np.isfinite(v):
                if v < 0.6:
                    row_str += f" **{v:.3f}** |"
                else:
                    row_str += f" {v:.3f} |"
            else:
                row_str += " NaN |"
        md_lines.append(row_str)

    md_lines.extend([
        "",
        "> **Bold** values indicate R² < 0.6 (weak cells).",
        "",
        "## Baseline Reference",
        "",
        "ep100 (current deployment): min(R²) = -0.262 (VSA AdsCH4_100kPa)",
        "",
        "## Top-4 Candidates for Full Pipeline",
        "",
    ])

    # Extract top 4
    top4 = summary_df.head(4)
    for i, (_, row) in enumerate(top4.iterrows(), 1):
        md_lines.append(f"{i}. **ep{int(row['epoch']):03d}**: min(R²) = {row['min_r2']:.4f}, "
                        f"mean(R²) = {row['mean_r2']:.4f}, weakest = {row['weakest_cell']}")

    md_lines.append("")

    md_path = output_dir / "ood_screening_summary.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Saved: {md_path}")

    # Print top results
    print(f"\n{'=' * 65}")
    print("TOP 4 CANDIDATES:")
    for i, (_, row) in enumerate(top4.iterrows(), 1):
        print(f"  {i}. ep{int(row['epoch']):03d}  min(R²)={row['min_r2']:.4f}  "
              f"mean(R²)={row['mean_r2']:.4f}  weakest={row['weakest_cell']}")
    print(f"{'=' * 65}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── eval subcommand ──────────────────────────────────────────────────────
    p_eval = subparsers.add_parser("eval", help="Evaluate a single checkpoint")
    p_eval.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint path to evaluate")
    p_eval.add_argument("--meta-checkpoint", type=str,
                        default=str(REPO_ROOT / "results/alignn/500ep_symlog_1e-3_ddp2g/best_model.pt"))
    p_eval.add_argument("--gcmc-csv", type=str,
                        default=str(REPO_ROOT / "results/alignn/gcmc_top_candidates/gcmc_vs_ml_comparison.csv"))
    p_eval.add_argument("--cif-dir", type=str,
                        default=str(REPO_ROOT / "results/cbm_screening/all_graphs_grids"))
    p_eval.add_argument("--xform-config", type=str,
                        default=str(REPO_ROOT / "data/alignn_symlog_1e-3/transform_config.json"))
    p_eval.add_argument("--output-dir", type=str,
                        default=str(REPO_ROOT / "results/alignn/model_selection"))
    p_eval.add_argument("--batch-size", type=int, default=8)
    p_eval.add_argument("--max-atoms", type=int, default=500)
    p_eval.add_argument("--no-cache", action="store_true")

    # ── build-cache subcommand ───────────────────────────────────────────────
    p_cache = subparsers.add_parser("build-cache",
                                     help="Pre-build graph cache (run before SLURM array)")
    p_cache.add_argument("--gcmc-csv", type=str,
                         default=str(REPO_ROOT / "results/alignn/gcmc_top_candidates/gcmc_vs_ml_comparison.csv"))
    p_cache.add_argument("--cif-dir", type=str,
                         default=str(REPO_ROOT / "results/cbm_screening/all_graphs_grids"))
    p_cache.add_argument("--output-dir", type=str,
                         default=str(REPO_ROOT / "results/alignn/model_selection"))
    p_cache.add_argument("--max-atoms", type=int, default=500)

    # ── merge subcommand ─────────────────────────────────────────────────────
    p_merge = subparsers.add_parser("merge", help="Merge all per-checkpoint JSONs")
    p_merge.add_argument("--output-dir", type=str,
                         default=str(REPO_ROOT / "results/alignn/model_selection"))

    args = parser.parse_args()

    if args.command == "eval":
        cmd_eval(args)
    elif args.command == "build-cache":
        cmd_build_cache(args)
    elif args.command == "merge":
        cmd_merge(args)


if __name__ == "__main__":
    main()
