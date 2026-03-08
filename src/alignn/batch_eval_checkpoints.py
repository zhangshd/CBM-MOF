"""
batch_eval_checkpoints.py
=========================
Evaluate all epoch checkpoints on validation and test sets to track
performance trends and inform early-stopping / checkpoint selection.

Key optimization: graph data is built ONCE; only model weights are swapped per
checkpoint. This makes it ~20x faster than running evaluate_alignn.py repeatedly.

Outputs:
  <output_dir>/checkpoint_trends.csv   — per-checkpoint R² / MAE table
  <output_dir>/checkpoint_trends.json  — full metrics (all 8 targets × val/test)
  <output_dir>/checkpoint_trends.png   — R² trend plot (val + test)

Usage:
    CUDA_VISIBLE_DEVICES=1 python src/alignn/batch_eval_checkpoints.py \
        --ckpt-dir  results/alignn/500ep_symlog_1e-3_ddp2g \
        --output-dir results/alignn/500ep_symlog_1e-3_ddp2g/checkpoint_trends \
        --data-dir  data/alignn_symlog_1e-3 \
        --max-atoms 500 \
        --with-test          # add to also evaluate test set (slow, ~30s/ckpt)
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import dgl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=UserWarning)

# ── Reuse helpers from evaluate_alignn ────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from evaluate_alignn import (
    REPO_ROOT, INFER_DIR, TARGET_COLS, UPTAKE_COLS, QST_COLS, UNITS, N_TARGETS,
    load_transform_config, invert_prediction,
    build_dataset_from_records, collate_fn, run_inference,
    load_test_labels, compute_metrics_orig,
)

# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def list_epoch_checkpoints(ckpt_dir: Path) -> list[tuple[int, Path]]:
    """Return sorted list of (epoch, path) for checkpoint_epochXXXX.pt files."""
    ckpts = sorted(ckpt_dir.glob("checkpoint_epoch*.pt"))
    result = []
    for p in ckpts:
        stem = p.stem  # e.g. "checkpoint_epoch0010"
        try:
            epoch = int(stem.replace("checkpoint_epoch", ""))
            result.append((epoch, p))
        except ValueError:
            pass
    return result


def load_model_from_ckpt(ckpt_path: Path, device: torch.device,
                          model=None,
                          meta_ckpt: dict = None):
    """Load (or reload) ALIGNN model from checkpoint.

    Epoch checkpoints (checkpoint_epochXXXX.pt) only contain model_state +
    optimizer. They lack 'config' and 'norm_stats'. Pass `meta_ckpt` (loaded
    from best_model.pt) to supply those fields when evaluating epoch ckpts.

    If `model` is None, instantiates a new model from config.
    If `model` is provided, only replaces the state_dict (fast path).

    Returns: (model, norm_mean, norm_std, epoch, val_mae)
    """
    from alignn.models.alignn import ALIGNN, ALIGNNConfig

    ckpt = torch.load(ckpt_path, map_location=device)

    # Config / norm_stats may come from a reference 'meta_ckpt' (best_model.pt)
    # when epoch checkpoints don't store them.
    ref = ckpt if "config" in ckpt else (meta_ckpt or {})
    cfg = ref.get("config", {})
    norm_stats = ref.get("norm_stats", {})
    norm_mean  = np.array(norm_stats.get("mean", [0.0] * N_TARGETS), dtype=np.float32)
    norm_std   = np.array(norm_stats.get("std",  [1.0] * N_TARGETS), dtype=np.float32)

    if model is None:
        alignn_cfg = ALIGNNConfig(
            name="alignn",
            atom_input_features=92,
            edge_input_features=cfg.get("edge_input_features", 80),
            triplet_input_features=cfg.get("triplet_input_features", 40),
            embedding_features=cfg.get("embedding_features", 64),
            hidden_features=cfg.get("hidden_features", 256),
            output_features=N_TARGETS,
            gcn_layers=cfg.get("n_layers", 4),
            alignn_layers=cfg.get("alignn_layers", 4),
            link=cfg.get("link", "identity"),
        )
        model = ALIGNN(alignn_cfg).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    epoch   = ckpt.get("epoch", -1)
    val_mae = ckpt.get("val_mae", float("nan"))
    return model, norm_mean, norm_std, epoch, val_mae


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loading (called once)
# ──────────────────────────────────────────────────────────────────────────────

def _load_split_dataset(data_dir: Path, split: str, max_atoms: int,
                         xform_cfg: dict):
    """Build graph dataset for val or test split.

    Returns (dataset, mol_ids, true_arr).
    """
    from jarvis.core.atoms import Atoms

    id_prop_csv = data_dir / split / "id_prop.csv"
    # CIF files live in shared cifs/ directory, not per-split dirs
    cif_dir     = data_dir / "cifs"
    print(f"\n  Building {split} dataset from {id_prop_csv}...")

    labels_df = load_test_labels(id_prop_csv, xform_cfg)   # works for val too
    records   = []
    for mol_id in labels_df.index:
        cif_path = cif_dir / f"{mol_id}.cif"
        if not cif_path.exists():
            continue
        try:
            atoms = Atoms.from_cif(str(cif_path))
            if max_atoms and len(atoms.elements) > max_atoms:
                continue
            records.append({
                "jid":    mol_id,
                "atoms":  atoms.to_dict(),
                "target": [0.0] * N_TARGETS,
            })
        except Exception:
            pass

    print(f"  Building {len(records)} {split} graphs...")
    dataset, mol_ids = build_dataset_from_records(records, max_atoms=max_atoms)
    true_arr = labels_df.loc[mol_ids, TARGET_COLS].values.astype(np.float32)
    return dataset, mol_ids, true_arr


# ──────────────────────────────────────────────────────────────────────────────
# Reporting helpers
# ──────────────────────────────────────────────────────────────────────────────

def _mean_r2(metrics: dict) -> float:
    r2s = [v["R2"] for v in metrics.values() if v.get("R2") is not None]
    return float(np.mean(r2s)) if r2s else float("nan")


def print_trend_table(rows: list[dict], splits: list[str]) -> None:
    """Print a pretty ASCII trend table."""
    header = f"{'epoch':>6}  " + \
             "  ".join(f"{s+' R²':>12}" for s in splits)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for row in rows:
        line = f"{row['epoch']:>6}  " + \
               "  ".join(f"{row.get(s+'_mean_r2', float('nan')):>12.4f}" for s in splits)
        print(line)
    print("=" * len(header))


def plot_trends(rows: list[dict], splits: list[str], output_path: Path,
                focus_target: str = "AdsCH4_1000kPa") -> None:
    """Plot mean R² trends + per-target focus (default: AdsCH4_1000kPa)."""
    epochs = [r["epoch"] for r in rows]
    n_rows = 2
    n_cols = len(splits)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 8),
                             sharex=True)
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    colors = {"val": "#FF9800", "test": "#2196F3"}

    for col_i, split in enumerate(splits):
        c = colors.get(split, "#9E9E9E")

        # Row 0: mean R²
        ax0 = axes[0, col_i]
        mean_r2_vals = [r.get(f"{split}_mean_r2", float("nan")) for r in rows]
        ax0.plot(epochs, mean_r2_vals, "o-", color=c, lw=2, ms=5)
        ax0.set_title(f"{split}\nmean R²", fontsize=10)
        ax0.set_ylabel("mean R²")
        ax0.grid(True, alpha=0.3)
        valid_vals = [v for v in mean_r2_vals if not np.isnan(v)]
        if valid_vals:
            ax0.set_ylim([max(0, min(valid_vals) - 0.05), 1.0])

        # Row 1: focus target R²
        ax1 = axes[1, col_i]
        focus_r2_vals = [
            r.get("all_metrics", {}).get(split, {}).get(focus_target, {}).get("R2", float("nan"))
            for r in rows
        ]
        ax1.plot(epochs, focus_r2_vals, "s--", color=c, lw=2, ms=5, alpha=0.8)
        ax1.set_title(f"{focus_target}\nR²", fontsize=10)
        ax1.set_ylabel("R²")
        ax1.set_xlabel("Epoch")
        ax1.grid(True, alpha=0.3)

    plt.suptitle(f"Checkpoint Trend — ALIGNN symlog τ=1e-3 (500ep)\nFocus: {focus_target}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nTrend plot saved: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch-evaluate all epoch checkpoints to track R² trends"
    )
    parser.add_argument("--ckpt-dir",    type=str,
                        default="results/alignn/500ep_symlog_1e-3_ddp2g",
                        help="Directory containing checkpoint_epochXXXX.pt files")
    parser.add_argument("--output-dir",  type=str,
                        default="results/alignn/500ep_symlog_1e-3_ddp2g/checkpoint_trends",
                        help="Output directory for trend files")
    parser.add_argument("--data-dir",    type=str, default=None,
                        help="Data directory (for transform_config.json and val/test splits)")
    parser.add_argument("--max-atoms",   type=int, default=500)
    parser.add_argument("--batch-size",  type=int, default=32)
    parser.add_argument("--with-test",   action="store_true",
                        help="Also evaluate test set (~30s per checkpoint; slow)")
    parser.add_argument("--focus-target", type=str, default="AdsCH4_1000kPa",
                        help="Target to highlight in the trend plot")
    args = parser.parse_args()

    ckpt_dir   = Path(args.ckpt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    REPO_ROOT_LOCAL = Path("/home/zhangsd/repos/CBM-MOF")
    data_dir = Path(args.data_dir) if args.data_dir else REPO_ROOT_LOCAL / "data/alignn"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint dir: {ckpt_dir}")
    print(f"Data dir: {data_dir}")

    # ── 1. List checkpoints ─────────────────────────────────
    epoch_ckpts = list_epoch_checkpoints(ckpt_dir)
    print(f"\nFound {len(epoch_ckpts)} epoch checkpoints: "
          f"ep{epoch_ckpts[0][0]} → ep{epoch_ckpts[-1][0]}")

    # Load best_model.pt as meta source for config + norm_stats
    # (epoch checkpoints lack these fields; best_model.pt always has them)
    best_ckpt_path = ckpt_dir / "best_model.pt"
    if not best_ckpt_path.exists():
        raise FileNotFoundError(f"best_model.pt not found in {ckpt_dir}")
    meta_ckpt = torch.load(best_ckpt_path, map_location="cpu")
    print(f"Meta (config+norm_stats) from: {best_ckpt_path}")
    print(f"  norm_mean: {meta_ckpt['norm_stats']['mean']}")

    # ── 2. Load transform config ────────────────────────────
    xform_cfg = load_transform_config(data_dir / "transform_config.json")
    print(f"Transform config loaded from {data_dir}/transform_config.json")

    # ── 3. Build datasets ONCE ──────────────────────────────
    print("\n" + "=" * 60)
    print("Building graph datasets (done ONCE for all checkpoints)...")
    print("=" * 60)

    splits_data = {}  # split_name -> (dataset, mol_ids, true_arr)

    # Always evaluate validation set
    val_dataset, val_ids, val_true = _load_split_dataset(
        data_dir, "val", args.max_atoms, xform_cfg
    )
    splits_data["val"] = (val_dataset, val_ids, val_true)

    if args.with_test:
        tst_dataset, tst_ids, tst_true = _load_split_dataset(
            data_dir, "test", args.max_atoms, xform_cfg
        )
        splits_data["test"] = (tst_dataset, tst_ids, tst_true)

    active_splits = list(splits_data.keys())
    print(f"\nActive splits: {active_splits}")
    print(f"Graph build complete. Starting checkpoint evaluation...\n")

    # ── 4. Iterate checkpoints ──────────────────────────────
    model = None
    all_rows = []
    full_records = {}  # epoch -> full metrics dict

    for i, (epoch, ckpt_path) in enumerate(epoch_ckpts):
        print(f"[{i+1:2d}/{len(epoch_ckpts)}] ep{epoch:04d} — {ckpt_path.name}", end="  ")

        model, norm_mean, norm_std, actual_epoch, val_mae = \
            load_model_from_ckpt(ckpt_path, device, model=model, meta_ckpt=meta_ckpt)

        row = {"epoch": epoch, "all_metrics": {}}
        epoch_metrics = {}

        for split_name, (dataset, mol_ids, true_arr) in splits_data.items():
            preds = run_inference(model, dataset, norm_mean, norm_std,
                                  device, batch_size=args.batch_size,
                                  xform_cfg=xform_cfg)
            metrics = compute_metrics_orig(preds, true_arr)
            epoch_metrics[split_name] = metrics
            row[f"{split_name}_mean_r2"] = _mean_r2(metrics)

            # Print focus target inline
            focus_r2 = metrics.get(args.focus_target, {}).get("R2", float("nan"))
            if not np.isnan(focus_r2):
                print(f"{split_name} {args.focus_target}={focus_r2:.4f}", end="  ", flush=True)

        row["all_metrics"] = epoch_metrics
        all_rows.append(row)
        full_records[epoch] = epoch_metrics

        # Print mean R² summary
        summary = "  |  ".join(
            f"{s}: {row.get(s+'_mean_r2', float('nan')):.4f}" for s in active_splits
        )
        print(f"\n         mean R²: {summary}")

    # ── 5. Output ────────────────────────────────────────────
    print_trend_table(all_rows, active_splits)

    # Save CSV
    csv_rows = []
    for row in all_rows:
        r = {"epoch": row["epoch"]}
        for s in active_splits:
            r[f"{s}_mean_r2"] = row.get(f"{s}_mean_r2", float("nan"))
            for col in TARGET_COLS:
                r[f"{s}_{col}_R2"] = \
                    row["all_metrics"].get(s, {}).get(col, {}).get("R2", float("nan"))
                r[f"{s}_{col}_MAE"] = \
                    row["all_metrics"].get(s, {}).get(col, {}).get("MAE", float("nan"))
        csv_rows.append(r)

    csv_path = output_dir / "checkpoint_trends.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False, float_format="%.5f")
    print(f"\nTrend CSV saved: {csv_path}")

    # Save JSON
    json_path = output_dir / "checkpoint_trends.json"
    with open(json_path, "w") as f:
        json.dump(full_records, f, indent=2)
    print(f"Trend JSON saved: {json_path}")

    # Plot
    plot_trends(all_rows, active_splits, output_dir / "checkpoint_trends.png",
                focus_target=args.focus_target)

    print("\nDone.")


if __name__ == "__main__":
    main()
