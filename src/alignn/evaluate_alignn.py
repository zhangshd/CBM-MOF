"""
evaluate_alignn.py
==================
Evaluate the trained ALIGNN checkpoint on three datasets in original physical space:
  A) Test set    — ~994 MOFs with symlog uptake + raw kJ/mol Qst labels
  B) Top-100 PSA — 100 MOFs with GCMC mol/kg uptake + Widom kJ/mol Qst labels
  C) Top-100 VSA — 100 MOFs (same label format as B)

For each dataset, produces:
  - {name}_predictions.csv  — per-MOF predictions vs. truth in original units
  - {name}_metrics.json     — per-target MAE and R² in original units
  - {name}_parity.png       — 2×4 parity plots with R²/MAE/RMSE annotations

Inverse transform chain:
  model output (z-score) → × std + mean → symlog / kJ/mol → inv_symlog [cols 0–5] → mol/kg

Usage:
    CUDA_VISIBLE_DEVICES=0 python src/alignn/evaluate_alignn.py

    # Custom checkpoint / output:
    python src/alignn/evaluate_alignn.py \\
        --checkpoint results/alignn/short_50ep/best_model.pt \\
        --output-dir results/alignn/short_50ep/evaluation \\
        --max-atoms 300
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import dgl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
REPO_ROOT   = Path("/home/zhangsd/repos/CBM-MOF")
ALIGNN_DATA = REPO_ROOT / "data/alignn"
INFER_DIR   = REPO_ROOT / "results/cbm_screening/inference"

UPTAKE_COLS = ["AdsCH4_10kPa", "AdsCH4_100kPa", "AdsCH4_1000kPa",
               "AdsN2_10kPa",  "AdsN2_100kPa",  "AdsN2_1000kPa"]
QST_COLS    = ["QstCH4", "QstN2"]
TARGET_COLS = UPTAKE_COLS + QST_COLS
N_TARGETS   = len(TARGET_COLS)

SYMLOG_THRESH = 1e-4

# Pressure bar → target suffix (gas-specific handled in load_gcmc_uptake)
PRESSURE_MAP = {0.1: "10kPa", 1.0: "100kPa", 10.0: "1000kPa"}
# Gas name → abbreviation
GAS_MAP = {"methane": "CH4", "N2": "N2"}

# Units for axis labels
UNITS = {col: "mol/kg" for col in UPTAKE_COLS}
UNITS.update({col: "kJ/mol" for col in QST_COLS})


# ──────────────────────────────────────────────────────────────
# Math helpers
# ──────────────────────────────────────────────────────────────
def inv_symlog(y: np.ndarray, threshold: float = SYMLOG_THRESH) -> np.ndarray:
    """Inverse of symlog(x, τ): sign(y) × τ × (10^|y| - 1)."""
    return np.sign(y) * threshold * (10.0 ** np.abs(y) - 1.0)


def invert_prediction(pred_zscore: np.ndarray, norm_mean: np.ndarray,
                      norm_std: np.ndarray) -> np.ndarray:
    """Full inverse: z-score → symlog/kJ/mol → mol/kg (uptakes only).

    Args:
        pred_zscore: (N, 8) model output in z-score space
        norm_mean:   (8,)   training set mean per target
        norm_std:    (8,)   training set std  per target

    Returns:
        (N, 8)  in original physical units (mol/kg and kJ/mol)
    """
    pred_orig = pred_zscore * norm_std + norm_mean    # → symlog / kJ/mol
    n_up = len(UPTAKE_COLS)
    pred_orig[:, :n_up] = inv_symlog(pred_orig[:, :n_up])  # → mol/kg
    return pred_orig


# ──────────────────────────────────────────────────────────────
# Checkpoint loading
# ──────────────────────────────────────────────────────────────
def load_checkpoint(ckpt_path: Path, device: torch.device):
    """Load ALIGNN model and norm_stats from best_model.pt.

    Returns:
        model      — ALIGNN model in eval mode on device
        norm_mean  — np.ndarray (8,)
        norm_std   — np.ndarray (8,)
        cfg        — dict of model hyperparameters
    """
    from alignn.models.alignn import ALIGNN, ALIGNNConfig

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt["config"]

    norm_stats = ckpt["norm_stats"]
    norm_mean  = np.array(norm_stats["mean"], dtype=np.float32)
    norm_std   = np.array(norm_stats["std"],  dtype=np.float32)

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
    model = ALIGNN(alignn_cfg)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    print(f"  Loaded checkpoint: epoch={epoch}, val_mae={ckpt.get('val_mae', '?'):.5f}")
    return model, norm_mean, norm_std, cfg


# ──────────────────────────────────────────────────────────────
# Dataset construction
# ──────────────────────────────────────────────────────────────
def build_dataset_from_records(records: list, max_atoms: int = 300):
    """Build an ALIGNN StructureDataset from a list of records.

    Each record: {"jid": str, "atoms": dict, "target": list[float]}
    Returns (dataset, mol_ids) or raises if no valid records.
    """
    from alignn.dataset import load_graphs
    from alignn.graphs import StructureDataset

    df = pd.DataFrame(records)
    graphs = load_graphs(
        df,
        neighbor_strategy="k-nearest",
        cutoff=8.0,
        cutoff_extra=3.0,
        max_neighbors=12,
        cachedir=None,
        use_canonize=False,
        id_tag="jid",
    )
    dataset = StructureDataset(
        df,
        graphs,
        target="target",
        atom_features="cgcnn",
        line_graph=True,
        id_tag="jid",
    )
    mol_ids = df["jid"].tolist()
    return dataset, mol_ids


def collate_fn(samples):
    """Custom collate: (g, lg, lat, labels) — required for ALIGNN 2025.x multi-task."""
    graphs, line_graphs, lattices, labels = map(list, zip(*samples))
    return (
        dgl.batch(graphs),
        dgl.batch(line_graphs),
        torch.stack(lattices),
        torch.stack(labels),
    )


# ──────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, dataset, norm_mean: np.ndarray, norm_std: np.ndarray,
                  device: torch.device, batch_size: int = 16) -> np.ndarray:
    """Run batch inference and return predictions in original physical units.

    Returns:
        preds_orig — (N, 8) float32 in original physical space
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True, collate_fn=collate_fn)
    preds_list = []
    for batch in loader:
        g, lg, lat, _ = batch
        g   = g.to(device)
        lg  = lg.to(device)
        lat = lat.to(device)
        out = model((g, lg, lat)).cpu().numpy()          # (b, 8) z-score
        out = np.atleast_2d(out)                          # guard: shape (8,) when last batch_size=1
        preds_list.append(invert_prediction(out, norm_mean, norm_std))
    return np.vstack(preds_list)


# ──────────────────────────────────────────────────────────────
# Label loaders
# ──────────────────────────────────────────────────────────────
def load_test_labels(id_prop_csv: Path) -> pd.DataFrame:
    """Load test set labels from id_prop.csv.

    id_prop.csv columns: mol_id, {symlog uptake cols...}, {Qst kJ/mol cols...}
    Returns DataFrame with columns [mol_id] + TARGET_COLS in original units.
    """
    df = pd.read_csv(id_prop_csv)
    # Invert symlog transform on uptake columns (labels are in symlog space)
    for col in UPTAKE_COLS:
        df[col] = inv_symlog(df[col].values)
    return df.set_index("mol_id")


def load_gcmc_uptake(gcmc_csv: Path) -> pd.DataFrame:
    """Parse raspa3 long-format CSV to wide uptake DataFrame.

    Returns DataFrame indexed by MofName with columns = UPTAKE_COLS (mol/kg).
    """
    df = pd.read_csv(gcmc_csv)
    # Build target column name: Ads{gas}_{suffix}
    df["col"] = df.apply(
        lambda r: f"Ads{GAS_MAP[r['GasName']]}_{PRESSURE_MAP[r['Pressure[bar]']]}",
        axis=1,
    )
    pivot = df.pivot_table(index="MofName", columns="col",
                           values="AbsLoading", aggfunc="first")
    # Ensure all 6 uptake columns present (may be missing for some MOFs)
    for col in UPTAKE_COLS:
        if col not in pivot.columns:
            pivot[col] = np.nan
    return pivot[UPTAKE_COLS].copy()


def load_widom_qst(widom_csv: Path) -> pd.DataFrame:
    """Parse raspa2 Widom long-format CSV to wide Qst DataFrame.

    Returns DataFrame indexed by MofName with columns = QST_COLS (kJ/mol).
    """
    df = pd.read_csv(widom_csv)
    df["col"] = df["GasName"].map(lambda g: f"Qst{GAS_MAP[g]}")
    pivot = df.pivot_table(index="MofName", columns="col",
                           values="AdsorptionHeat", aggfunc="first")
    for col in QST_COLS:
        if col not in pivot.columns:
            pivot[col] = np.nan
    return pivot[QST_COLS].copy()


def build_top100_labels(gcmc_csv: Path, widom_csv: Path) -> pd.DataFrame:
    """Merge GCMC uptake + Widom Qst into a single wide DataFrame indexed by MofName."""
    uptake_df = load_gcmc_uptake(gcmc_csv)
    qst_df    = load_widom_qst(widom_csv)
    merged    = pd.concat([uptake_df, qst_df], axis=1)
    return merged[TARGET_COLS]


# ──────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────
def compute_metrics_orig(pred: np.ndarray, true: np.ndarray) -> dict:
    """Compute per-target MAE, RMSE, R² in original physical units."""
    results = {}
    for i, col in enumerate(TARGET_COLS):
        y_p = pred[:, i]
        y_t = true[:, i]
        # Drop NaN pairs (some top-100 MOFs may be missing labels)
        mask = np.isfinite(y_p) & np.isfinite(y_t)
        if mask.sum() < 2:
            results[col] = {"MAE": None, "RMSE": None, "R2": None, "n": int(mask.sum())}
            continue
        y_p, y_t = y_p[mask], y_t[mask]
        mae  = float(np.mean(np.abs(y_p - y_t)))
        rmse = float(np.sqrt(np.mean((y_p - y_t) ** 2)))
        ss_res = np.sum((y_t - y_p) ** 2)
        ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
        r2   = float(1 - ss_res / (ss_tot + 1e-12))
        results[col] = {"MAE": mae, "RMSE": rmse, "R2": r2, "n": int(mask.sum()),
                        "unit": UNITS[col]}
    return results


# ──────────────────────────────────────────────────────────────
# Parity plots
# ──────────────────────────────────────────────────────────────
def plot_parity(pred_df: pd.DataFrame, true_df: pd.DataFrame,
                metrics: dict, title: str, out_path: Path) -> None:
    """Generate 2×4 parity plots for all 8 targets.

    Args:
        pred_df: DataFrame (N × 8) predictions in original units
        true_df: DataFrame (N × 8) ground truth in original units
        metrics: output of compute_metrics_orig
        title:   suptitle for the figure
        out_path: output PNG path
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    for i, col in enumerate(TARGET_COLS):
        ax = axes[i]
        y_p = pred_df[col].values
        y_t = true_df[col].values
        mask = np.isfinite(y_p) & np.isfinite(y_t)
        y_p, y_t = y_p[mask], y_t[mask]

        ax.scatter(y_t, y_p, s=8, alpha=0.5, edgecolors="none",
                   color="#2196F3" if col in UPTAKE_COLS else "#FF5722")

        # Perfect-prediction line
        lo = min(y_t.min(), y_p.min())
        hi = max(y_t.max(), y_p.max())
        pad = (hi - lo) * 0.05
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                "k--", linewidth=0.8, alpha=0.6)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)

        m = metrics.get(col, {})
        mae  = m.get("MAE");  rmse = m.get("RMSE");  r2 = m.get("R2")
        n    = m.get("n", len(y_t))
        ann  = f"MAE={mae:.3f}\nRMSE={rmse:.3f}\nR²={r2:.3f}\nn={n}" \
               if (mae is not None) else "N/A"
        ax.text(0.03, 0.97, ann, transform=ax.transAxes,
                verticalalignment="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        unit = UNITS[col]
        ax.set_xlabel(f"True ({unit})", fontsize=8)
        ax.set_ylabel(f"Predicted ({unit})", fontsize=8)
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Parity plot saved: {out_path}")


# ──────────────────────────────────────────────────────────────
# Dataset evaluators
# ──────────────────────────────────────────────────────────────
def evaluate_test_set(model, norm_mean, norm_std, device, output_dir, max_atoms):
    """Evaluate model on the held-out test split."""
    from jarvis.core.atoms import Atoms

    print("\n=== Dataset A: Test Set ===")
    id_prop_csv = ALIGNN_DATA / "test" / "id_prop.csv"
    cif_dir     = ALIGNN_DATA / "cifs"

    df = pd.read_csv(id_prop_csv)
    print(f"  Total in CSV: {len(df)}")

    # Build records (skip missing CIFs and oversized structures)
    records, mol_ids_true, true_rows = [], [], []
    missing, skipped_size = 0, 0
    for _, row in df.iterrows():
        mol_id   = row["mol_id"]
        cif_path = cif_dir / f"{mol_id}.cif"
        if not cif_path.exists():
            missing += 1
            continue
        atoms = Atoms.from_cif(str(cif_path), use_cif2cell=False)
        if max_atoms > 0 and atoms.num_atoms > max_atoms:
            skipped_size += 1
            continue
        records.append({"jid": mol_id, "atoms": atoms.to_dict(),
                         "target": [0.0] * N_TARGETS})  # dummy label
        mol_ids_true.append(mol_id)
        true_rows.append(row)

    print(f"  Missing CIFs: {missing}  |  Skipped (>{max_atoms} atoms): {skipped_size}")
    print(f"  Evaluating: {len(records)} structures")

    dataset, mol_ids = build_dataset_from_records(records, max_atoms)
    preds_orig = run_inference(model, dataset, norm_mean, norm_std, device)  # (N, 8) mol/kg + kJ/mol

    # Ground truth: invert symlog for uptakes
    true_df = pd.DataFrame(true_rows).set_index("mol_id")
    true_orig = true_df[TARGET_COLS].copy()
    for col in UPTAKE_COLS:
        true_orig[col] = inv_symlog(true_orig[col].values)

    pred_df = pd.DataFrame(preds_orig, columns=TARGET_COLS, index=mol_ids)
    true_aligned = true_orig.loc[pred_df.index]

    metrics = compute_metrics_orig(pred_df.values, true_aligned.values)
    _print_metrics("Test", metrics)

    # Save
    out = output_dir / "test"
    out.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out / "test_predictions.csv")
    true_aligned.to_csv(out / "test_truth.csv")
    combined = pred_df.copy().add_suffix("_pred")
    combined = combined.join(true_aligned.add_suffix("_true"))
    combined.to_csv(out / "test_predictions_vs_truth.csv")
    with open(out / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    plot_parity(pred_df, true_aligned, metrics,
                title="ALIGNN Test Set — Original Space",
                out_path=out / "test_parity.png")
    return metrics


def evaluate_top100(model, norm_mean, norm_std, device, output_dir, max_atoms,
                    name: str, cif_dir: Path, gcmc_csv: Path, widom_csv: Path):
    """Evaluate model on Top-100 PSA or VSA dataset."""
    from jarvis.core.atoms import Atoms

    print(f"\n=== Dataset {name} ===")

    # Load labels
    labels_df = build_top100_labels(gcmc_csv, widom_csv)
    print(f"  Labels loaded: {len(labels_df)} MOFs")

    # Build records
    cif_files = sorted(cif_dir.glob("*.cif"))
    records, mol_ids_valid, label_rows = [], [], []
    missing, skipped_size = 0, 0
    for cif_path in cif_files:
        mol_id = cif_path.stem
        if mol_id not in labels_df.index:
            missing += 1
            continue
        atoms = Atoms.from_cif(str(cif_path), use_cif2cell=False)
        if max_atoms > 0 and atoms.num_atoms > max_atoms:
            skipped_size += 1
            continue
        records.append({"jid": mol_id, "atoms": atoms.to_dict(),
                         "target": [0.0] * N_TARGETS})
        mol_ids_valid.append(mol_id)
        label_rows.append(labels_df.loc[mol_id])

    print(f"  Missing labels: {missing}  |  Skipped (>{max_atoms} atoms): {skipped_size}")
    print(f"  Evaluating: {len(records)} structures")

    if not records:
        print(f"  WARNING: No valid records for {name}, skipping.")
        return {}

    dataset, mol_ids = build_dataset_from_records(records, max_atoms)
    preds_orig = run_inference(model, dataset, norm_mean, norm_std, device)

    pred_df   = pd.DataFrame(preds_orig, columns=TARGET_COLS, index=mol_ids)
    true_df   = pd.DataFrame(label_rows, index=mol_ids)[TARGET_COLS]

    metrics = compute_metrics_orig(pred_df.values, true_df.values)
    _print_metrics(name, metrics)

    # Save
    tag = name.lower().replace(" ", "_").replace("-", "_")
    out = output_dir / tag
    out.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out / f"{tag}_predictions.csv")
    true_df.to_csv(out / f"{tag}_truth.csv")
    combined = pred_df.copy().add_suffix("_pred").join(true_df.add_suffix("_true"))
    combined.to_csv(out / f"{tag}_predictions_vs_truth.csv")
    with open(out / f"{tag}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    plot_parity(pred_df, true_df, metrics,
                title=f"ALIGNN {name} — Original Space",
                out_path=out / f"{tag}_parity.png")
    return metrics


# ──────────────────────────────────────────────────────────────
# Printing helpers
# ──────────────────────────────────────────────────────────────
def _print_metrics(name: str, metrics: dict) -> None:
    print(f"\n  [{name}] Metrics in original physical space:")
    print(f"  {'Target':25s}  {'MAE':>10}  {'RMSE':>10}  {'R²':>8}  {'n':>6}")
    print("  " + "-" * 65)
    for col in TARGET_COLS:
        m = metrics.get(col, {})
        mae = m.get("MAE");  rmse = m.get("RMSE");  r2 = m.get("R2")
        n   = m.get("n", 0); unit = UNITS[col]
        if mae is not None:
            print(f"  {col:25s}  {mae:10.4f}  {rmse:10.4f}  {r2:8.4f}  {n:6d}  ({unit})")
        else:
            print(f"  {col:25s}  {'N/A':>10}  {'N/A':>10}  {'N/A':>8}  {n:6d}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ALIGNN checkpoint on test set + top-100 PSA/VSA"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(REPO_ROOT / "results/alignn/short_50ep/best_model.pt"),
        help="Path to best_model.pt checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "results/alignn/short_50ep/evaluation"),
        help="Directory for evaluation outputs",
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=300,
        help="Filter out structures with more than N atoms (0 = disabled)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip test set evaluation (e.g., for quick top-100 check)",
    )
    args = parser.parse_args()

    ckpt_path  = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint : {ckpt_path}")
    print(f"Output dir : {output_dir}")
    print(f"Max atoms  : {args.max_atoms}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"Device     : {device}  ({n_gpus} GPU(s) visible)")

    # Load model
    print("\nLoading checkpoint...")
    model, norm_mean, norm_std, cfg = load_checkpoint(ckpt_path, device)
    print(f"  norm_mean: {norm_mean.tolist()}")
    print(f"  norm_std : {norm_std.tolist()}")

    all_metrics = {}

    # ── Dataset A: Test Set ──────────────────────────────────
    if not args.skip_test:
        all_metrics["test"] = evaluate_test_set(
            model, norm_mean, norm_std, device, output_dir, args.max_atoms
        )

    # ── Dataset B: Top-100 PSA ───────────────────────────────
    all_metrics["top100_psa"] = evaluate_top100(
        model, norm_mean, norm_std, device, output_dir, args.max_atoms,
        name="Top-100 PSA",
        cif_dir=INFER_DIR / "top_100_psa_performers_cifs_ml",
        gcmc_csv=INFER_DIR / "gcmc_top_100_psa_ml/raspa3_parsed_results.csv",
        widom_csv=INFER_DIR / "widom_top_100_psa_ml_org/raspa2_parsed_results.csv",
    )

    # ── Dataset C: Top-100 VSA ───────────────────────────────
    all_metrics["top100_vsa"] = evaluate_top100(
        model, norm_mean, norm_std, device, output_dir, args.max_atoms,
        name="Top-100 VSA",
        cif_dir=INFER_DIR / "top_100_vsa_performers_cifs_ml",
        gcmc_csv=INFER_DIR / "gcmc_top_100_vsa_ml/raspa3_parsed_results.csv",
        widom_csv=INFER_DIR / "widom_top_100_vsa_ml_org/raspa2_parsed_results.csv",
    )

    # ── Summary ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY — Mean MAE across 8 targets (original physical space)")
    print("=" * 70)
    for dset_name, metrics in all_metrics.items():
        maes = [m["MAE"] for m in metrics.values() if m.get("MAE") is not None]
        if maes:
            print(f"  {dset_name:20s}  mean MAE = {np.mean(maes):.4f}")
        else:
            print(f"  {dset_name:20s}  no valid metrics")

    # Save combined summary
    with open(output_dir / "all_metrics_summary.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nAll metrics saved: {output_dir}/all_metrics_summary.json")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
