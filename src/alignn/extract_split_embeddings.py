"""
extract_split_embeddings.py
============================
Extract latent embeddings and run inference on a single train/val/test split
for ALIGNN ep100 model deployment (UQ pipeline prerequisite).

Outputs per split:
  {split}_latent_features.npz   — shape (N, 256), includes mol_ids array
  {split}_predictions.csv       — columns: mof_id + 8 targets (physical space)
  {split}_groundtruth.csv       — columns: mof_id + 8 targets (physical space)

For test split only (or --save-metrics flag):
  metrics_summary.json          — test R² per target + mean R²

Checkpoint loading (two-file strategy):
  --meta-checkpoint best_model.pt        → config + norm_stats
  --checkpoint     checkpoint_epoch0100.pt → model_state weights

Graph cache (reuses train_alignn.py convention):
  data/alignn_symlog_1e-3/cache/{split}/graphs_maxatoms{N}_cache.pkl

Usage (single split, interactive):
    CUDA_VISIBLE_DEVICES=0 python src/alignn/extract_split_embeddings.py \\
        --checkpoint  results/alignn/500ep_symlog_1e-3_ddp2g/checkpoint_epoch0100.pt \\
        --meta-checkpoint results/alignn/500ep_symlog_1e-3_ddp2g/best_model.pt \\
        --data-dir    data/alignn_symlog_1e-3 \\
        --output-dir  results/alignn/ep100_deployment \\
        --split       train \\
        --batch-size  32 \\
        --max-atoms   500

SLURM (submitted via run_extract_embeddings.sh --array=0-2):
    split index: 0=train  1=val  2=test
"""

import argparse
import json
import pickle as pk
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=UserWarning)

# ── Constants ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path("/home/zhangsd/repos/CBM-MOF")

UPTAKE_COLS = ["AdsCH4_10kPa", "AdsCH4_100kPa", "AdsCH4_1000kPa",
               "AdsN2_10kPa",  "AdsN2_100kPa",  "AdsN2_1000kPa"]
QST_COLS    = ["QstCH4", "QstN2"]
TARGET_COLS = UPTAKE_COLS + QST_COLS
N_TARGETS   = len(TARGET_COLS)

SPLIT_INDEX = ["train", "val", "test"]   # maps SLURM_ARRAY_TASK_ID → split name


# ── Transform helpers ──────────────────────────────────────────────────────────

def inv_symlog(y: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(y) * tau * (10.0 ** np.abs(y) - 1.0)


def _inv_col(y: np.ndarray, cfg: dict) -> np.ndarray:
    if cfg["type"] == "symlog":
        return inv_symlog(y, cfg["tau"])
    elif cfg["type"] == "log10":
        eps = cfg.get("eps", 1e-8)
        return 10.0 ** y - eps
    return y.copy()   # raw: identity


def invert_targets(arr: np.ndarray, xform_cfg: dict) -> np.ndarray:
    """Invert per-column transforms on an (N, 8) array → physical units."""
    out = arr.copy()
    for i, col in enumerate(TARGET_COLS):
        out[:, i] = _inv_col(arr[:, i], xform_cfg.get(col, {"type": "raw"}))
    return out


# ── Checkpoint loading ─────────────────────────────────────────────────────────

def load_model(ckpt_path: Path, meta_ckpt_path: Path, device: torch.device):
    """
    Load ALIGNN model weights from epoch checkpoint; config + norm_stats from
    best_model.pt (which always carries both fields).

    Returns:
        model      — ALIGNN in eval mode on device
        norm_mean  — np.ndarray (8,)
        norm_std   — np.ndarray (8,)
        cfg        — dict of model hyperparameters
    """
    from alignn.models.alignn import ALIGNN, ALIGNNConfig

    # Config + norm_stats from best_model.pt
    meta = torch.load(meta_ckpt_path, map_location="cpu")
    cfg  = meta["config"]
    norm_stats = meta["norm_stats"]
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

    # Weights from epoch checkpoint
    epoch_ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(epoch_ckpt["model_state"])
    model = model.to(device)
    model.eval()

    epoch = epoch_ckpt.get("epoch", "?")
    print(f"  Weights from epoch checkpoint: epoch={epoch}")
    print(f"  Config/norm_stats from: {meta_ckpt_path.name}")
    print(f"  norm_mean: {norm_mean.round(4).tolist()}")
    print(f"  norm_std : {norm_std.round(4).tolist()}")
    return model, norm_mean, norm_std, cfg


# ── Graph dataset loading (with pickle cache) ──────────────────────────────────

def collate_fn(samples):
    import dgl
    graphs, line_graphs, lattices, labels = map(list, zip(*samples))
    return (
        dgl.batch(graphs),
        dgl.batch(line_graphs),
        torch.stack(lattices),
        torch.stack(labels),
    )


def load_split_dataset(data_dir: Path, split: str, max_atoms: int):
    """
    Load StructureDataset for the given split, reusing the pickle cache from
    train_alignn.py (graphs_maxatoms{N}_cache.pkl).

    Returns:
        dataset  — StructureDataset
        mol_ids  — list[str] in dataset order
        labels_xform — np.ndarray (N, 8) in transformed space
    """
    from alignn.dataset import load_graphs
    from alignn.graphs import StructureDataset
    from jarvis.core.atoms import Atoms

    id_prop_csv = data_dir / split / "id_prop.csv"
    cif_dir     = data_dir / "cifs"

    df = pd.read_csv(id_prop_csv)
    print(f"  [{split}] CSV rows: {len(df)}")

    records, mol_ids, labels_list = [], [], []
    skipped_cif, skipped_size = 0, 0

    for _, row in df.iterrows():
        mol_id   = row["mol_id"]
        cif_path = cif_dir / f"{mol_id}.cif"
        if not cif_path.exists():
            skipped_cif += 1
            continue
        atoms = Atoms.from_cif(str(cif_path), use_cif2cell=False)
        if max_atoms > 0 and atoms.num_atoms > max_atoms:
            skipped_size += 1
            continue
        target_vals = [float(row[c]) for c in TARGET_COLS]
        records.append({
            "jid":    mol_id,
            "atoms":  atoms.to_dict(),
            "target": target_vals,
        })
        mol_ids.append(mol_id)
        labels_list.append(target_vals)

    n = len(records)
    print(f"  [{split}] Loaded: {n}  "
          f"(skipped — missing CIF: {skipped_cif}, >{max_atoms} atoms: {skipped_size})")

    df_records = pd.DataFrame(records)
    labels_xform = np.array(labels_list, dtype=np.float32)  # (N, 8) transformed

    # Pickle cache (same convention as train_alignn.py)
    cache_dir  = data_dir / "cache" / split
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_tag  = f"maxatoms{max_atoms}" if max_atoms > 0 else "all"
    cache_file = cache_dir / f"graphs_{cache_tag}_cache.pkl"

    if cache_file.exists():
        print(f"  [{split}] Loading graph cache: {cache_file.name}")
        graphs = pk.load(open(cache_file, "rb"))
    else:
        print(f"  [{split}] Building graphs (no cache found)...")
        graphs = load_graphs(
            df_records,
            neighbor_strategy="k-nearest",
            cutoff=8.0,
            cutoff_extra=3.0,
            max_neighbors=12,
            cachedir=None,       # alignn 2025.4.1 bug: cachedir broken
            use_canonize=False,
            id_tag="jid",
        )
        pk.dump(graphs, open(cache_file, "wb"))
        print(f"  [{split}] Graph cache saved: {cache_file}")

    dataset = StructureDataset(
        df_records,
        graphs,
        target="target",
        atom_features="cgcnn",
        line_graph=True,
        id_tag="jid",
    )
    return dataset, mol_ids, labels_xform


# ── Embedding extraction ───────────────────────────────────────────────────────

def find_embedding_layer(model: nn.Module) -> tuple:
    """
    Locate the first FC Linear layer (fc_layers[0] input → 256-dim features).
    Returns (layer, layer_name).
    """
    print("\n  ALIGNN FC/output layers:")
    fc_candidates = []
    for name, module in model.named_modules():
        if ("fc" in name.lower() or "out" in name.lower()) and isinstance(module, nn.Linear):
            print(f"    {name}: {module}")
            fc_candidates.append((name, module))

    if not fc_candidates:
        raise RuntimeError("No FC Linear layers found in ALIGNN model.")

    layer_name, layer = fc_candidates[0]
    print(f"\n  Hook layer: '{layer_name}'  ({layer})")
    return layer, layer_name


@torch.no_grad()
def run_inference_with_embeddings(
    model: nn.Module,
    dataset,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    hook_layer: nn.Module,
    device: torch.device,
    batch_size: int = 32,
) -> tuple:
    """
    Run batch inference and capture latent embeddings via forward hook.

    Returns:
        preds_xform — (N, 8) float32 in transformed space (z-score inverted)
        embeddings  — (N, D) float32 latent vectors
    """
    captured = {}

    def hook_fn(module, inp, out):
        # Capture input[0] to FC layer (pre-activation latent features)
        x = inp[0] if isinstance(inp, tuple) else inp
        captured["emb"] = x.detach().cpu().float()

    handle = hook_layer.register_forward_hook(hook_fn)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    all_preds, all_embs = [], []
    for batch in loader:
        g, lg, lat, _ = batch
        out = model((g.to(device), lg.to(device), lat.to(device)))
        all_preds.append(out.cpu().numpy())
        all_embs.append(captured["emb"].numpy())

    handle.remove()

    preds_zscore = np.vstack(all_preds).astype(np.float32)   # (N, 8)
    embeddings   = np.vstack(all_embs).astype(np.float32)    # (N, D)

    # z-score → transformed space
    preds_xform = preds_zscore * norm_std + norm_mean         # (N, 8)

    print(f"  Inference done — predictions: {preds_xform.shape}, embeddings: {embeddings.shape}")
    return preds_xform, embeddings


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_r2(pred: np.ndarray, true: np.ndarray) -> dict:
    """Compute per-target R² in original physical units."""
    results = {}
    for i, col in enumerate(TARGET_COLS):
        y_p = pred[:, i]
        y_t = true[:, i]
        mask = np.isfinite(y_p) & np.isfinite(y_t)
        if mask.sum() < 2:
            results[col] = None
            continue
        y_p, y_t = y_p[mask], y_t[mask]
        ss_res = float(np.sum((y_t - y_p) ** 2))
        ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        results[col] = round(r2, 6)
    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import os

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(REPO_ROOT / "results/alignn/500ep_symlog_1e-3_ddp2g/checkpoint_epoch0100.pt"),
        help="Epoch checkpoint (weights only)",
    )
    parser.add_argument(
        "--meta-checkpoint",
        type=str,
        default=str(REPO_ROOT / "results/alignn/500ep_symlog_1e-3_ddp2g/best_model.pt"),
        help="best_model.pt (config + norm_stats)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(REPO_ROOT / "data/alignn_symlog_1e-3"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "results/alignn/ep100_deployment"),
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default=None,
        help="Split to process. If omitted, uses SLURM_ARRAY_TASK_ID (0=train,1=val,2=test).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-atoms",  type=int, default=500)
    parser.add_argument(
        "--save-metrics",
        action="store_true",
        help="Always save metrics_summary.json (auto-enabled for test split).",
    )
    args = parser.parse_args()

    # Resolve split from arg or SLURM_ARRAY_TASK_ID
    if args.split is not None:
        split = args.split
    else:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        split   = SPLIT_INDEX[task_id]

    ckpt_path      = Path(args.checkpoint)
    meta_ckpt_path = Path(args.meta_checkpoint)
    data_dir       = Path(args.data_dir)
    output_dir     = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"extract_split_embeddings.py")
    print(f"  Split         : {split}")
    print(f"  Checkpoint    : {ckpt_path.name}")
    print(f"  Meta ckpt     : {meta_ckpt_path.name}")
    print(f"  Data dir      : {data_dir}")
    print(f"  Output dir    : {output_dir}")
    print(f"  Batch size    : {args.batch_size}")
    print(f"  Max atoms     : {args.max_atoms}")
    print(f"{'=' * 60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. Load model ──────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading model...")
    model, norm_mean, norm_std, cfg = load_model(ckpt_path, meta_ckpt_path, device)

    # ── 2. Identify embedding layer ────────────────────────────────────────────
    print(f"\n[2/4] Identifying embedding layer...")
    hook_layer, hook_layer_name = find_embedding_layer(model)

    # ── 3. Load transform config ───────────────────────────────────────────────
    xform_cfg_path = data_dir / "transform_config.json"
    with open(xform_cfg_path) as f:
        xform_cfg = json.load(f)
    print(f"\n[3/4] Transform config loaded ({len(xform_cfg)} entries): {xform_cfg_path}")

    # ── 4. Load split dataset + run inference ──────────────────────────────────
    print(f"\n[4/4] Loading [{split}] dataset and running inference...")
    dataset, mol_ids, labels_xform = load_split_dataset(data_dir, split, args.max_atoms)

    preds_xform, embeddings = run_inference_with_embeddings(
        model, dataset, norm_mean, norm_std, hook_layer, device, args.batch_size,
    )

    # Invert transforms → physical space
    labels_orig = invert_targets(labels_xform, xform_cfg)   # (N, 8) mol/kg + kJ/mol
    preds_orig  = invert_targets(preds_xform,  xform_cfg)   # (N, 8)

    # Validate no NaN/Inf in embeddings
    n_nan_emb  = int(np.sum(~np.isfinite(embeddings)))
    n_nan_pred = int(np.sum(~np.isfinite(preds_orig)))
    print(f"  NaN/Inf in embeddings : {n_nan_emb}")
    print(f"  NaN/Inf in predictions: {n_nan_pred}")

    # ── Save outputs ───────────────────────────────────────────────────────────
    # Latent features (.npz with mol_ids)
    npz_path = output_dir / f"{split}_latent_features.npz"
    np.savez_compressed(
        npz_path,
        features=embeddings,
        mol_ids=np.array(mol_ids, dtype=str),
    )
    print(f"\n  Saved: {npz_path}  (shape={embeddings.shape})")

    # Predictions CSV
    pred_df = pd.DataFrame(preds_orig, columns=TARGET_COLS)
    pred_df.insert(0, "mof_id", mol_ids)
    pred_csv = output_dir / f"{split}_predictions.csv"
    pred_df.to_csv(pred_csv, index=False)
    print(f"  Saved: {pred_csv}")

    # Ground truth CSV
    gt_df = pd.DataFrame(labels_orig, columns=TARGET_COLS)
    gt_df.insert(0, "mof_id", mol_ids)
    gt_csv = output_dir / f"{split}_groundtruth.csv"
    gt_df.to_csv(gt_csv, index=False)
    print(f"  Saved: {gt_csv}")

    # ── Metrics (always for test, opt-in for others) ───────────────────────────
    if split == "test" or args.save_metrics:
        r2_per_target = compute_r2(preds_orig, labels_orig)
        valid_r2 = [v for v in r2_per_target.values() if v is not None]
        mean_r2  = float(np.mean(valid_r2)) if valid_r2 else None

        metrics_summary = {
            "split":          split,
            "checkpoint":     str(ckpt_path),
            "meta_checkpoint": str(meta_ckpt_path),
            "n_samples":      len(mol_ids),
            "embedding_dim":  int(embeddings.shape[1]),
            "hook_layer":     hook_layer_name,
            "r2_per_target":  r2_per_target,
            "mean_r2":        mean_r2,
        }
        metrics_path = output_dir / "metrics_summary.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_summary, f, indent=2)

        print(f"\n  Test metrics (original physical space):")
        print(f"  {'Target':25s}  {'R²':>8}")
        print(f"  " + "-" * 38)
        for col in TARGET_COLS:
            r2 = r2_per_target.get(col)
            r2_str = f"{r2:8.4f}" if r2 is not None else "     N/A"
            print(f"  {col:25s}  {r2_str}")
        if mean_r2 is not None:
            print(f"\n  Mean R²: {mean_r2:.4f}  (target ≈ 0.908)")
        print(f"\n  Saved: {metrics_path}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"DONE — [{split}]")
    print(f"  Embeddings shape : {embeddings.shape}")
    print(f"  Output dir       : {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
