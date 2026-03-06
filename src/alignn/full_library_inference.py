"""
full_library_inference.py
=========================
Task 1.1c: Run ALIGNN inference on the full CBM-MOF library (~234,649 MOFs).

Designed for SLURM array jobs: each array task processes one batch of CIFs.
Progress is reported via periodic print statements (no tqdm) for clean SLURM logs.

Outputs per batch:
  results/alignn/full_library_inference/batches/
      batch_{idx:04d}_features.npz    -- (N_i, 256) latent embeddings + mol_ids
      batch_{idx:04d}_predictions.csv -- mof_id + 8 targets (physical space)
  results/alignn/full_library_inference/
      failed_mofs_{idx:04d}.txt       -- MOFs skipped due to CIF error or size filter

Checkpoint loading strategy (two-file, same as extract_split_embeddings.py):
  --checkpoint      checkpoint_epoch0100.pt  -- model weights
  --meta-checkpoint best_model.pt            -- config + norm_stats

Usage (interactive dry-run, single batch):
    cd /home/zhangsd/repos/CBM-MOF
    CUDA_VISIBLE_DEVICES=0 python src/alignn/full_library_inference.py \\
        --batch-idx 0 --n-batches 24 --batch-size 8 --max-atoms 500

SLURM submission:
    sbatch --array=0-23 src/alignn/run_full_library_inference.sh
"""

import argparse
import json
import os
import pickle as pk
import sys
import time
import warnings
from pathlib import Path

# Suppress tqdm progress bars from ALIGNN internals when running non-interactively
# (SLURM log files receive tqdm escape codes that pollute output — use print instead)
if not sys.stdout.isatty() or os.environ.get("SLURM_JOB_ID"):
    os.environ["TQDM_DISABLE"] = "1"

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

LOG_INTERVAL = 500   # print progress every N MOFs during graph building


# ── Transform helpers (verbatim from extract_split_embeddings.py) ─────────────

def inv_symlog(y: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(y) * tau * (10.0 ** np.abs(y) - 1.0)


def _inv_col(y: np.ndarray, cfg: dict) -> np.ndarray:
    if cfg["type"] == "symlog":
        return inv_symlog(y, cfg["tau"])
    elif cfg["type"] == "log10":
        eps = cfg.get("eps", 1e-8)
        return 10.0 ** y - eps
    return y.copy()


def invert_targets(arr: np.ndarray, xform_cfg: dict) -> np.ndarray:
    """Invert per-column transforms on an (N, 8) array → physical units."""
    out = arr.copy()
    for i, col in enumerate(TARGET_COLS):
        out[:, i] = _inv_col(arr[:, i], xform_cfg.get(col, {"type": "raw"}))
    return out


# ── Checkpoint loading (verbatim from extract_split_embeddings.py) ────────────

def load_model(ckpt_path: Path, meta_ckpt_path: Path, device: torch.device):
    """
    Load ALIGNN model weights from epoch checkpoint; config + norm_stats from
    best_model.pt (meta checkpoint).

    Returns:
        model      -- ALIGNN in eval mode on device
        norm_mean  -- np.ndarray (8,)
        norm_std   -- np.ndarray (8,)
        cfg        -- dict of model hyperparameters
    """
    from alignn.models.alignn import ALIGNN, ALIGNNConfig

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

    epoch_ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(epoch_ckpt["model_state"])
    model = model.to(device)
    model.eval()

    epoch = epoch_ckpt.get("epoch", "?")
    print(f"  Weights loaded: epoch={epoch}")
    print(f"  Config/norm_stats from: {meta_ckpt_path.name}")
    print(f"  norm_mean: {norm_mean.round(4).tolist()}")
    print(f"  norm_std : {norm_std.round(4).tolist()}")
    return model, norm_mean, norm_std, cfg


# ── Embedding layer (verbatim from extract_split_embeddings.py) ───────────────

def find_embedding_layer(model: nn.Module) -> tuple:
    """Locate the first FC Linear layer (256-dim latent space)."""
    print("  ALIGNN FC/output layers:")
    fc_candidates = []
    for name, module in model.named_modules():
        if ("fc" in name.lower() or "out" in name.lower()) and isinstance(module, nn.Linear):
            print(f"    {name}: {module}")
            fc_candidates.append((name, module))

    if not fc_candidates:
        raise RuntimeError("No FC Linear layers found in ALIGNN model.")

    layer_name, layer = fc_candidates[0]
    print(f"  Hook layer: '{layer_name}'  ({layer})")
    return layer, layer_name


# ── DataLoader collate (verbatim from extract_split_embeddings.py) ────────────

def collate_fn(samples):
    import dgl
    graphs, line_graphs, lattices, labels = map(list, zip(*samples))
    return (
        dgl.batch(graphs),
        dgl.batch(line_graphs),
        torch.stack(lattices),
        torch.stack(labels),
    )


# ── Batch splitting ────────────────────────────────────────────────────────────

def get_batch_ids(all_ids: list, n_batches: int, batch_idx: int) -> list:
    """Return the slice of mol_ids for this batch (roughly equal-sized chunks)."""
    n = len(all_ids)
    chunk = (n + n_batches - 1) // n_batches   # ceiling division
    start = batch_idx * chunk
    end   = min(start + chunk, n)
    return all_ids[start:end]


# ── Graph building with per-MOF error isolation ───────────────────────────────

def build_batch_graphs(
    mol_ids: list,
    cif_dir: Path,
    max_atoms: int,
    graph_cache_path: Path,
) -> tuple:
    """
    Build ALIGNN graphs for a batch of MOFs.

    Attempts to load from graph_cache_path first.
    On cache miss, builds graphs from CIF files with per-MOF error isolation.

    Returns:
        valid_ids   -- list[str] of successfully processed MOF IDs
        graphs      -- list of DGL graphs from alignn (paired with valid_ids)
        records     -- list[dict] with keys jid/atoms/target (for StructureDataset)
        failed_ids  -- list[str] of MOF IDs that failed (CIF parse or size)
    """
    from alignn.dataset import load_graphs
    from jarvis.core.atoms import Atoms

    # ── Try loading from cache ─────────────────────────────────────────────────
    if graph_cache_path is not None and graph_cache_path.exists():
        print(f"  Loading graph cache: {graph_cache_path.name}")
        with open(graph_cache_path, "rb") as f:
            cache = pk.load(f)
        return cache["valid_ids"], cache["graphs"], cache["records"], cache["failed_ids"]

    # ── Build from scratch ─────────────────────────────────────────────────────
    print(f"  Building graphs from CIFs (n={len(mol_ids)}, max_atoms={max_atoms})...")
    t0 = time.time()

    records   = []
    valid_ids = []
    failed_ids = []

    n_missing = 0
    n_toolarge = 0
    n_cif_err  = 0

    for i, mol_id in enumerate(mol_ids):
        if (i + 1) % LOG_INTERVAL == 0:
            elapsed = time.time() - t0
            print(f"    CIF parsing: {i+1}/{len(mol_ids)}  "
                  f"(ok={len(valid_ids)}, failed={len(failed_ids)})  "
                  f"elapsed={elapsed:.0f}s")

        cif_path = cif_dir / f"{mol_id}.cif"

        if not cif_path.exists():
            n_missing += 1
            failed_ids.append(f"{mol_id}\tmissing_cif")
            continue

        try:
            atoms = Atoms.from_cif(str(cif_path), use_cif2cell=False)
        except Exception as e:
            n_cif_err += 1
            failed_ids.append(f"{mol_id}\tcif_parse_error\t{str(e)[:80]}")
            continue

        if max_atoms > 0 and atoms.num_atoms > max_atoms:
            n_toolarge += 1
            failed_ids.append(f"{mol_id}\ttoo_large\t{atoms.num_atoms}")
            continue

        records.append({
            "jid":    mol_id,
            "atoms":  atoms.to_dict(),
            "target": [0.0] * N_TARGETS,   # placeholder labels
        })
        valid_ids.append(mol_id)

    t_parse = time.time() - t0
    print(f"  CIF parsing done in {t_parse:.1f}s: "
          f"valid={len(valid_ids)}, missing={n_missing}, "
          f"too_large={n_toolarge}, parse_error={n_cif_err}")

    if not records:
        print("  WARNING: No valid records in this batch — nothing to infer.")
        return [], [], [], failed_ids

    # ── Build ALIGNN line graphs ───────────────────────────────────────────────
    print(f"  Building ALIGNN line graphs for {len(records)} MOFs...")
    t1 = time.time()
    df_records = pd.DataFrame(records)
    graphs = load_graphs(
        df_records,
        neighbor_strategy="k-nearest",
        cutoff=8.0,
        cutoff_extra=3.0,
        max_neighbors=12,
        cachedir=None,      # alignn 2025 bug: internal cachedir broken
        use_canonize=False,
        id_tag="jid",
    )
    t_graph = time.time() - t1
    print(f"  Graph building done in {t_graph:.1f}s  (total graphs: {len(graphs)})")

    # ── Save cache (skipped if --no-cache) ────────────────────────────────────
    if graph_cache_path is not None:
        graph_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(graph_cache_path, "wb") as f:
            pk.dump({
                "valid_ids": valid_ids,
                "graphs":    graphs,
                "records":   records,
                "failed_ids": failed_ids,
            }, f)
        print(f"  Graph cache saved: {graph_cache_path}")

    return valid_ids, graphs, records, failed_ids


# ── Inference + embedding extraction ──────────────────────────────────────────

@torch.no_grad()
def run_inference_with_embeddings(
    model: nn.Module,
    dataset,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    hook_layer: nn.Module,
    device: torch.device,
    batch_size: int,
    total_mofs: int,
) -> tuple:
    """
    Run batch inference and capture latent embeddings via forward hook.
    Prints progress every ~10% of batches for clean SLURM logs (no tqdm).

    Returns:
        preds_xform -- (N, 8) float32 in transformed space (z-score inverted)
        embeddings  -- (N, 256) float32 latent vectors
    """
    captured = {}

    def hook_fn(module, inp, out):
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

    n_batches = len(loader)
    log_every = max(1, n_batches // 10)   # log ~10 times total

    all_preds, all_embs = [], []
    t0 = time.time()

    for i, batch in enumerate(loader):
        g, lg, lat, _ = batch
        out = model((g.to(device), lg.to(device), lat.to(device)))
        all_preds.append(out.cpu().numpy())
        all_embs.append(captured["emb"].numpy())

        if (i + 1) % log_every == 0 or (i + 1) == n_batches:
            done = (i + 1) * batch_size
            elapsed = time.time() - t0
            print(f"    Inference: batch {i+1}/{n_batches}  "
                  f"(~{min(done, total_mofs)}/{total_mofs} MOFs)  "
                  f"elapsed={elapsed:.0f}s")

    handle.remove()

    preds_zscore = np.vstack(all_preds).astype(np.float32)   # (N, 8)
    embeddings   = np.vstack(all_embs).astype(np.float32)    # (N, 256)

    preds_xform = preds_zscore * norm_std + norm_mean         # z-score → transformed

    elapsed_total = time.time() - t0
    print(f"  Inference done in {elapsed_total:.1f}s: "
          f"preds={preds_xform.shape}, embeddings={embeddings.shape}")
    return preds_xform, embeddings


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(REPO_ROOT / "results/alignn/500ep_symlog_1e-3_ddp2g/checkpoint_epoch0100.pt"),
        help="Epoch checkpoint (model weights)",
    )
    parser.add_argument(
        "--meta-checkpoint",
        type=str,
        default=str(REPO_ROOT / "results/alignn/500ep_symlog_1e-3_ddp2g/best_model.pt"),
        help="best_model.pt (config + norm_stats)",
    )
    parser.add_argument(
        "--cif-dir",
        type=str,
        default=str(REPO_ROOT / "results/cbm_screening/all_graphs_grids"),
        help="Directory containing all *.cif files",
    )
    parser.add_argument(
        "--xform-config",
        type=str,
        default=str(REPO_ROOT / "data/alignn_symlog_1e-3/transform_config.json"),
        help="Path to transform_config.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "results/alignn/full_library_inference"),
    )
    parser.add_argument(
        "--n-batches",
        type=int,
        default=24,
        help="Total number of array batches (default: 24 for ~234,649 MOFs)",
    )
    parser.add_argument(
        "--batch-idx",
        type=int,
        default=None,
        help="Index of this batch (0-based). Defaults to SLURM_ARRAY_TASK_ID.",
    )
    parser.add_argument("--batch-size",  type=int, default=8,
                        help="DataLoader batch size (default: 8 for OOM safety)")
    parser.add_argument("--max-atoms",   type=int, default=500,
                        help="Skip MOFs with more atoms than this")
    parser.add_argument("--no-cache",    action="store_true",
                        help="Disable graph cache (re-build even if cache exists)")
    args = parser.parse_args()

    # ── Resolve batch index ────────────────────────────────────────────────────
    if args.batch_idx is not None:
        batch_idx = args.batch_idx
    else:
        batch_idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

    ckpt_path      = Path(args.checkpoint)
    meta_ckpt_path = Path(args.meta_checkpoint)
    cif_dir        = Path(args.cif_dir)
    xform_cfg_path = Path(args.xform_config)
    output_dir     = Path(args.output_dir)
    batches_dir    = output_dir / "batches"
    cache_dir      = output_dir / "graph_cache"

    batches_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{batch_idx:04d}"

    print("=" * 65)
    print("full_library_inference.py  —  Task 1.1c")
    print(f"  Batch idx     : {batch_idx} / {args.n_batches - 1}")
    print(f"  Checkpoint    : {ckpt_path.name}")
    print(f"  Meta ckpt     : {meta_ckpt_path.name}")
    print(f"  CIF dir       : {cif_dir}")
    print(f"  Output dir    : {output_dir}")
    print(f"  Batch size    : {args.batch_size}")
    print(f"  Max atoms     : {args.max_atoms}")
    print(f"  Start time    : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # ── Resume check ──────────────────────────────────────────────────────────
    out_npz = batches_dir / f"batch_{tag}_features.npz"
    out_csv = batches_dir / f"batch_{tag}_predictions.csv"
    if out_npz.exists() and out_csv.exists():
        print(f"\n[SKIP] Output already exists: {out_npz.name} — skipping batch {batch_idx}.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── 1. Load model ──────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading model...")
    model, norm_mean, norm_std, cfg = load_model(ckpt_path, meta_ckpt_path, device)

    # ── 2. Identify embedding layer ────────────────────────────────────────────
    print(f"\n[2/5] Identifying embedding layer...")
    hook_layer, hook_layer_name = find_embedding_layer(model)

    # ── 3. Load transform config ───────────────────────────────────────────────
    print(f"\n[3/5] Loading transform config...")
    with open(xform_cfg_path) as f:
        xform_cfg = json.load(f)
    print(f"  Transform config: {xform_cfg_path.name} ({len(xform_cfg)} entries)")

    # ── 4. Scan CIF directory and split into batches ───────────────────────────
    print(f"\n[4/5] Scanning CIF directory and building graphs...")
    all_cif_files = sorted(cif_dir.glob("*.cif"))
    all_mol_ids   = [f.stem for f in all_cif_files]
    n_total       = len(all_mol_ids)
    print(f"  Total CIF files found: {n_total}")

    batch_ids = get_batch_ids(all_mol_ids, args.n_batches, batch_idx)
    print(f"  This batch: [{batch_idx}] → {len(batch_ids)} MOFs "
          f"(IDs {batch_ids[0]} ... {batch_ids[-1]})")

    # Graph cache path (None disables cache entirely)
    graph_cache_path = (
        None if args.no_cache
        else cache_dir / f"batch_{tag}_graphs.pkl"
    )

    valid_ids, graphs, records, failed_ids = build_batch_graphs(
        batch_ids, cif_dir, args.max_atoms, graph_cache_path,
    )

    # Save failed MOFs log
    failed_path = output_dir / f"failed_mofs_{tag}.txt"
    with open(failed_path, "w") as f:
        f.write(f"# Batch {batch_idx} — failed MOFs ({len(failed_ids)} total)\n")
        f.write("# Format: mol_id <TAB> reason [<TAB> detail]\n")
        for line in failed_ids:
            f.write(line + "\n")
    print(f"  Failed MOFs logged: {failed_path} ({len(failed_ids)} entries)")

    if not valid_ids:
        print("  No valid MOFs in this batch — exiting early.")
        return

    # ── 5. Build StructureDataset and run inference ────────────────────────────
    print(f"\n[5/5] Running inference + extracting embeddings...")
    from alignn.graphs import StructureDataset

    # Use full records (jid + atoms + target) required by StructureDataset
    df_records = pd.DataFrame(records)
    dataset = StructureDataset(
        df_records,
        graphs,
        target="target",
        atom_features="cgcnn",
        line_graph=True,
        id_tag="jid",
    )

    preds_xform, embeddings = run_inference_with_embeddings(
        model, dataset, norm_mean, norm_std, hook_layer, device,
        args.batch_size, len(valid_ids),
    )

    # Invert transforms → physical space
    preds_orig = invert_targets(preds_xform, xform_cfg)

    # Sanity checks
    n_nan_emb  = int(np.sum(~np.isfinite(embeddings)))
    n_nan_pred = int(np.sum(~np.isfinite(preds_orig)))
    print(f"  NaN/Inf in embeddings  : {n_nan_emb}")
    print(f"  NaN/Inf in predictions : {n_nan_pred}")

    # ── Save outputs ───────────────────────────────────────────────────────────
    np.savez_compressed(
        out_npz,
        features=embeddings,
        mol_ids=np.array(valid_ids, dtype=str),
    )
    print(f"\n  Saved: {out_npz}  (shape={embeddings.shape})")

    pred_df = pd.DataFrame(preds_orig, columns=TARGET_COLS)
    pred_df.insert(0, "mof_id", valid_ids)
    pred_df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"DONE — batch {batch_idx}")
    print(f"  Valid MOFs     : {len(valid_ids)}")
    print(f"  Failed MOFs    : {len(failed_ids)}")
    print(f"  Embeddings     : {embeddings.shape}")
    print(f"  Finish time    : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
