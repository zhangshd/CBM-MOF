"""
compute_uq.py
=============
ALIGNN Uncertainty Quantification via Latent Space Variance (LSV) using faiss.

Algorithm (LSV):
  1. Extract latent embeddings (dim=256) from the last hidden layer before the
     output FC layers by registering a forward hook.
  2. Build a faiss L2 index from training set embeddings.
  3. For each query MOF, find k=5 nearest training neighbors, compute the
     Gaussian-weighted variance of their labels in original physical space.
  4. Calibrate on val/test: compute Spearman ρ(LSV, |pred_error|) per target.
  5. Apply to Top-100 PSA/VSA screening candidates.

Usage:
    # Use the 50ep symlog_1e-3 model to validate the UQ code:
    CUDA_VISIBLE_DEVICES=0 python src/alignn/compute_uq.py \\
        --checkpoint results/alignn/short_50ep_symlog_1e-3/best_model.pt \\
        --data-dir data/alignn_symlog_1e-3 \\
        --output-dir results/alignn/short_50ep_symlog_1e-3

    # After 500ep training, switch checkpoint:
    CUDA_VISIBLE_DEVICES=0 python src/alignn/compute_uq.py \\
        --checkpoint results/alignn/500ep_symlog_1e-3_ddp2g/best_model.pt \\
        --data-dir data/alignn_symlog_1e-3 \\
        --output-dir results/alignn/500ep_symlog_1e-3_ddp2g
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=UserWarning)

# ── Constants ──────────────────────────────────────────────────────────────────
REPO_ROOT   = Path("/home/zhangsd/repos/CBM-MOF")
INFER_DIR   = REPO_ROOT / "results/cbm_screening/inference"

UPTAKE_COLS = ["AdsCH4_10kPa", "AdsCH4_100kPa", "AdsCH4_1000kPa",
               "AdsN2_10kPa",  "AdsN2_100kPa",  "AdsN2_1000kPa"]
QST_COLS    = ["QstCH4", "QstN2"]
TARGET_COLS = UPTAKE_COLS + QST_COLS
N_TARGETS   = len(TARGET_COLS)

K_NEIGHBORS = 5          # k for k-NN LSV
MIN_SPEARMAN_RHO = 0.3   # calibration target


# ── Transform helpers (shared with evaluate_alignn.py) ────────────────────────

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
    """Invert per-column transforms on an (N, 8) array."""
    out = arr.copy()
    for i, col in enumerate(TARGET_COLS):
        out[:, i] = _inv_col(arr[:, i], xform_cfg.get(col, {"type": "raw"}))
    return out


def load_transform_config(config_path: Path) -> dict:
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    raise FileNotFoundError(f"transform_config.json not found: {config_path}")


# ── Checkpoint loading ─────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path: Path, device: torch.device):
    """Load ALIGNN model + norm_stats from best_model.pt."""
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


# ── Dataset construction ───────────────────────────────────────────────────────

def collate_fn(samples):
    import dgl
    graphs, line_graphs, lattices, labels = map(list, zip(*samples))
    return (
        dgl.batch(graphs),
        dgl.batch(line_graphs),
        torch.stack(lattices),
        torch.stack(labels),
    )


def build_dataset_from_records(records: list, max_atoms: int = 500):
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
        df, graphs, target="target",
        atom_features="cgcnn", line_graph=True, id_tag="jid",
    )
    return dataset, df["jid"].tolist()


def load_split_records(data_dir: Path, split: str, max_atoms: int, xform_cfg: dict):
    """Load train/val/test records from data_dir/{split}/id_prop.csv + cifs/."""
    from jarvis.core.atoms import Atoms

    id_prop_csv = data_dir / split / "id_prop.csv"
    cif_dir     = data_dir / "cifs"

    df = pd.read_csv(id_prop_csv)
    records, mol_ids, labels_raw = [], [], []
    skipped = 0
    for _, row in df.iterrows():
        mol_id   = row["mol_id"]
        cif_path = cif_dir / f"{mol_id}.cif"
        if not cif_path.exists():
            continue
        atoms = Atoms.from_cif(str(cif_path), use_cif2cell=False)
        if max_atoms > 0 and atoms.num_atoms > max_atoms:
            skipped += 1
            continue
        target_vals = [float(row[c]) for c in TARGET_COLS]
        records.append({"jid": mol_id, "atoms": atoms.to_dict(), "target": target_vals})
        mol_ids.append(mol_id)
        labels_raw.append(target_vals)  # in transformed space

    print(f"  [{split}] Loaded: {len(records)} (skipped >{max_atoms} atoms: {skipped})")
    labels_arr = np.array(labels_raw, dtype=np.float32)  # (N, 8) transformed
    return records, mol_ids, labels_arr


# ── Embedding extraction ───────────────────────────────────────────────────────

def find_embedding_layer(model: nn.Module) -> tuple[nn.Module, str]:
    """
    Locate the last hidden FC layer before output in ALIGNN.

    Prints model architecture summary and returns (layer, layer_name).
    Expected: model.fc_layers[0] (input dim=256, output dim=256 or similar).
    """
    print("\n  ALIGNN named modules (fc/output layers):")
    fc_candidates = []
    for name, module in model.named_modules():
        if "fc" in name.lower() or "out" in name.lower():
            print(f"    {name}: {type(module).__name__} — {module}")
            if isinstance(module, nn.Linear):
                fc_candidates.append((name, module))

    if not fc_candidates:
        raise RuntimeError("No FC layers found in model. Check model architecture.")

    # Use the first FC layer (input to output projection)
    layer_name, layer = fc_candidates[0]
    print(f"\n  Selected hook layer: '{layer_name}' ({layer})")
    return layer, layer_name


def extract_embeddings(
    model: nn.Module,
    dataset,
    mol_ids: list,
    device: torch.device,
    batch_size: int = 32,
    layer: nn.Module = None,
) -> np.ndarray:
    """
    Extract latent embeddings using a forward hook on the given layer.

    Returns:
        embeddings: (N, D) float32 numpy array
    """
    captured = {}

    def hook_fn(module, input, output):
        # Capture the input to the FC layer (pre-activation features)
        if isinstance(input, tuple) and len(input) > 0:
            captured["emb"] = input[0].detach().cpu().float()
        else:
            captured["emb"] = input.detach().cpu().float()

    handle = layer.register_forward_hook(hook_fn)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate_fn,
    )

    all_embs = []
    with torch.no_grad():
        for batch in loader:
            g, lg, lat, _ = batch
            g   = g.to(device)
            lg  = lg.to(device)
            lat = lat.to(device)
            _ = model((g, lg, lat))
            all_embs.append(captured["emb"].numpy())

    handle.remove()
    embs = np.vstack(all_embs).astype(np.float32)
    print(f"  Embeddings shape: {embs.shape}")
    return embs


# ── faiss index ───────────────────────────────────────────────────────────────

def build_faiss_index(train_emb: np.ndarray) -> object:
    """
    Build a faiss IndexFlatL2 from training embeddings.
    Falls back to CPU if faiss-gpu is unavailable.
    """
    try:
        import faiss
    except ImportError:
        raise ImportError("faiss not installed. Run: conda install -c conda-forge faiss-gpu")

    d = train_emb.shape[1]
    index_cpu = faiss.IndexFlatL2(d)
    index_cpu.add(train_emb)

    # Try GPU acceleration
    try:
        res   = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
        print(f"  faiss index: GPU (L2, d={d}, n_train={train_emb.shape[0]})")
    except Exception:
        index = index_cpu
        print(f"  faiss index: CPU (L2, d={d}, n_train={train_emb.shape[0]})")

    return index


# ── LSV computation ───────────────────────────────────────────────────────────

def compute_baseline_dist(index, train_emb: np.ndarray, k: int = K_NEIGHBORS) -> float:
    """
    Compute the mean k-NN L2 distance within training set (excluding self).
    Used as a normalization factor for LSV scores.
    """
    import faiss
    # search k+1 to exclude self (distance=0)
    D, _ = index.search(train_emb, k + 1)
    D = D[:, 1:]  # exclude self (index 0)
    return float(D.mean())


def compute_lsv(
    query_emb: np.ndarray,
    train_labels_orig: np.ndarray,
    index,
    baseline_dist: float,
    k: int = K_NEIGHBORS,
    baseline_lsv_mean: np.ndarray = None,
) -> np.ndarray:
    """
    Compute Latent Space Variance (LSV) uncertainty scores.

    The raw LSV is the Gaussian-weighted variance of k-NN training labels:
        LSV_t = sum_i w_i * (y_{t,i} - y_bar_t)^2

    When baseline_lsv_mean is provided, LSV is normalised to a dimensionless
    relative uncertainty score anchored to the training distribution:
        LSV_norm_t = LSV_t / mean(LSV_train_t)

    Semantic interpretation of LSV_norm:
        ≈ 1 : as uncertain as a typical training sample (in-distribution)
        << 1: less uncertain than average (well in-distribution)
        >> 1: far more uncertain than training baseline (OOD signal)

    Args:
        query_emb:         (N, D) float32 query embeddings
        train_labels_orig: (M, T) float32 training labels in original space
        index:             faiss index containing training embeddings
        baseline_dist:     mean k-NN L2 distance in training set (normalises Gaussian kernel)
        k:                 number of nearest neighbors
        baseline_lsv_mean: (T,) per-target mean raw LSV of training set (optional).
                           If provided, output is dimensionless LSV_norm = LSV / mean(LSV_train).
                           Compute by calling this function on training embeddings first
                           (without baseline_lsv_mean), then taking .mean(axis=0).

    Returns:
        lsv: (N, T) LSV scores; dimensionless relative uncertainty if baseline_lsv_mean is given.
    """
    D, I = index.search(query_emb, k)   # D: (N, k) L2 distances, I: (N, k) indices

    # Gaussian weights: w_i = exp(-d²_i / baseline_dist²)
    # Normalize so weights sum to 1 per query
    sigma2 = max(baseline_dist ** 2, 1e-8)
    w = np.exp(-D / sigma2)              # (N, k)
    w = w / (w.sum(axis=1, keepdims=True) + 1e-12)   # normalize rows

    neighbor_labels = train_labels_orig[I]  # (N, k, T)

    # Weighted mean
    weighted_mean = (w[:, :, None] * neighbor_labels).sum(axis=1)  # (N, T)

    # Weighted variance
    diff = neighbor_labels - weighted_mean[:, None, :]  # (N, k, T)
    lsv  = (w[:, :, None] * diff ** 2).sum(axis=1)      # (N, T)

    # Normalise by per-target mean training LSV → dimensionless relative uncertainty
    # LSV_norm_t = LSV_t / mean(LSV_train_t); training samples average ≈ 1.0
    if baseline_lsv_mean is not None:
        lsv = lsv / (baseline_lsv_mean + 1e-12)

    return lsv


# ── Calibration ───────────────────────────────────────────────────────────────

def calibrate_uq(
    lsv: np.ndarray,
    pred_orig: np.ndarray,
    true_orig: np.ndarray,
) -> dict:
    """
    Compute Spearman ρ(LSV, |pred_error|) per target.

    Returns dict: {task_name: {"rho": float, "pval": float, "n": int}}
    """
    results = {}
    for i, col in enumerate(TARGET_COLS):
        lsv_vals  = lsv[:, i]
        abs_error = np.abs(pred_orig[:, i] - true_orig[:, i])
        mask      = np.isfinite(lsv_vals) & np.isfinite(abs_error)
        n         = mask.sum()
        if n < 10:
            results[col] = {"rho": None, "pval": None, "n": int(n)}
            continue
        rho, pval = spearmanr(lsv_vals[mask], abs_error[mask])
        results[col] = {"rho": float(rho), "pval": float(pval), "n": int(n)}

    # Summary
    valid_rhos = [v["rho"] for v in results.values() if v["rho"] is not None]
    n_pass = sum(1 for r in valid_rhos if r > MIN_SPEARMAN_RHO)
    print(f"\n  Calibration (Spearman ρ, threshold={MIN_SPEARMAN_RHO}):")
    for col, m in results.items():
        if m["rho"] is not None:
            flag = "✅" if m["rho"] > MIN_SPEARMAN_RHO else "❌"
            print(f"    {flag} {col:25s}  ρ={m['rho']:+.3f}  (p={m['pval']:.2e}, n={m['n']})")
        else:
            print(f"    ❓ {col:25s}  n={m['n']} (too few samples)")
    print(f"\n  Passed: {n_pass}/{len(valid_rhos)} tasks with ρ > {MIN_SPEARMAN_RHO}")

    return results


def plot_calibration(
    lsv: np.ndarray,
    pred_orig: np.ndarray,
    true_orig: np.ndarray,
    calib_results: dict,
    out_path: Path,
) -> None:
    """Generate 2×4 scatter plots: LSV vs |error| per target."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    for i, col in enumerate(TARGET_COLS):
        ax  = axes[i]
        lv  = lsv[:, i]
        err = np.abs(pred_orig[:, i] - true_orig[:, i])
        mask = np.isfinite(lv) & np.isfinite(err)
        lv_m, err_m = lv[mask], err[mask]

        ax.scatter(lv_m, err_m, s=6, alpha=0.4, edgecolors="none", color="#2196F3")
        ax.set_xlabel("LSV score", fontsize=8)
        ax.set_ylabel("|Prediction error|", fontsize=8)
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)

        m   = calib_results.get(col, {})
        rho = m.get("rho")
        ann = f"ρ={rho:+.3f}" if rho is not None else "N/A"
        ax.text(0.97, 0.03, ann, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    fig.suptitle("ALIGNN UQ Calibration — LSV vs |Prediction Error|", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Calibration plot saved: {out_path}")


# ── Inference helper ───────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference_raw(model, dataset, norm_mean, norm_std, device, batch_size=16) -> np.ndarray:
    """Run inference; return predictions in z-score space (before inversion)."""
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate_fn,
    )
    preds = []
    for batch in loader:
        g, lg, lat, _ = batch
        out = model((g.to(device), lg.to(device), lat.to(device))).cpu().numpy()
        preds.append(np.atleast_2d(out))
    preds_zscore = np.vstack(preds)  # (N, 8) z-score
    # Invert z-score to transformed space
    return preds_zscore * norm_std + norm_mean  # (N, 8) transformed space


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint",  type=str,
        default=str(REPO_ROOT / "results/alignn/short_50ep_symlog_1e-3/best_model.pt"))
    parser.add_argument("--data-dir",    type=str,
        default=str(REPO_ROOT / "data/alignn_symlog_1e-3"))
    parser.add_argument("--output-dir",  type=str,
        default=str(REPO_ROOT / "results/alignn/short_50ep_symlog_1e-3"))
    parser.add_argument("--max-atoms",   type=int, default=500)
    parser.add_argument("--batch-size",  type=int, default=32)
    parser.add_argument("--k",           type=int, default=K_NEIGHBORS)
    parser.add_argument("--skip-top100", action="store_true",
        help="Skip Top-100 PSA/VSA uncertainty scoring (faster for debugging)")
    args = parser.parse_args()

    ckpt_path  = Path(args.checkpoint)
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint : {ckpt_path}")
    print(f"Data dir   : {data_dir}")
    print(f"Output dir : {output_dir}")
    print(f"Max atoms  : {args.max_atoms}")
    print(f"k-NN       : {args.k}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")

    # ── Step 1: Load model ─────────────────────────────────────────────────────
    print("\n[1/6] Loading checkpoint...")
    model, norm_mean, norm_std, cfg = load_checkpoint(ckpt_path, device)

    # ── Step 2: Identify hook layer ────────────────────────────────────────────
    print("\n[2/6] Identifying embedding layer...")
    hook_layer, hook_layer_name = find_embedding_layer(model)

    # ── Step 3: Load transform config ─────────────────────────────────────────
    print("\n[3/6] Loading transform config...")
    xform_cfg = load_transform_config(data_dir / "transform_config.json")
    print(f"  Transform config loaded ({len(xform_cfg)} entries)")

    # ── Step 4: Extract embeddings for train/val/test ──────────────────────────
    print("\n[4/6] Extracting latent embeddings...")

    split_data = {}
    for split in ("train", "val", "test"):
        records, mol_ids, labels_xform = load_split_records(
            data_dir, split, args.max_atoms, xform_cfg
        )
        dataset, mol_ids = build_dataset_from_records(records, args.max_atoms)
        embs = extract_embeddings(model, dataset, mol_ids, device,
                                  batch_size=args.batch_size, layer=hook_layer)
        # Invert labels to original physical space
        labels_orig = invert_targets(labels_xform, xform_cfg)
        # Get predictions in transformed space → invert to original
        preds_xform = run_inference_raw(model, dataset, norm_mean, norm_std, device, args.batch_size)
        preds_orig  = invert_targets(preds_xform, xform_cfg)

        split_data[split] = {
            "mol_ids":     mol_ids,
            "embs":        embs,
            "labels_orig": labels_orig,
            "preds_orig":  preds_orig,
        }

    # ── Step 5: Build faiss index on training embeddings ──────────────────────
    print("\n[5/6] Building faiss index...")
    train_emb = split_data["train"]["embs"]
    index     = build_faiss_index(train_emb)

    baseline_dist = compute_baseline_dist(index, train_emb, k=args.k)
    print(f"  Baseline k-NN L2 distance (training set): {baseline_dist:.4f}")

    # ── Step 6: Compute LSV and calibrate on val + test ────────────────────────
    print("\n[6/6] Computing LSV and calibrating...")

    train_labels_orig = split_data["train"]["labels_orig"]

    # Compute LSV for val and test (combined for calibration plot)
    for split_name in ("val", "test"):
        sd = split_data[split_name]
        lsv = compute_lsv(sd["embs"], train_labels_orig, index, baseline_dist, k=args.k)
        sd["lsv"] = lsv

    # Combined val + test calibration
    comb_lsv  = np.vstack([split_data["val"]["lsv"],  split_data["test"]["lsv"]])
    comb_pred = np.vstack([split_data["val"]["preds_orig"],  split_data["test"]["preds_orig"]])
    comb_true = np.vstack([split_data["val"]["labels_orig"], split_data["test"]["labels_orig"]])

    calib_results = calibrate_uq(comb_lsv, comb_pred, comb_true)

    # Save calibration results
    calib_path = output_dir / "uq_calibration.json"
    summary    = {
        "checkpoint": str(ckpt_path),
        "data_dir":   str(data_dir),
        "k":          args.k,
        "baseline_dist": float(baseline_dist),
        "hook_layer": hook_layer_name,
        "n_train":    len(train_emb),
        "embedding_dim": int(train_emb.shape[1]),
        "calibration": calib_results,
        "n_pass_rho_threshold": sum(
            1 for v in calib_results.values()
            if v.get("rho") is not None and v["rho"] > MIN_SPEARMAN_RHO
        ),
    }
    with open(calib_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Calibration results saved: {calib_path}")

    # Calibration plot
    plot_calibration(
        comb_lsv, comb_pred, comb_true, calib_results,
        out_path=output_dir / "uq_calibration.png",
    )

    # ── Optional: Top-100 uncertainty scores ───────────────────────────────────
    if not args.skip_top100:
        print("\n[+] Computing uncertainty for Top-100 PSA/VSA candidates...")
        from jarvis.core.atoms import Atoms

        for label, cif_subdir in [
            ("PSA", "top_100_psa_performers_cifs_ml"),
            ("VSA", "top_100_vsa_performers_cifs_ml"),
        ]:
            cif_dir_top = INFER_DIR / cif_subdir
            if not cif_dir_top.exists():
                print(f"  [skip] {cif_dir_top} not found")
                continue

            records, mol_ids_top = [], []
            for cif_path in sorted(cif_dir_top.glob("*.cif")):
                mol_id = cif_path.stem
                try:
                    atoms = Atoms.from_cif(str(cif_path), use_cif2cell=False)
                except Exception as e:
                    print(f"    [warn] {mol_id}: {e}")
                    continue
                if args.max_atoms > 0 and atoms.num_atoms > args.max_atoms:
                    continue
                records.append({"jid": mol_id, "atoms": atoms.to_dict(),
                                "target": [0.0] * N_TARGETS})
                mol_ids_top.append(mol_id)

            if not records:
                print(f"  [skip] No valid CIFs in {cif_dir_top}")
                continue

            print(f"  Top-100 {label}: {len(records)} MOFs")
            dataset_top, _ = build_dataset_from_records(records, args.max_atoms)
            embs_top = extract_embeddings(model, dataset_top, mol_ids_top, device,
                                         batch_size=args.batch_size, layer=hook_layer)
            lsv_top = compute_lsv(embs_top, train_labels_orig, index, baseline_dist, k=args.k)

            # Save per-MOF LSV scores
            out_df = pd.DataFrame(
                lsv_top,
                columns=[f"lsv_{t}" for t in TARGET_COLS],
                index=mol_ids_top,
            )
            out_df.index.name = "mof_id"
            out_csv = output_dir / f"uncertainty_top100_{label.lower()}.csv"
            out_df.to_csv(out_csv)
            print(f"  Saved: {out_csv}  (shape: {out_df.shape})")

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("UQ SUMMARY")
    print("=" * 60)
    print(f"  Model       : {ckpt_path.name}")
    print(f"  Hook layer  : {hook_layer_name}")
    print(f"  Embed dim   : {train_emb.shape[1]}")
    print(f"  k-NN        : {args.k}")
    print(f"  n_train     : {len(train_emb)}")
    valid_rhos = [(col, v["rho"]) for col, v in calib_results.items() if v.get("rho") is not None]
    n_pass = sum(1 for _, r in valid_rhos if r > MIN_SPEARMAN_RHO)
    print(f"  Calibration : {n_pass}/{len(valid_rhos)} tasks with ρ > {MIN_SPEARMAN_RHO}")
    if valid_rhos:
        mean_rho = np.mean([r for _, r in valid_rhos])
        print(f"  Mean ρ      : {mean_rho:.3f}")
    print(f"\nOutputs:")
    print(f"  {output_dir}/uq_calibration.json")
    print(f"  {output_dir}/uq_calibration.png")
    if not args.skip_top100:
        print(f"  {output_dir}/uncertainty_top100_psa.csv")
        print(f"  {output_dir}/uncertainty_top100_vsa.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
