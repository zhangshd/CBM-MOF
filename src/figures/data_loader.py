"""
Unified data loading for CBM-MOF publication figures.

Each loader returns a DataFrame with columns:
    {Task}_true, {Task}_pred   (physical units: mol/kg or kJ/mol)
and optionally CifId as index.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ── Project root (CBM-MOF repo, not CBM-MOF-paper) ─────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Fallback: if running from CBM-MOF-paper, look for the sibling CBM-MOF repo
if not (PROJECT_ROOT / "results").is_dir():
    _alt = PROJECT_ROOT.parent / "CBM-MOF"
    if (_alt / "results").is_dir():
        PROJECT_ROOT = _alt

# ── Task definitions ────────────────────────────────────────────────────────
TASK_LIST = [
    "AdsCH4_10kPa", "AdsCH4_100kPa", "AdsCH4_1000kPa",
    "AdsN2_10kPa",  "AdsN2_100kPa",  "AdsN2_1000kPa",
    "QstCH4", "QstN2",
]

UPTAKE_TASKS = [t for t in TASK_LIST if t.startswith("Ads")]
QST_TASKS    = [t for t in TASK_LIST if t.startswith("Qst")]

# Symlog file-name mapping (MFT/CGCNN-symlog CSVs use different names)
_SYMLOG_FNAME = {}
for t in UPTAKE_TASKS:
    _SYMLOG_FNAME[t] = f"test_results_symlog{t}_1e3.csv"
for t in QST_TASKS:
    _SYMLOG_FNAME[t] = f"test_results_{t}.csv"


# ── Inverse transforms ──────────────────────────────────────────────────────

def inv_symlog(y: np.ndarray, tau: float = 1e-3) -> np.ndarray:
    """Inverse of symlog(x, tau) = sign(x) * log10(|x|/tau + 1).

    Returns: sign(y) * tau * (10^|y| - 1).
    """
    return np.sign(y) * tau * (10.0 ** np.abs(y) - 1.0)


def inv_log10(y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Inverse of log10(x + eps)."""
    return 10.0 ** y - eps


# ── ALIGNN loader ────────────────────────────────────────────────────────────

ALIGNN_BASE = (PROJECT_ROOT / "results" / "alignn"
               / "500ep_symlog_1e-3_ddp2g" / "evaluation_ep100")


def load_alignn_predictions(split: str = "test") -> pd.DataFrame:
    """Load ALIGNN ep100 predictions (already in physical units).

    Parameters
    ----------
    split : str
        One of 'test', 'top_100_psa', 'top_100_vsa'.

    Returns
    -------
    DataFrame with CifId index and {Task}_true, {Task}_pred columns.
    """
    if split == "test":
        path = ALIGNN_BASE / "test" / "test_predictions_vs_truth.csv"
    elif split == "top_100_psa":
        path = ALIGNN_BASE / "top_100_psa" / "top_100_psa_predictions_vs_truth.csv"
    elif split == "top_100_vsa":
        path = ALIGNN_BASE / "top_100_vsa" / "top_100_vsa_predictions_vs_truth.csv"
    else:
        raise ValueError(f"Unknown split: {split}")

    df = pd.read_csv(path, index_col=0)
    df.index.name = "CifId"
    return df


# ── XGBoost loader ──────────────────────────────────────────────────────────

XGBOOST_BASE = (PROJECT_ROOT / "results" / "ml_models" / "round2"
                / "RAC_and_zeo_features_with_id_prop")


def load_xgboost_predictions() -> pd.DataFrame:
    """Load XGBoost predictions for all 8 tasks (physical units).

    Returns DataFrame with numeric index and {Task}_true, {Task}_pred columns.
    No CifId available for XGBoost.
    """
    frames = {}
    for task in TASK_LIST:
        path = XGBOOST_BASE / task / "test_predicted_XGBRegressor.csv"
        df = pd.read_csv(path)
        frames[f"{task}_true"] = df["GroundTruth"].values
        frames[f"{task}_pred"] = df["Predicted"].values
    # All tasks share the same sample ordering; combine into one DF
    n = len(frames[f"{TASK_LIST[0]}_true"])
    return pd.DataFrame(frames, index=range(n))


# ── MOFTransformer-symlog loader ────────────────────────────────────────────

MFT_BASE = (PROJECT_ROOT / "results" / "moftransformer_models"
            / "ads_qst_ch4_n2_symlog_1e3_seed42_moftransformer_from_pmtransformer"
            / "version_2")


def load_mft_predictions() -> pd.DataFrame:
    """Load MOFTransformer-symlog predictions, inverse-transforming uptakes.

    Uptake CSVs are in symlog(tau=1e-3) space → apply inv_symlog.
    Qst CSVs are already in physical units.
    """
    all_data: dict[str, pd.Series] = {}
    cif_ids = None

    for task in TASK_LIST:
        path = MFT_BASE / _SYMLOG_FNAME[task]
        df = pd.read_csv(path)
        if cif_ids is None:
            cif_ids = df["CifId"].values

        gt = df["GroundTruth"].values
        pred = df["Predicted"].values

        if task in UPTAKE_TASKS:
            gt = inv_symlog(gt)
            pred = inv_symlog(pred)

        all_data[f"{task}_true"] = gt
        all_data[f"{task}_pred"] = pred

    out = pd.DataFrame(all_data)
    if cif_ids is not None:
        out.index = cif_ids
        out.index.name = "CifId"
    return out


# ── CGCNN-log10 loader (SI only) ────────────────────────────────────────────

CGCNN_L10_BASE = (PROJECT_ROOT / "results" / "cgcnn_models"
                  / "ads_qst_ch4_n2_org_seed42_att_cgcnn" / "version_0")


def load_cgcnn_predictions(variant: str = "log10") -> pd.DataFrame:
    """Load CGCNN predictions (physical units for log10 variant).

    Parameters
    ----------
    variant : str
        'log10' (default, SI) or 'symlog' (also SI, different training).
    """
    if variant == "log10":
        base = CGCNN_L10_BASE
        fname_template = "test_results_{task}.csv"
        needs_inv = False
    elif variant == "symlog":
        base = (PROJECT_ROOT / "results" / "cgcnn_models"
                / "ads_qst_ch4_n2_symlog_1e3_seed42_att_cgcnn" / "version_3")
        fname_template = None  # will use _SYMLOG_FNAME
        needs_inv = True
    else:
        raise ValueError(f"Unknown CGCNN variant: {variant}")

    all_data: dict[str, np.ndarray] = {}
    cif_ids = None

    for task in TASK_LIST:
        if variant == "symlog":
            path = base / _SYMLOG_FNAME[task]
        else:
            path = base / f"test_results_{task}.csv"
        df = pd.read_csv(path)
        if cif_ids is None:
            cif_ids = df["CifId"].values

        gt = df["GroundTruth"].values
        pred = df["Predicted"].values

        if needs_inv and task in UPTAKE_TASKS:
            gt = inv_symlog(gt)
            pred = inv_symlog(pred)

        all_data[f"{task}_true"] = gt
        all_data[f"{task}_pred"] = pred

    out = pd.DataFrame(all_data)
    if cif_ids is not None:
        out.index = cif_ids
        out.index.name = "CifId"
    return out


# ── Synthesizability data ────────────────────────────────────────────────────

SYNTH_PATH = PROJECT_ROOT / "results" / "synthesizability" / "synthesizability_screen.csv"


def load_synthesizability() -> pd.DataFrame:
    """Load synthesizability screening results."""
    return pd.read_csv(SYNTH_PATH)


# ── ALIGNN training curves ──────────────────────────────────────────────────

CHECKPOINT_TRENDS_PATH = (PROJECT_ROOT / "results" / "alignn"
                          / "500ep_symlog_1e-3_ddp2g" / "checkpoint_trends"
                          / "checkpoint_trends.csv")


def load_checkpoint_trends() -> pd.DataFrame:
    """Load ALIGNN checkpoint evaluation trends."""
    return pd.read_csv(CHECKPOINT_TRENDS_PATH)


# ── Model comparison JSON ────────────────────────────────────────────────────

def load_r2_matrix(models=None):
    """Build R² matrix from model_comparison.json.

    Returns
    -------
    dict : {model_name: {task: r2_value, ..., "Mean": mean_r2}}
    """
    import json
    json_path = (PROJECT_ROOT.parent / "CBM-MOF-paper" / "results"
                 / "summary" / "model_comparison.json")
    if not json_path.exists():
        # Try same repo
        json_path = PROJECT_ROOT / "results" / "summary" / "model_comparison.json"

    with open(json_path) as f:
        data = json.load(f)

    test_models = data["splits"]["test"]["models"]

    # Name mapping from JSON keys to display names
    name_map = {
        "XGBoost": "XGBoost",
        "CGCNN (log10)": "CGCNN",
        "MFT (symlog)": "MOFTransformer",
        "ALIGNN (symlog1e-3-ep100)": "ALIGNN",
    }

    result = {}
    for json_key, display_name in name_map.items():
        if json_key not in test_models:
            continue
        if models and display_name not in models:
            continue
        m = test_models[json_key]
        r2s = {task: m[task]["R2"] for task in TASK_LIST if task in m}
        r2s["Mean"] = np.mean(list(r2s.values()))
        result[display_name] = r2s

    return result


# ── Utility: compute R² ─────────────────────────────────────────────────────

def r2_score(y_true, y_pred):
    """Compute R² (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot
