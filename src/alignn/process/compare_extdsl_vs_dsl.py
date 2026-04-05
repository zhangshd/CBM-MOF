"""Compare extDSL vs DSL NSGA-II optimization outcomes for overlapping MOFs.

For each overlapping material (present in both Results/ and Results/archive_DSL/),
parse the last generation's feasible Pareto front and summarize objective ranges,
knee-point, and dominated-area metrics.

Objectives (both minimized by NSGA-II):
  - Obj1 = -productivity [mol CH4 / (kg ads · h)] -> we report productivity = -Obj1
  - Obj2 = energy_requirements [kWh / kg CH4]
Constraint: purity >= 0.44 (enforced via Cons3).

Output:
  - extdsl_vs_dsl_comparison.csv : per-material summary statistics
  - extdsl_vs_dsl_pareto.png     : Pareto front overlays (one subplot per material)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path("/home/zhangsd/repos/CBM-MOF")
sys.path.insert(0, str(_REPO / "src"))

from alignn.process.parse_nsga2_results import (  # noqa: E402
    parse_result_file,
    filter_feasible,
    compute_pareto_front,
)

_EXT_DIR = _REPO / "src" / "SuperPSA" / "Results"
_DSL_DIR = _REPO / "src" / "SuperPSA" / "Results" / "archive_DSL"
_OUT_DIR = Path("/home/zhangsd/repos/CBM-MOF-paper/notes/research/extdsl_vs_dsl_outputs")
_OUT_DIR.mkdir(exist_ok=True, parents=True)

_FNAME_RE = re.compile(
    r"^opt_(?P<mode>PSA|VSA)_HR_\d{3}_(?P<name>.+?)_\d{8}_\d{6}\.txt$"
)


def index_files(directory: Path) -> dict[tuple[str, str], Path]:
    """Return mapping (mode, material_name) -> file path."""
    out: dict[tuple[str, str], Path] = {}
    for f in directory.glob("opt_*_HR_*.txt"):
        m = _FNAME_RE.match(f.name)
        if not m:
            continue
        key = (m.group("mode"), m.group("name"))
        out[key] = f
    return out


def extract_pareto(filepath: Path) -> pd.DataFrame | None:
    parsed = parse_result_file(filepath, min_gen=10)
    if parsed is None:
        return None
    pop = parsed["population"]
    feas = filter_feasible(pop)
    if feas.empty:
        return None
    pareto = compute_pareto_front(feas)
    # Convert to physically intuitive metrics
    pareto = pareto.assign(
        productivity=-pareto["Obj1"],
        energy=pareto["Obj2"],
    )
    return pareto


def summarize(pareto: pd.DataFrame) -> dict:
    prod = pareto["productivity"].values
    ener = pareto["energy"].values
    # Sort by productivity ascending
    order = np.argsort(prod)
    prod_s = prod[order]
    ener_s = ener[order]
    # Knee point: min Euclidean distance to utopia after normalization
    if len(prod_s) >= 2:
        p_norm = (prod_s - prod_s.min()) / (prod_s.max() - prod_s.min() + 1e-12)
        e_norm = (ener_s - ener_s.min()) / (ener_s.max() - ener_s.min() + 1e-12)
        # Utopia = max productivity (1), min energy (0)
        dist = np.sqrt((1 - p_norm) ** 2 + e_norm ** 2)
        knee = int(np.argmin(dist))
    else:
        knee = 0
    return {
        "n_pareto": len(pareto),
        "prod_max": float(prod.max()),
        "prod_min": float(prod.min()),
        "energy_min": float(ener.min()),
        "energy_max": float(ener.max()),
        "knee_prod": float(prod_s[knee]),
        "knee_energy": float(ener_s[knee]),
    }


def main() -> None:
    ext_files = index_files(_EXT_DIR)
    dsl_files = index_files(_DSL_DIR)
    common = sorted(set(ext_files) & set(dsl_files))
    print(f"Overlap: {len(common)} (mode, material) pairs")
    only_ext = sorted(set(ext_files) - set(dsl_files))
    only_dsl = sorted(set(dsl_files) - set(ext_files))
    print(f"Only extDSL: {len(only_ext)}  Only DSL: {len(only_dsl)}")
    for x in only_ext:
        print(f"  ext-only: {x}")
    for x in only_dsl:
        print(f"  dsl-only: {x}")

    rows = []
    pareto_data: dict = {}
    for key in common:
        mode, name = key
        ext_p = extract_pareto(ext_files[key])
        dsl_p = extract_pareto(dsl_files[key])
        if ext_p is None or dsl_p is None:
            print(f"SKIP {key}: missing feasible pareto")
            continue
        pareto_data[key] = (dsl_p, ext_p)
        ext_s = summarize(ext_p)
        dsl_s = summarize(dsl_p)
        row = {
            "mode": mode,
            "material": name,
            **{f"dsl_{k}": v for k, v in dsl_s.items()},
            **{f"ext_{k}": v for k, v in ext_s.items()},
            "delta_prod_max_pct": 100 * (ext_s["prod_max"] - dsl_s["prod_max"]) / dsl_s["prod_max"],
            "delta_energy_min_pct": 100 * (ext_s["energy_min"] - dsl_s["energy_min"]) / dsl_s["energy_min"],
            "delta_knee_prod_pct": 100 * (ext_s["knee_prod"] - dsl_s["knee_prod"]) / dsl_s["knee_prod"],
            "delta_knee_energy_pct": 100 * (ext_s["knee_energy"] - dsl_s["knee_energy"]) / dsl_s["knee_energy"],
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["mode", "material"])
    out_csv = _OUT_DIR / "extdsl_vs_dsl_comparison.csv"
    df.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"Saved: {out_csv}  ({len(df)} rows)")
    print()
    print(df.to_string(index=False))

    # Pareto overlay plots
    for mode in ("PSA", "VSA"):
        sub = [(k, v) for k, v in pareto_data.items() if k[0] == mode]
        if not sub:
            continue
        n = len(sub)
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), squeeze=False)
        for idx, (key, (dsl_p, ext_p)) in enumerate(sub):
            ax = axes[idx // ncols][idx % ncols]
            ax.scatter(dsl_p["productivity"], dsl_p["energy"], s=12, c="tab:blue", alpha=0.7, label="DSL")
            ax.scatter(ext_p["productivity"], ext_p["energy"], s=12, c="tab:red", alpha=0.7, label="extDSL")
            ax.set_xlabel("Productivity [mol/(kg·h)]", fontsize=8)
            ax.set_ylabel("Energy [kWh/kg CH4]", fontsize=8)
            ax.set_title(key[1][:40], fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_yscale("log")
            ax.legend(fontsize=7, loc="best")
            ax.grid(alpha=0.3)
        for idx in range(n, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)
        fig.suptitle(f"{mode}-HR Pareto fronts: DSL vs Extended DSL", fontsize=11)
        fig.tight_layout()
        out_png = _OUT_DIR / f"extdsl_vs_dsl_pareto_{mode}.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
