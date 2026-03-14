"""
plot_isotherm_fits.py — Task 3.1d: Generate isotherm fit verification plots.

For each MOF, plots experimental data points + fitted Langmuir curve
for CH4 and N2 on a single 2-panel figure.

Usage:
    python src/alignn/plot_isotherm_fits.py
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIT_DIR = (
    REPO_ROOT / "results" / "alignn" / "model_ep150"
    / "bkt_candidates" / "isotherm_fits"
)

# Publication style (optional: import from src/figures/style.py if available)
try:
    import sys
    sys.path.insert(0, str(REPO_ROOT / "src" / "figures"))
    from style import apply_style
    apply_style()
except Exception:
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "figure.dpi": 150,
    })


GAS_DISPLAY = {"methane": "CH₄", "N2": "N₂"}
GAS_COLORS = {"methane": "#d62728", "N2": "#1f77b4"}


def langmuir_loading(p: np.ndarray, K: float, n_m: float) -> np.ndarray:
    """Langmuir isotherm: q = n_m * K * p / (1 + K * p)."""
    return n_m * K * p / (1.0 + K * p)


def plot_mof(
    mof_name: str,
    exp_data: pd.DataFrame,
    fit_params: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Create a 2-panel (CH4 + N2) isotherm fit plot for one MOF."""
    gases = sorted(exp_data["GasName"].unique())
    n_panels = len(gases)

    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 3.5))
    if n_panels == 1:
        axes = [axes]

    for ax, gas in zip(axes, gases):
        sub_exp = exp_data[exp_data["GasName"] == gas].sort_values("Pressure[bar]")
        pressures = sub_exp["Pressure[bar]"].values
        loadings = sub_exp["AbsLoading"].values

        # Get fit parameters
        sub_fit = fit_params[fit_params["GasName"] == gas]
        if sub_fit.empty:
            ax.set_title(f"{GAS_DISPLAY.get(gas, gas)} — NO FIT")
            continue
        row = sub_fit.iloc[0]
        K = row["K"]
        n_m = row["n_m"]
        r2 = row["R2"]

        # Generate smooth fitted curve
        p_smooth = np.linspace(0, pressures.max() * 1.05, 200)
        q_smooth = langmuir_loading(p_smooth, K, n_m)

        color = GAS_COLORS.get(gas, "gray")
        ax.scatter(pressures, loadings, s=40, color=color, zorder=3,
                   edgecolors="k", linewidths=0.5, label="GCMC")
        ax.plot(p_smooth, q_smooth, "-", color=color, linewidth=1.5,
                label=f"Langmuir (R²={r2:.4f})")

        ax.set_xlabel("Pressure (bar)")
        ax.set_ylabel("Loading (mol/kg)")
        ax.set_title(GAS_DISPLAY.get(gas, gas))
        ax.legend(fontsize=8, loc="lower right")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    # Shorten MOF name for title
    short_name = mof_name if len(mof_name) < 45 else mof_name[:42] + "..."
    fig.suptitle(short_name, fontsize=10, fontweight="bold", y=1.02)
    fig.tight_layout()

    safe_name = mof_name.replace("/", "_").replace(" ", "_")
    out_path = output_dir / f"isotherm_fit_{safe_name}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 3.1d: Generate isotherm fit verification plots."
    )
    parser.add_argument(
        "--fit-dir", type=str, default=str(DEFAULT_FIT_DIR),
        help="Directory containing fit results.",
    )
    args = parser.parse_args()

    fit_dir = Path(args.fit_dir)
    merged_csv = fit_dir / "pure_component_data_merged.csv"
    best_csv = fit_dir / "best_isotherm_fits.csv"
    plot_dir = fit_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    exp_data = pd.read_csv(merged_csv)
    fit_params = pd.read_csv(best_csv)

    mof_names = sorted(fit_params["MofName"].unique())
    print(f"Generating plots for {len(mof_names)} MOFs → {plot_dir}")

    for mof in mof_names:
        mof_exp = exp_data[exp_data["MofName"] == mof]
        mof_fit = fit_params[fit_params["MofName"] == mof]
        out = plot_mof(mof, mof_exp, mof_fit, plot_dir)
        print(f"  {mof}: {out.name}")

    print(f"\nDone. {len(mof_names)} plots saved to {plot_dir}")


if __name__ == "__main__":
    main()
