"""
Shared utilities for CBM-MOF experiment scripts.

Usage
-----
from utils import add_test_arg, resolve_output_dir, resolve_data_dir, sbatch_submit, setup_matplotlib

All output paths should be resolved through resolve_output_dir() / resolve_data_dir()
so that --test mode writes to results/test_run/** instead of production paths.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root (two levels up from src/experiments/)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]

# Nature-journal compatible color palette used across all experiment scripts
NATURE_COLORS = {
    "blue":    "#0173B2",
    "orange":  "#DE8F05",
    "green":   "#029E73",
    "red":     "#CC78BC",
    "cyan":    "#56B4E9",
    "magenta": "#CA9161",
    "yellow":  "#ECE133",
    "purple":  "#949494",
}


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def add_test_arg(parser: argparse.ArgumentParser) -> None:
    """Add --test flag to an argument parser."""
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help=(
            "Run in test mode: all output files are written to "
            "results/test_run/** instead of production directories. "
            "SLURM sbatch calls are replaced with dry-run echo statements."
        ),
    )


def parse_args_with_test(description: str = "") -> argparse.Namespace:
    """Create a minimal parser with only --test and return parsed args."""
    parser = argparse.ArgumentParser(description=description)
    add_test_arg(parser)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_output_dir(test_mode: bool, subdir: str) -> Path:
    """
    Return the output directory for figures / derived data.

    In normal mode  → REPO_ROOT/results/<subdir>
    In test mode    → REPO_ROOT/results/test_run/<subdir>

    The directory is created automatically (parents=True).
    """
    if test_mode:
        base = REPO_ROOT / "results" / "test_run"
    else:
        base = REPO_ROOT / "results"
    out = base / subdir
    out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_data_dir(test_mode: bool, subdir: str) -> Path:
    """
    Return a data **output** directory.

    Use this only for writing derived / intermediate data files.
    Input data paths should always point to the production ``data/`` directory
    regardless of test mode.

    In normal mode  → REPO_ROOT/data/<subdir>
    In test mode    → REPO_ROOT/results/test_run/data/<subdir>
    """
    if test_mode:
        base = REPO_ROOT / "results" / "test_run" / "data"
    else:
        base = REPO_ROOT / "data"
    out = base / subdir
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# SLURM helpers
# ---------------------------------------------------------------------------

def sbatch_submit(script_path: str | Path, test_mode: bool, cwd: str | Path | None = None) -> None:
    """
    Submit a SLURM job script via sbatch, or print a dry-run message in test mode.

    Parameters
    ----------
    script_path : path to the SLURM shell script
    test_mode   : if True, print instead of calling sbatch
    cwd         : working directory for the sbatch call
    """
    script_path = Path(script_path)
    if test_mode:
        print(f"[DRY-RUN] would submit: sbatch {script_path}")
        return
    cmd = ["sbatch", str(script_path)]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
    )
    if result.returncode != 0:
        print(f"[ERROR] sbatch failed for {script_path}:\n{result.stderr}", file=sys.stderr)
    else:
        print(f"[SUBMITTED] {result.stdout.strip()}")


# ---------------------------------------------------------------------------
# Matplotlib helpers
# ---------------------------------------------------------------------------

def setup_matplotlib() -> None:
    """
    Configure matplotlib for headless (non-interactive) publication-quality output.
    Must be called before any import of matplotlib.pyplot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["figure.titlesize"] = 12
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["grid.linewidth"] = 0.5
    plt.rcParams["lines.linewidth"] = 1.5
    plt.rcParams["patch.linewidth"] = 0.5
    plt.rcParams["xtick.major.width"] = 1.0
    plt.rcParams["ytick.major.width"] = 1.0
    plt.rcParams["xtick.major.size"] = 4
    plt.rcParams["ytick.major.size"] = 4
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.pad_inches"] = 0.1


def apply_nature_axes(ax) -> None:
    """Apply Nature-journal spine/tick style to a matplotlib Axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", which="major", width=1.0, length=4)
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)


def savefig(fig, path: Path, close: bool = True) -> None:
    """Save figure to *path* at 300 dpi and optionally close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[SAVED] {path}")
    if close:
        import matplotlib.pyplot as plt
        plt.close(fig)
