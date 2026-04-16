"""Compatibility shim for legacy UQ plotting imports.

Canonical plotting implementations now live in ``src.figures.fig_uq_validation``.
New code should import from that module directly.
"""

from __future__ import annotations

from src.figures.fig_uq_validation import (
    plot_calibration,
    plot_distribution_panel,
    plot_k_sweep,
    plot_sr_analysis,
)

__all__ = [
    "plot_calibration",
    "plot_distribution_panel",
    "plot_k_sweep",
    "plot_sr_analysis",
]
