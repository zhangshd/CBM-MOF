"""Reusable annotation-layout helpers for scatter/parity panels."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def estimate_annotation_box_fraction(
    panel_width_inch: float,
    panel_height_inch: float,
    font_size_pt: float,
    *,
    n_lines: int = 3,
    max_line_chars: int = 12,
    text_pad_pt: float = 4.0,
    char_width_factor: float = 0.56,
    line_spacing: float = 1.15,
) -> tuple[float, float]:
    """Estimate the occupied annotation box in axes-fraction coordinates."""
    font_height_inch = font_size_pt / 72.0
    text_pad_inch = text_pad_pt / 72.0
    char_width_inch = font_height_inch * char_width_factor
    line_height_inch = font_height_inch * line_spacing

    box_width = (max_line_chars * char_width_inch + 2.0 * text_pad_inch) / panel_width_inch
    box_height = (n_lines * line_height_inch + 2.0 * text_pad_inch) / panel_height_inch

    return (
        min(0.45, max(0.18, box_width)),
        min(0.40, max(0.16, box_height)),
    )


def build_corner_annotation_candidates(
    *,
    panel_width_inch: float,
    panel_height_inch: float,
    font_size_pt: float,
    n_lines: int = 3,
    max_line_chars: int = 12,
    pad_x: float = 0.04,
    pad_y: float = 0.04,
    outer_expand_factor: float = 0.35,
    min_outer_expand: float = 0.10,
) -> list[dict[str, float | str | tuple[float, float, float, float]]]:
    """Create four corner candidates from panel geometry and text metrics."""
    box_width, box_height = estimate_annotation_box_fraction(
        panel_width_inch,
        panel_height_inch,
        font_size_pt,
        n_lines=n_lines,
        max_line_chars=max_line_chars,
    )
    outer_expand_x = max(min_outer_expand, outer_expand_factor * box_width)
    outer_expand_y = max(min_outer_expand, outer_expand_factor * box_height)

    return [
        {
            "name": "upper left",
            "x": pad_x,
            "y": 1.0 - pad_y,
            "ha": "left",
            "va": "top",
            "core_box": (0.0, box_width, 1.0 - box_height, 1.0),
            "outer_box": (
                0.0,
                min(1.0, box_width + outer_expand_x),
                max(0.0, 1.0 - box_height - outer_expand_y),
                1.0,
            ),
        },
        {
            "name": "upper right",
            "x": 1.0 - pad_x,
            "y": 1.0 - pad_y,
            "ha": "right",
            "va": "top",
            "core_box": (1.0 - box_width, 1.0, 1.0 - box_height, 1.0),
            "outer_box": (
                max(0.0, 1.0 - box_width - outer_expand_x),
                1.0,
                max(0.0, 1.0 - box_height - outer_expand_y),
                1.0,
            ),
        },
        {
            "name": "lower left",
            "x": pad_x,
            "y": pad_y,
            "ha": "left",
            "va": "bottom",
            "core_box": (0.0, box_width, 0.0, box_height),
            "outer_box": (
                0.0,
                min(1.0, box_width + outer_expand_x),
                0.0,
                min(1.0, box_height + outer_expand_y),
            ),
        },
        {
            "name": "lower right",
            "x": 1.0 - pad_x,
            "y": pad_y,
            "ha": "right",
            "va": "bottom",
            "core_box": (1.0 - box_width, 1.0, 0.0, box_height),
            "outer_box": (
                max(0.0, 1.0 - box_width - outer_expand_x),
                1.0,
                0.0,
                min(1.0, box_height + outer_expand_y),
            ),
        },
    ]


def choose_annotation_anchor(
    x: np.ndarray,
    y: np.ndarray,
    limits: tuple[float, float],
    *,
    candidates: list[dict[str, float | str | tuple[float, float, float, float]]] | None = None,
    preferred_name: str | None = None,
    outer_weight: float = 0.35,
) -> dict[str, float | str | tuple[float, float, float, float]]:
    """Select the least occupied corner candidate for a single panel."""
    if candidates is None:
        raise ValueError("candidates must be provided for annotation placement")

    if preferred_name is not None:
        for candidate in candidates:
            if candidate["name"] == preferred_name:
                return candidate
        raise ValueError(f"Unknown preferred annotation anchor: {preferred_name}")

    lower, upper = limits
    span = upper - lower if upper > lower else 1.0
    x_norm = (x - lower) / span
    y_norm = (y - lower) / span

    best = None
    best_score = None
    for candidate in candidates:
        cx0, cx1, cy0, cy1 = candidate["core_box"]  # type: ignore[index]
        ox0, ox1, oy0, oy1 = candidate["outer_box"]  # type: ignore[index]
        core_count = np.count_nonzero(
            (x_norm >= cx0) & (x_norm <= cx1) & (y_norm >= cy0) & (y_norm <= cy1)
        )
        outer_count = np.count_nonzero(
            (x_norm >= ox0) & (x_norm <= ox1) & (y_norm >= oy0) & (y_norm <= oy1)
        )
        score = core_count + outer_weight * outer_count
        if best_score is None or score < best_score:
            best = candidate
            best_score = score

    assert best is not None
    return best


def choose_common_annotation_anchor(
    panels: Iterable[tuple[np.ndarray, np.ndarray, tuple[float, float]]],
    *,
    candidates: list[dict[str, float | str | tuple[float, float, float, float]]] | None = None,
    outer_weight: float = 0.35,
) -> str:
    """Choose one common corner for a set of scatter/parity panels."""
    if candidates is None:
        raise ValueError("candidates must be provided for common anchor selection")

    aggregate_scores = {candidate["name"]: 0.0 for candidate in candidates}
    for x, y, limits in panels:
        lower, upper = limits
        span = upper - lower if upper > lower else 1.0
        x_norm = (x - lower) / span
        y_norm = (y - lower) / span
        for candidate in candidates:
            cx0, cx1, cy0, cy1 = candidate["core_box"]  # type: ignore[index]
            ox0, ox1, oy0, oy1 = candidate["outer_box"]  # type: ignore[index]
            core_count = np.count_nonzero(
                (x_norm >= cx0) & (x_norm <= cx1) & (y_norm >= cy0) & (y_norm <= cy1)
            )
            outer_count = np.count_nonzero(
                (x_norm >= ox0) & (x_norm <= ox1) & (y_norm >= oy0) & (y_norm <= oy1)
            )
            aggregate_scores[candidate["name"]] += core_count + outer_weight * outer_count
    return min(aggregate_scores, key=aggregate_scores.get)
