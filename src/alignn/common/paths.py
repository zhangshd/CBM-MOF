"""Path helpers for canonical model-result layouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ModelPaths:
    """Canonical directory layout for one ALIGNN model directory."""

    model_dir: Path
    deployment_dir: Path
    uq_dir: Path
    inference_dir: Path
    top_candidates_dir: Path
    process_candidates_dir: Path


def resolve_repo_path(path_str: str | Path) -> Path:
    """Resolve an absolute path or a repo-relative path."""
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_model_paths(model_dir: str | Path) -> ModelPaths:
    """Build canonical model-result subpaths from a model directory."""
    resolved = resolve_repo_path(model_dir)
    return ModelPaths(
        model_dir=resolved,
        deployment_dir=resolved / "deployment",
        uq_dir=resolved / "uq",
        inference_dir=resolved / "full_library_inference",
        top_candidates_dir=resolved / "top_candidates",
        process_candidates_dir=resolved / "process_candidates",
    )
