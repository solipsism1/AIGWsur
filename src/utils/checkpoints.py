"""Helpers for discovering model checkpoints and related artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


BEST_CHECKPOINT_RE = re.compile(
    r"best_model_epoch(?P<epoch>\d+)_loss(?P<loss>-?\d+(?:\.\d+)?)\.pt$"
)


def _score_best_checkpoint(path: Path) -> tuple[float, int, float]:
    """Return a sortable score for epoch-tagged best-model checkpoints."""
    match = BEST_CHECKPOINT_RE.match(path.name)
    if not match:
        return (float("inf"), -1, path.stat().st_mtime)

    return (
        float(match.group("loss")),
        -int(match.group("epoch")),
        -path.stat().st_mtime,
    )


def find_best_checkpoint(search_dir: str | Path = "checkpoints") -> Path:
    """Find the best checkpoint in a directory.

    Preference order:
      1. A stable ``best_model.pt`` alias if present.
      2. The epoch-tagged checkpoint with the lowest validation loss.
    """
    search_path = Path(search_dir)
    alias_path = search_path / "best_model.pt"
    if alias_path.exists():
        return alias_path

    candidates = sorted(search_path.glob("best_model_epoch*_loss*.pt"), key=_score_best_checkpoint)
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"Could not find a best-model checkpoint in {search_path.resolve()}"
    )


def resolve_checkpoint_path(
    checkpoint_path: Optional[str | Path] = None,
    search_dir: str | Path = "checkpoints",
) -> Path:
    """Resolve a checkpoint path, auto-discovering the best checkpoint if needed."""
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    return find_best_checkpoint(search_dir)


def resolve_compressor_path(
    checkpoint_path: str | Path,
    compressor_path: Optional[str | Path] = None,
    fallback_dir: str | Path = "checkpoints",
) -> Path:
    """Resolve the compressor path associated with a checkpoint."""
    if compressor_path is not None:
        path = Path(compressor_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"Compressor not found: {path}")

    checkpoint_path = Path(checkpoint_path)
    sibling_path = checkpoint_path.parent / "compressor.h5"
    if sibling_path.exists():
        return sibling_path

    fallback_path = Path(fallback_dir) / "compressor.h5"
    if fallback_path.exists():
        return fallback_path

    raise FileNotFoundError(
        f"Could not find compressor.h5 next to {checkpoint_path} or in {Path(fallback_dir)}"
    )
