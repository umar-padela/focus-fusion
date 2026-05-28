"""Visualization placeholders for LitePT/FocusFusion reports."""

from __future__ import annotations

from pathlib import Path


def ensure_visualization_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path
