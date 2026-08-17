"""
Calibrated scoring thresholds loaded from RNA-Puzzles decoy analysis.

Replaces hardcoded heuristics in score.py when calibration data is available.
"""
from __future__ import annotations

import json
from pathlib import Path

_DEFAULTS = {
    "rmsd_pass": 2.0,
    "rmsd_warn": 4.0,
    "rmsd_af3_pass": 18.0,
    "rmsd_af3_warn": 30.0,
    "geom_pass": 30.0,
    "geom_warn": 60.0,
    "geom_af3_pass": 90.0,
    "geom_af3_warn": 140.0,
    "gt_pass": -2.0,
    "gt_warn": 0.5,
    "contact_pass_per_residue": -1.5,
    "contact_warn_per_residue": -0.3,
    "calibration_source": "heuristic",
    "n_samples": 0,
}

_CACHE: dict | None = None


def reset_threshold_cache() -> None:
    """Clear cached thresholds (for tests)."""
    global _CACHE
    _CACHE = None


_THRESHOLDS_PATH = Path(__file__).parent.parent / "data" / "calibrated_thresholds.json"


def load_thresholds(path: str | Path | None = None) -> dict:
    """Load calibrated thresholds, falling back to defaults."""
    global _CACHE
    if _CACHE is not None and path is None:
        return _CACHE

    p = Path(path) if path else _THRESHOLDS_PATH
    if p.exists():
        with open(p) as f:
            data = json.load(f)
        merged = {**_DEFAULTS, **data}
    else:
        merged = dict(_DEFAULTS)

    if path is None:
        _CACHE = merged
    return merged


def save_thresholds(thresholds: dict, path: str | Path | None = None) -> None:
    """Persist calibrated thresholds to disk."""
    p = Path(path) if path else _THRESHOLDS_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(thresholds, f, indent=2)
    global _CACHE
    _CACHE = {**_DEFAULTS, **thresholds}
