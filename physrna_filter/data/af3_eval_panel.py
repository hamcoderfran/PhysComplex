"""
AF3 holdout evaluation panel (P1–P5 positives, N1–N5 negatives).

Machine-readable definitions: ``physrna_filter/data/af3_eval_panel.json``
"""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any

_PANEL_PATH = Path(__file__).with_name("af3_eval_panel.json")
_EXTENDED_PANEL_PATH = Path(__file__).with_name("af3_eval_panel_extended.json")


def default_panel_path() -> Path:
    return _PANEL_PATH


def extended_panel_path() -> Path:
    """20-job extended panel when available, else legacy 10-job panel."""
    if _EXTENDED_PANEL_PATH.is_file():
        return _EXTENDED_PANEL_PATH
    return _PANEL_PATH


def load_af3_eval_panel(path: str | Path | None = None) -> list[dict[str, Any]]:
    p = Path(path) if path else _PANEL_PATH
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _norm_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def match_panel_entry(label: str, panel: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    """
    Match a screening job label (zip path or cif name) to a panel entry.
    """
    panel = panel or load_af3_eval_panel()
    stem = _norm_key(Path(label).stem)
    for entry in panel:
        token = _norm_key(entry.get("zip_match", entry.get("af3_job_name", "")))
        if token and token in stem:
            return entry
    return None


def contrastive_pairs(panel: list[dict[str, Any]] | None = None) -> list[tuple[dict, dict]]:
    """Return (positive, negative) panel entries for AF3 contrastive fine-tuning."""
    panel = panel or load_af3_eval_panel()
    by_id = {e["id"]: e for e in panel}
    pairs: list[tuple[dict, dict]] = []
    seen: set[tuple[str, str]] = set()
    for neg in panel:
        if neg.get("label") != "negative":
            continue
        pos_id = neg.get("positive_id")
        if not pos_id or pos_id not in by_id:
            continue
        key = (pos_id, neg["id"])
        if key in seen:
            continue
        seen.add(key)
        pairs.append((by_id[pos_id], neg))
    return pairs


def filter_contrastive_pairs(
    pairs: list[tuple[dict, dict]],
    *,
    partner_groups: list[str] | None = None,
    entry_ids: list[str] | None = None,
) -> list[tuple[dict, dict]]:
    """Keep pairs whose positive entry matches partner group and/or entry id filters."""
    if not partner_groups and not entry_ids:
        return pairs
    groups = {g.lower() for g in partner_groups} if partner_groups else None
    ids = {i.upper() for i in entry_ids} if entry_ids else None
    filtered: list[tuple[dict, dict]] = []
    for pos, neg in pairs:
        if groups is not None:
            pg = str(pos.get("partner_group", "")).lower()
            if pg not in groups:
                continue
        if ids is not None:
            if pos.get("id", "").upper() not in ids and neg.get("id", "").upper() not in ids:
                continue
        filtered.append((pos, neg))
    return filtered


def partner_groups(panel: list[dict[str, Any]] | None = None) -> dict[str, list[dict]]:
    """Group panel entries by ``partner_group`` (same protein family)."""
    panel = panel or load_af3_eval_panel()
    groups: dict[str, list[dict]] = {}
    for entry in panel:
        g = entry.get("partner_group", "unknown")
        groups.setdefault(g, []).append(entry)
    return groups


def panel_zip_path(entry: dict[str, Any], search_dir: str | Path) -> Path | None:
    """Find an AF3 Server zip for a panel entry under ``search_dir``."""
    root = Path(search_dir)
    if not root.is_dir():
        return None
    token = _norm_key(entry.get("zip_match", ""))
    for child in sorted(root.glob("*.zip")):
        if token not in _norm_key(child.stem):
            continue
        if child.stat().st_size < 128 or not zipfile.is_zipfile(child):
            continue
        return child
    return None


def count_panel_zips(panel: list[dict[str, Any]], zip_dir: str | Path) -> int:
    """Number of panel entries with a valid AF3 zip on disk."""
    return sum(1 for entry in panel if panel_zip_path(entry, zip_dir) is not None)


def resolve_panel_for_zip_dir(zip_dir: str | Path) -> Path:
    """
    Pick extended vs legacy panel based on which matches more zips on disk.

    Avoids evaluating P6–P10 when only the bundled 10-job set is present.
    """
    zip_dir = Path(zip_dir)
    legacy = load_af3_eval_panel(default_panel_path())
    extended_path = extended_panel_path()
    if extended_path == default_panel_path():
        return default_panel_path()

    extended = load_af3_eval_panel(extended_path)
    n_legacy = count_panel_zips(legacy, zip_dir)
    n_extended = count_panel_zips(extended, zip_dir)

    if n_extended > n_legacy:
        return extended_path
    return default_panel_path()


def panel_native_reference(
    entry: dict[str, Any],
    panel: list[dict[str, Any]],
) -> str | None:
    """Native RNA from the paired positive control (for shuffle/swap negatives)."""
    pos_id = entry.get("positive_id")
    if not pos_id:
        return None
    by_id = {e["id"]: e for e in panel}
    pos = by_id.get(pos_id)
    if not pos:
        return None
    return pos.get("rna_sequence")


def export_missing_panel_jobs(
    panel: list[dict[str, Any]],
    zip_dir: str | Path,
    output_path: str | Path,
) -> Path:
    """Write AlphaFold Server JSON for panel entries missing zips on disk."""
    from .af3_client import build_alphafold_server_job

    zip_dir = Path(zip_dir)
    jobs: list[dict[str, Any]] = []
    for entry in panel:
        if panel_zip_path(entry, zip_dir) is not None:
            continue
        protein = entry.get("protein_sequence")
        rna = entry.get("rna_sequence")
        if not protein or not rna:
            continue
        job_name = entry.get("af3_job_name") or entry.get("zip_match", entry["id"])
        jobs.append(
            build_alphafold_server_job(
                protein,
                rna,
                job_name=str(job_name).replace(" ", "_"),
            )
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jobs, indent=2), encoding="utf-8")
    return out


def list_invalid_zips(zip_dir: str | Path) -> list[Path]:
    """AF3-named zips that are too small or not valid zip archives."""
    root = Path(zip_dir)
    if not root.is_dir():
        return []
    bad: list[Path] = []
    for child in root.glob("*.zip"):
        if child.stat().st_size < 128 or not zipfile.is_zipfile(child):
            bad.append(child)
    return bad
