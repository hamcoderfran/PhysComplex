"""
Candidate manifest for batch AF3 ranking (no hand-built panel JSON).

Users place AF3 Server zips in a folder and optionally provide a CSV manifest.
When no manifest exists, job metadata is inferred from each zip's job_request.json.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from ..structure.af3_confidence import sequences_from_af3_zip
from ..structure.af3_io import is_af3_zip

_MANIFEST_COLUMNS = (
    "job_name",
    "zip_file",
    "rbp_name",
    "rna_sequence",
    "protein_sequence",
    "partner_group",
    "chrom",
    "genomic_start",
    "genomic_end",
    "notes",
)


def _norm_group(name: str | None) -> str:
    if not name:
        return "unknown"
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "unknown"


def _rbp_matches(needle: str, candidate_rbp: str | None, job_name: str | None = None) -> bool:
    """True if ``needle`` (e.g. LIN28A) matches this job's RBP or title."""
    if not needle:
        return True
    n = _norm_group(needle)
    for hay in (candidate_rbp, job_name):
        if not hay:
            continue
        h = _norm_group(hay)
        if n == h or n in h or h in n:
            return True
        # lin28 <-> lin28a
        if "lin28" in n and "lin28" in h:
            return True
    return False


def _is_wrong_partner_control(job_name: str | None) -> bool:
    """Panel-style wrong-partner AF3 jobs (N*_*_with_*), not oligo candidates."""
    if not job_name:
        return False
    low = job_name.lower().replace("-", "_")
    return "_with_" in low or bool(re.match(r"^n\d+_", low))


def _guess_rbp_from_name(job_name: str) -> str | None:
    """Best-effort RBP label from AF3 job title."""
    text = job_name.replace("_", " ")
    mapping = (
        ("LIN28", "LIN28A"),
        ("U1A", "U1A"),
        ("MS2", "MS2"),
        ("REV", "HIV_REV"),
        ("RBFOX", "RBFOX1"),
        ("PTBP", "PTBP1"),
        ("PABP", "PABPC1"),
        ("PUM", "PUMILIO"),
        ("NOVA", "NOVA2"),
        ("HUD", "HUD"),
    )
    for token, label in mapping:
        if token.lower() in text.lower():
            return label
    return None


def load_manifest_csv(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for i, raw in enumerate(reader, start=1):
            row = {k: (raw.get(k) or "").strip() or None for k in _MANIFEST_COLUMNS}
            row["id"] = f"C{i}"
            if not row.get("partner_group") and row.get("rbp_name"):
                row["partner_group"] = _norm_group(row["rbp_name"])
            rows.append(row)
    return rows


def write_manifest_template(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerow({
            "job_name": "my_protein_RNA1",
            "zip_file": "fold_my_protein_rna1.zip",
            "rbp_name": "LIN28A",
            "rna_sequence": "GGCAGGGAUUUUGCCCGGAG",
            "protein_sequence": "",
            "partner_group": "lin28",
            "chrom": "chr1",
            "genomic_start": "1000000",
            "genomic_end": "1000020",
            "notes": "optional eCLIP coordinates",
        })
    return path


def discover_af3_jobs(
    zip_dir: str | Path,
    *,
    rbp_filter: str | None = None,
    include_controls: bool = False,
) -> list[dict[str, Any]]:
    """
    Build candidate rows from all valid AF3 zips in a directory.

    Parses ``*_job_request.json`` inside each zip for sequences and job name.

    When ``rbp_filter`` is set (e.g. LIN28A), only jobs whose title or inferred
    RBP match are included — other zips in the folder are ignored.
    """
    root = Path(zip_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Zip directory not found: {root}")

    rows: list[dict[str, Any]] = []
    for i, child in enumerate(sorted(root.glob("*.zip")), start=1):
        if not is_af3_zip(child) or child.stat().st_size < 128:
            continue
        job_name, protein, rna = sequences_from_af3_zip(child)
        job_name = job_name or child.stem
        if not include_controls and _is_wrong_partner_control(job_name):
            continue
        rbp = _guess_rbp_from_name(job_name)
        if rbp_filter and not _rbp_matches(rbp_filter, rbp, job_name):
            continue
        if rbp_filter and not rbp:
            rbp = rbp_filter
        rows.append({
            "id": f"C{len(rows) + 1}",
            "job_name": job_name,
            "zip_file": child.name,
            "zip_path": str(child.resolve()),
            "rbp_name": rbp,
            "rna_sequence": rna,
            "protein_sequence": protein,
            "partner_group": _norm_group(rbp),
            "chrom": None,
            "genomic_start": None,
            "genomic_end": None,
            "notes": "auto-discovered from zip",
        })
    return rows


def resolve_zip_path(candidate: dict[str, Any], zip_dir: Path) -> Path | None:
    if candidate.get("zip_path"):
        p = Path(candidate["zip_path"])
        return p if p.is_file() else None
    name = candidate.get("zip_file") or candidate.get("job_name")
    if not name:
        return None
    root = zip_dir
    stem = Path(name).stem.lower()
    for pattern in ("*.zip", "*.cif", "*.mmcif", "*.pdb"):
        for child in root.rglob(pattern):
            child_stem = child.stem.lower()
            if stem in child_stem or child_stem.startswith(stem):
                return child
    for child in root.glob("*.zip"):
        if name.lower() in child.name.lower():
            return child
    direct = root / name
    if direct.is_file():
        return direct
    for suffix in (".zip", ".cif", ".mmcif", ".pdb"):
        candidate_path = direct.with_suffix(suffix)
        if candidate_path.is_file():
            return candidate_path
    return None


def load_candidates(
    zip_dir: str | Path,
    manifest_csv: str | Path | None = None,
    *,
    rbp_name: str | None = None,
    include_controls: bool = False,
) -> list[dict[str, Any]]:
    """Load manifest CSV or auto-discover zips under ``zip_dir``."""
    zip_dir = Path(zip_dir)
    if manifest_csv:
        rows = load_manifest_csv(manifest_csv)
        for row in rows:
            zp = resolve_zip_path(row, zip_dir)
            if zp:
                row["zip_path"] = str(zp.resolve())
            if not row.get("rbp_name") and rbp_name:
                row["rbp_name"] = rbp_name
            if not row.get("partner_group") and row.get("rbp_name"):
                row["partner_group"] = _norm_group(row["rbp_name"])
        if rbp_name:
            rows = [
                r for r in rows
                if _rbp_matches(rbp_name, r.get("rbp_name"), r.get("job_name"))
            ]
        if not include_controls:
            rows = [r for r in rows if not _is_wrong_partner_control(r.get("job_name"))]
        return rows
    return discover_af3_jobs(
        zip_dir, rbp_filter=rbp_name, include_controls=include_controls
    )


def candidates_to_panel_json(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert manifest rows to af3_eval_panel-style positive entries."""
    panel: list[dict[str, Any]] = []
    for c in candidates:
        stem = Path(c.get("zip_file") or c.get("job_name", "job")).stem
        panel.append({
            "id": c["id"],
            "label": "positive",
            "partner_group": c.get("partner_group") or "unknown",
            "af3_job_name": c.get("job_name") or stem,
            "zip_match": stem.lower(),
            "pdb": None,
            "rbp_name": c.get("rbp_name"),
            "protein_sequence": c.get("protein_sequence"),
            "rna_sequence": c.get("rna_sequence"),
            "notes": c.get("notes") or "candidate ranking",
        })
    return panel


def generate_swap_panel(
    candidates: list[dict[str, Any]],
    *,
    max_negatives_per_positive: int = 3,
) -> list[dict[str, Any]]:
    """
  Build a contrastive panel: each candidate is positive; negatives swap RNAs
  within the same ``partner_group`` (same protein, wrong RNA).
    """
    positives = candidates_to_panel_json(candidates)
    by_group: dict[str, list[dict]] = {}
    for c in candidates:
        g = c.get("partner_group") or "unknown"
        by_group.setdefault(g, []).append(c)

    panel = list(positives)
    neg_idx = 1
    for pos in positives:
        group = pos["partner_group"]
        pool = [c for c in by_group.get(group, []) if c["id"] != pos["id"]]
        if not pool:
            # Cross-group RNA swaps when only one candidate per protein
            pool = [c for c in candidates if c["id"] != pos["id"]]
        for other in pool[:max_negatives_per_positive]:
            neg_id = f"N{neg_idx}"
            neg_idx += 1
            other_stem = Path(
                other.get("zip_file") or other.get("job_name", "neg")
            ).stem
            # Negative uses same protein metadata as positive, wrong RNA
            panel.append({
                "id": neg_id,
                "label": "negative",
                "partner_group": group,
                "af3_job_name": f"{pos['af3_job_name']}_with_{other_stem}_rna",
                "zip_match": other_stem.lower(),
                "pdb": None,
                "rbp_name": pos.get("rbp_name"),
                "protein_sequence": pos.get("protein_sequence"),
                "rna_sequence": other.get("rna_sequence"),
                "positive_id": pos["id"],
                "notes": f"wrong RNA swap from {other.get('job_name')}",
            })
    return panel


def save_panel_json(panel: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(panel, fh, indent=2)
        fh.write("\n")
    return path
