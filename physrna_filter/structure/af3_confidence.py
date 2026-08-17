"""
Parse AlphaFold 3 Server confidence scores from downloaded zip archives.

Each AF3 zip contains ``*_summary_confidences_{rank}.json`` with ipTM, pTM, etc.
"""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any


def _confidence_member_names(zip_path: Path, model_rank: int) -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        pattern = re.compile(
            rf"summary_confidences_{model_rank}\.json$", re.IGNORECASE
        )
        return [n for n in zf.namelist() if pattern.search(n)]


def read_af3_confidence(
    zip_path: str | Path,
    model_rank: int = 0,
) -> dict[str, Any]:
    """
    Return AF3 confidence dict for ``model_rank`` (0 = top model).

    Keys typically include: ``iptm``, ``ptm``, ``ranking_score``,
    ``chain_iptm``, ``chain_pair_iptm``.
    """
    zip_path = Path(zip_path)
    if not zip_path.is_file():
        return {}

    try:
        with zipfile.ZipFile(zip_path) as zf:
            members = _confidence_member_names(zip_path, model_rank)
            if not members:
                # Fall back to any summary confidences file
                members = [
                    n for n in zf.namelist()
                    if "summary_confidences" in n.lower() and n.endswith(".json")
                ]
                members.sort()
                if model_rank < len(members):
                    members = [members[model_rank]]
                elif members:
                    members = [members[0]]
            if not members:
                return {}
            raw = zf.read(members[0]).decode("utf-8", errors="replace")
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
    except (zipfile.BadZipFile, json.JSONDecodeError, KeyError):
        return {}
    return {}


def af3_iptm(zip_path: str | Path, model_rank: int = 0) -> float | None:
    conf = read_af3_confidence(zip_path, model_rank=model_rank)
    val = conf.get("iptm")
    return float(val) if val is not None else None


def af3_ptm(zip_path: str | Path, model_rank: int = 0) -> float | None:
    conf = read_af3_confidence(zip_path, model_rank=model_rank)
    val = conf.get("ptm")
    return float(val) if val is not None else None


def af3_ranking_score(zip_path: str | Path, model_rank: int = 0) -> float | None:
    conf = read_af3_confidence(zip_path, model_rank=model_rank)
    val = conf.get("ranking_score")
    if val is not None:
        return float(val)
    return af3_iptm(zip_path, model_rank=model_rank)


def read_af3_job_request(zip_path: str | Path) -> dict[str, Any]:
    """Parse the AF3 Server job request JSON inside a zip (sequences, job name)."""
    zip_path = Path(zip_path)
    if not zip_path.is_file():
        return {}

    try:
        with zipfile.ZipFile(zip_path) as zf:
            members = [n for n in zf.namelist() if n.endswith("_job_request.json")]
            if not members:
                members = [n for n in zf.namelist() if "job_request.json" in n.lower()]
            if not members:
                return {}
            raw = zf.read(members[0]).decode("utf-8", errors="replace")
            payload = json.loads(raw)
    except (zipfile.BadZipFile, json.JSONDecodeError):
        return {}

    if isinstance(payload, list) and payload:
        return payload[0] if isinstance(payload[0], dict) else {}
    if isinstance(payload, dict):
        return payload
    return {}


def sequences_from_af3_zip(zip_path: str | Path) -> tuple[str | None, str | None, str | None]:
    """
    Extract (job_name, protein_sequence, rna_sequence) from an AF3 zip.

    Returns empty strings as None when unavailable.
    """
    job = read_af3_job_request(zip_path)
    name = job.get("name") or Path(zip_path).stem
    protein: str | None = None
    rna: str | None = None
    for block in job.get("sequences") or []:
        if not isinstance(block, dict):
            continue
        if "proteinChain" in block:
            protein = block["proteinChain"].get("sequence") or protein
        if "rnaSequence" in block:
            rna = block["rnaSequence"].get("sequence") or rna
    return name, protein, rna
