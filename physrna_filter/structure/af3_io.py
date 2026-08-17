"""
AlphaFold 3 Server output helpers.

AF3 delivers results as ZIP archives containing ranked mmCIF models, e.g.:

    fold_6sqn_u1a_hairpin.zip
      fold_6sqn_u1a_hairpin_model_0.cif   # highest confidence
      fold_6sqn_u1a_hairpin_model_1.cif
      ...

This module extracts the requested model and returns a path usable by
parse_complex() and the screening pipeline.
"""
from __future__ import annotations

import os
import re
import zipfile
from pathlib import Path

_STRUCTURE_SUFFIXES = {".pdb", ".cif", ".mmcif"}
_ZIP_SUFFIX = ".zip"


def _af3_cache_dir() -> Path:
    base = Path(os.environ.get("PHYRNA_AF3_CACHE", Path.home() / ".cache" / "physrna_filter" / "af3_extracts"))
    base.mkdir(parents=True, exist_ok=True)
    return base


def _model_rank_from_name(name: str) -> int:
    """Lower rank = higher AF3 confidence (model_0 is best)."""
    base = Path(name).name
    m = re.search(r"model_(\d+)", base, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    if re.search(r"model\.cif$", base, flags=re.IGNORECASE):
        return 0
    return 10_000


def list_af3_models_in_zip(zip_path: Path) -> list[str]:
    """Return mmCIF member paths inside an AF3 zip, sorted by model rank."""
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            n for n in zf.namelist()
            if Path(n).suffix.lower() in (".cif", ".mmcif") and "/__MACOSX" not in n
        ]
    return sorted(members, key=_model_rank_from_name)


def extract_af3_zip(
    zip_path: Path,
    model_rank: int = 0,
    cache_dir: Path | None = None,
) -> Path:
    """
    Extract the AF3 mmCIF model with the given rank from a Server zip.

    Returns path to the extracted .cif file (cached under ~/.cache/physrna_filter).
    """
    zip_path = Path(zip_path).resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(f"AF3 zip not found: {zip_path}")

    members = list_af3_models_in_zip(zip_path)
    if not members:
        raise ValueError(f"No .cif models found inside {zip_path}")

    by_rank = sorted(members, key=_model_rank_from_name)
    rank_to_member = {_model_rank_from_name(m): m for m in by_rank}
    if model_rank in rank_to_member:
        member = rank_to_member[model_rank]
    else:
        member = by_rank[0]

    out_dir = (cache_dir or _af3_cache_dir()) / zip_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / Path(member).name

    zip_mtime = zip_path.stat().st_mtime
    if out_path.exists() and out_path.stat().st_mtime >= zip_mtime:
        return out_path

    with zipfile.ZipFile(zip_path) as zf:
        out_path.write_bytes(zf.read(member))
    return out_path


def is_structure_file(path: Path) -> bool:
    return path.suffix.lower() in _STRUCTURE_SUFFIXES


def is_af3_zip(path: Path) -> bool:
    return path.suffix.lower() == _ZIP_SUFFIX


def resolve_structure_path(
    path: str | Path,
    model_rank: int = 0,
) -> Path:
    """
    Return a local .pdb / .cif path, extracting AF3 zips when needed.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {p}")

    if is_af3_zip(p):
        return extract_af3_zip(p, model_rank=model_rank)

    # Zip without .zip extension (misnamed download)
    if p.is_file():
        with open(p, "rb") as fh:
            if fh.read(2) == b"PK":
                return extract_af3_zip(p, model_rank=model_rank)

    if is_structure_file(p):
        return p.resolve()

    raise ValueError(
        f"Unsupported structure input: {p}. "
        "Expected .pdb, .cif, .mmcif, or AF3 .zip"
    )


def collect_structure_inputs(
    paths: list[str | Path],
    model_rank: int = 0,
) -> list[Path]:
    """
    Expand directories / zip files into concrete structure file paths.
    """
    seen: set[str] = set()
    out: list[Path] = []

    def add(path: Path) -> None:
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            out.append(path)

    for raw in paths:
        p = Path(raw)
        if not p.exists():
            continue
        if p.is_dir():
            for pattern in ("*.pdb", "*.cif", "*.mmcif", "*.zip"):
                for child in sorted(p.glob(pattern)):
                    if child.suffix.lower() == _ZIP_SUFFIX:
                        add(resolve_structure_path(child, model_rank=model_rank))
                    elif is_structure_file(child):
                        add(child.resolve())
        elif is_af3_zip(p):
            add(resolve_structure_path(p, model_rank=model_rank))
        elif is_structure_file(p):
            add(p.resolve())

    return out
