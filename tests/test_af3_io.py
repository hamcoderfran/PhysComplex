"""Tests for AlphaFold 3 Server zip / mmCIF input handling."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from Bio.PDB import MMCIFIO, PDBParser

from physrna_filter.structure.af3_io import (
    collect_structure_inputs,
    extract_af3_zip,
    list_af3_models_in_zip,
    resolve_structure_path,
)
from physrna_filter.structure.parse_complex import parse_complex


def _make_af3_zip(tmp_path: Path, stem: str = "fold_test_job") -> Path:
    """Build an AF3-style zip with two ranked mmCIF models from 1urn.pdb."""
    repo_root = Path(__file__).resolve().parents[1]
    pdb_source = repo_root / "physrna_filter" / "data" / "structures" / "1urn.pdb"
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", str(pdb_source))

    zip_path = tmp_path / f"{stem}.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for rank in (1, 0):
            cif_name = f"{stem}_model_{rank}.cif"
            cif_path = tmp_path / cif_name
            io = MMCIFIO()
            io.set_structure(structure)
            io.save(str(cif_path))
            zf.write(cif_path, cif_name)
    return zip_path


def test_list_af3_models_sorted_by_rank(tmp_path):
    zip_path = _make_af3_zip(tmp_path)
    members = list_af3_models_in_zip(zip_path)
    assert [Path(m).name for m in members] == [
        "fold_test_job_model_0.cif",
        "fold_test_job_model_1.cif",
    ]


def test_extract_af3_zip_picks_model_zero(tmp_path):
    zip_path = _make_af3_zip(tmp_path)
    cache = tmp_path / "cache"
    out = extract_af3_zip(zip_path, model_rank=0, cache_dir=cache)
    assert out.name == "fold_test_job_model_0.cif"
    assert out.exists()
    assert out.read_text().startswith("data_")


def test_extract_af3_zip_uses_cache(tmp_path):
    zip_path = _make_af3_zip(tmp_path)
    cache = tmp_path / "cache"
    first = extract_af3_zip(zip_path, cache_dir=cache)
    second = extract_af3_zip(zip_path, cache_dir=cache)
    assert first == second


def test_resolve_structure_path_for_zip(tmp_path):
    zip_path = _make_af3_zip(tmp_path, stem="fold_6sqn_u1a_hairpin")
    resolved = resolve_structure_path(zip_path, model_rank=0)
    assert resolved.suffix.lower() == ".cif"
    assert "model_0" in resolved.name


def test_collect_structure_inputs_zip_and_dir(tmp_path):
    zip_path = _make_af3_zip(tmp_path)
    (tmp_path / "solo.pdb").write_text("ATOM")
    inputs = collect_structure_inputs([str(zip_path), str(tmp_path / "solo.pdb")])
    assert len(inputs) == 2
    assert any(p.suffix.lower() == ".cif" for p in inputs)
    assert any(p.suffix.lower() == ".pdb" for p in inputs)


def test_parse_complex_from_af3_zip(tmp_path):
    zip_path = _make_af3_zip(tmp_path, stem="fold_6sqn_u1a_hairpin")
    parsed = parse_complex(str(zip_path))
    assert len(parsed.protein_chains) >= 1
    assert len(parsed.rna_chains) >= 1


def test_load_structure_sniffs_zip_without_extension(tmp_path):
    zip_path = _make_af3_zip(tmp_path, stem="fold_misnamed")
    misnamed = tmp_path / "af3_download"
    misnamed.write_bytes(zip_path.read_bytes())
    parsed = parse_complex(str(misnamed))
    assert len(parsed.rna_chains) >= 1


def test_load_structure_utf8_mmcif(tmp_path):
    from physrna_filter.structure.parse_complex import _load_structure

    repo_root = Path(__file__).resolve().parents[1]
    pdb_source = repo_root / "physrna_filter" / "data" / "structures" / "1urn.pdb"
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", str(pdb_source))
    cif_path = tmp_path / "utf8_model.cif"
    io = MMCIFIO()
    io.set_structure(structure)
    io.save(str(cif_path))
    # AF3 files can include non-ASCII metadata; ensure UTF-8 open works.
    text = cif_path.read_text(encoding="utf-8")
    cif_path.write_text(text + "# note: café — AF3 metadata\n", encoding="utf-8")
    loaded = _load_structure(str(cif_path))
    assert len(list(loaded.get_chains())) >= 1


def test_resolve_structure_path_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_structure_path(tmp_path / "missing.zip")
