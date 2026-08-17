"""Tests for oat / WSL PDB conversion helpers."""

from __future__ import annotations

from pathlib import Path


def test_write_wsl_pdb_oxdna_script(tmp_path):
    from physrna_filter.simulation.oat_convert import write_wsl_pdb_oxdna_script

    path = write_wsl_pdb_oxdna_script(str(tmp_path))
    text = Path(path).read_text(encoding="utf-8")
    assert "PDB_oxDNA" in text
    assert path.endswith("_physrna_pdb_oxdna.py")


def test_wsl_oat_pdb_module_ok_windows(monkeypatch):
    from physrna_filter.simulation import oat_convert

    monkeypatch.setattr(oat_convert.sys, "platform", "win32")

    calls: list = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        class Result:
            returncode = 0
            stdout = ""
            stderr = ""
        return Result()

    monkeypatch.setattr(oat_convert.subprocess, "run", fake_run)
    assert oat_convert.wsl_oat_pdb_module_ok("/home/frann/miniconda3/bin/oat") is True
    assert calls[0][:2] == ["wsl", "/home/frann/miniconda3/bin/python"]


def test_smoke_test_conversion_success(monkeypatch, tmp_path):
    from physrna_filter.simulation import oat_convert

    def fake_pdb_to_oxrna(pdb_path, top_path, conf_path):
        Path(top_path).write_text("top", encoding="utf-8")
        Path(conf_path).write_text("conf", encoding="utf-8")

    monkeypatch.setattr(
        "physrna_filter.simulation.run_simulation._pdb_to_oxrna",
        fake_pdb_to_oxrna,
    )
    ok, message = oat_convert.smoke_test_conversion(str(tmp_path / "in.pdb"))
    assert ok is True
    assert "OK" in message
