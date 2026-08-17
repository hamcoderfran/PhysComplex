"""Tests for RNA folding helpers."""

from __future__ import annotations

from pathlib import Path


def test_detect_available_method_without_which_command(monkeypatch, tmp_path):
    """Windows has no `which` binary; detection must not call it."""
    import physrna_filter.simulation.fold_rna as fold_rna

    calls: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(fold_rna.subprocess, "run", fake_run)
    monkeypatch.setattr(fold_rna.shutil, "which", lambda name: None)

    assert fold_rna._detect_available_method() == "extended"
    assert calls == []


def test_build_extended_strand_writes_pdb(tmp_path):
    from physrna_filter.simulation.fold_rna import _build_extended_strand

    out = tmp_path / "rna.pdb"
    _build_extended_strand("AUGC", str(out))
    text = Path(out).read_text(encoding="utf-8")
    assert "ATOM" in text
    assert "END" in text
    assert "O5'" in text
    assert "C1'" in text
    assert "C3'" in text
    assert "N1 " in text or "N9 " in text
    assert text.count("C2 ") >= 1


def test_locate_oxdna_outputs_conf(tmp_path):
    from physrna_filter.simulation.run_simulation import _locate_oxdna_outputs

    (tmp_path / "generated.top").write_text("top", encoding="utf-8")
    (tmp_path / "generated.conf").write_text("conf", encoding="utf-8")
    found = _locate_oxdna_outputs(str(tmp_path), str(tmp_path / "generated"))
    assert found is not None
    assert found[0].endswith("generated.top")


def test_locate_oxdna_outputs_dat(tmp_path):
    from physrna_filter.simulation.run_simulation import _locate_oxdna_outputs

    (tmp_path / "generated.top").write_text("top", encoding="utf-8")
    (tmp_path / "generated.dat").write_text("conf", encoding="utf-8")
    found = _locate_oxdna_outputs(str(tmp_path), str(tmp_path / "generated"))
    assert found is not None
    assert found[0].endswith("generated.top")
    assert found[1].endswith("generated.dat")


def test_wsl_oat_shell_command_uses_oat_binary():
    from physrna_filter.simulation.run_simulation import _wsl_oat_shell_command

    cmd = _wsl_oat_shell_command(
        "/home/frann/miniconda3/bin/oat",
        "/mnt/c/tmp/work",
        "/mnt/c/tmp/work/rna.pdb",
        "/mnt/c/tmp/work/generated",
    )
    assert "/home/frann/miniconda3/bin/oat' PDB_oxDNA" in cmd
    assert "oxDNA_analysis_tools.entry" not in cmd
