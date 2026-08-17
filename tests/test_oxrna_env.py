"""Tests for oxDNA discovery helpers."""

from __future__ import annotations


def test_resolve_oxdna_command_from_env(monkeypatch):
    from physrna_filter.simulation.oxrna_env import resolve_oxdna_command

    monkeypatch.setenv("OXDNA_BIN", "wsl:/home/user/oxDNA/build/bin/oxDNA")
    assert resolve_oxdna_command() == [
        "wsl", "/home/user/oxDNA/build/bin/oxDNA"
    ]


def test_oxdna_available_usage_probe(monkeypatch, tmp_path):
    from physrna_filter.simulation import oxrna_env

    fake_bin = tmp_path / "oxDNA"
    fake_bin.write_text("#!/bin/sh\necho Usage is oxDNA input_file\n")
    fake_bin.chmod(0o755)
    monkeypatch.setenv("OXDNA_BIN", str(fake_bin))

    assert oxrna_env.oxdna_available() is True


def test_oxdna_available_false_when_missing(monkeypatch):
    from physrna_filter.simulation import oxrna_env

    monkeypatch.setenv("OXDNA_BIN", "/nonexistent/oxDNA")
    monkeypatch.setattr(oxrna_env.shutil, "which", lambda _: None)

    def fake_run(*args, **kwargs):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(oxrna_env.subprocess, "run", fake_run)
    assert oxrna_env.oxdna_available() is False


def test_extract_chain_pair(tmp_path):
    from physrna_filter.data.extract_chain_pair import extract_chain_pair
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "physrna_filter" / "data" / "structures" / "1urn.pdb"
    out = tmp_path / "single.pdb"
    extract_chain_pair(str(src), "A", "P", str(out))
    text = out.read_text(encoding="utf-8")
    assert "ATOM" in text
    assert out.stat().st_size > 1000
