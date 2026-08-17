"""Tests for Windows/WSL path bridging."""

from __future__ import annotations


def test_windows_to_wsl_path(monkeypatch):
    from physrna_filter.simulation import wsl_paths

    monkeypatch.setattr(wsl_paths.sys, "platform", "win32")
    assert (
        wsl_paths.windows_to_wsl_path(r"C:\Users\frann\Temp\sim")
        == "/mnt/c/Users/frann/Temp/sim"
    )


def test_host_path_unchanged_on_linux(monkeypatch):
    from physrna_filter.simulation import wsl_paths

    monkeypatch.setattr(wsl_paths.sys, "platform", "linux")
    assert wsl_paths.host_path_for_subprocess("/tmp/foo") == "/tmp/foo"


def test_oxdna_uses_wsl_env(monkeypatch):
    from physrna_filter.simulation.wsl_paths import oxdna_uses_wsl

    monkeypatch.setenv("OXDNA_BIN", "wsl:/home/frann/oxDNA/build/bin/oxDNA")
    assert oxdna_uses_wsl() is True

    monkeypatch.setenv("OXDNA_BIN", "/usr/local/bin/oxDNA")
    assert oxdna_uses_wsl() is False


def test_wsl_find_oat_path_from_env(monkeypatch):
    from physrna_filter.simulation import wsl_paths

    monkeypatch.setattr(wsl_paths.sys, "platform", "win32")
    monkeypatch.setenv(
        "OAT_BIN",
        "wsl:/home/frann/miniconda3/bin/oat",
    )
    assert wsl_paths.wsl_find_oat_path() == "/home/frann/miniconda3/bin/oat"


def test_host_path_preserves_wsl_posix_on_windows(monkeypatch):
    from physrna_filter.simulation import wsl_paths

    monkeypatch.setattr(wsl_paths.sys, "platform", "win32")
    monkeypatch.setenv("OXDNA_BIN", "wsl:/home/frann/oxDNA/build/bin/oxDNA")
    assert (
        wsl_paths.host_path_for_subprocess("/home/frann/miniconda3/bin/oat")
        == "/home/frann/miniconda3/bin/oat"
    )
    assert wsl_paths.host_path_for_subprocess("/tmp/physrna_sim_abcd") == "/tmp/physrna_sim_abcd"


def test_looks_like_posix_path_on_windows(monkeypatch):
    from physrna_filter.simulation.wsl_paths import _looks_like_path

    monkeypatch.setattr("physrna_filter.simulation.wsl_paths.sys.platform", "win32")
    assert _looks_like_path("/home/frann/oxDNA/build/bin/oxDNA") is True


def test_wsl_python_for_oat_keeps_posix_path_on_windows(monkeypatch):
    from physrna_filter.simulation.wsl_paths import wsl_python_for_oat

    monkeypatch.setattr("physrna_filter.simulation.wsl_paths.sys.platform", "win32")
    assert (
        wsl_python_for_oat("/home/frann/miniconda3/bin/oat")
        == "/home/frann/miniconda3/bin/python"
    )
