"""Tests for PhysRNA config and accuracy-first CLI defaults."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest


def test_resolve_prefers_user_checkpoint(tmp_path, monkeypatch):
    from physrna_filter.config import (
        resolve_gt_checkpoint,
        shipped_checkpoint,
        user_data_dir,
    )

    monkeypatch.setenv("PHYSRNA_HOME", str(tmp_path))
    shipped = shipped_checkpoint()
    if not shipped.is_file():
        pytest.skip("shipped checkpoint not in tree")

    user = user_data_dir() / "gt_checkpoint.pt"
    shutil.copy2(shipped, user)
    resolved = resolve_gt_checkpoint()
    assert resolved == user.resolve()


def test_writable_copies_shipped_to_user(tmp_path, monkeypatch):
    from physrna_filter.config import shipped_checkpoint, user_checkpoint, writable_gt_checkpoint

    monkeypatch.setenv("PHYSRNA_HOME", str(tmp_path))
    shipped = shipped_checkpoint()
    if not shipped.is_file():
        pytest.skip("shipped checkpoint not in tree")

    out = writable_gt_checkpoint()
    assert out == user_checkpoint().resolve()
    assert out.stat().st_size > 10_000


def test_rank_defaults_deep_not_fast():
    import inspect

    from physrna_filter.validation.rank_af3_candidates import rank_af3_candidates

    sig = inspect.signature(rank_af3_candidates)
    assert sig.parameters["fast_mode"].default is False
    assert sig.parameters["finetune_epochs"].default == 50


def test_cli_parser_has_init_rank_panel():
    from physrna_filter.cli import build_parser

    ap = build_parser()
    assert "init" in ap.format_help()
    assert "rank" in ap.format_help()
    assert "panel" in ap.format_help()
    assert "predict" in ap.format_help()
    assert "configure" in ap.format_help()
    assert "boltz" in ap.format_help()
