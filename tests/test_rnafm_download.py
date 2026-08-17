"""Tests for RNA-FM local-only loading (no network during training)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch


def test_get_rnafm_embeddings_uses_fallback_without_local_weights(monkeypatch):
    from physrna_filter.analysis import rnafm_embeddings as re

    re._model = None
    re._package_installed = True
    re._load_warning_emitted = False
    re._mode_announced = False
    monkeypatch.delenv("RNAFM_CHECKPOINT", raising=False)
    monkeypatch.setattr(re, "resolve_weights_path", lambda: None)

    result = re.get_rnafm_embeddings("1abc", "R", "AUGC", [1, 2, 3, 4])
    assert len(result) == 4
    assert all(v.shape[0] == re.RNAFM_FALLBACK for v in result.values())


def test_load_model_never_calls_hub_without_local_path(monkeypatch):
    from physrna_filter.analysis import rnafm_embeddings as re

    re._model = None
    re._package_installed = True
    monkeypatch.setattr(re, "resolve_weights_path", lambda: None)

    with pytest.raises(RuntimeError, match="not found locally"):
        re._load_model()


def test_effective_rnafm_feature_dim_no_network(monkeypatch):
    from physrna_filter.analysis.rnafm_embeddings import (
        effective_rnafm_feature_dim,
        RNAFM_FALLBACK,
        RNAFM_DIM,
    )
    from physrna_filter.analysis import rnafm_embeddings as re

    re._package_installed = True
    re._model = None
    re._load_failed = False
    monkeypatch.setattr(re, "resolve_weights_path", lambda: None)
    assert effective_rnafm_feature_dim(True) == RNAFM_FALLBACK

    re._model = object()  # simulate loaded model
    assert effective_rnafm_feature_dim(True) == RNAFM_DIM


def test_announce_rnafm_mode_prints_once(capsys, monkeypatch):
    from physrna_filter.analysis import rnafm_embeddings as re

    re._mode_announced = False
    re._package_installed = False
    monkeypatch.setattr(re, "resolve_weights_path", lambda: None)

    re.announce_rnafm_mode()
    re.announce_rnafm_mode()
    out = capsys.readouterr().out
    assert out.count("RNA-FM:") == 1


def test_download_script_exists():
    from physrna_filter.data.download_rnafm_weights import DEST_DIR, FILENAME
    assert FILENAME == "RNA-FM_pretrained.pth"
    assert "rnafm_weights" in str(DEST_DIR)


def test_rejects_small_file_as_invalid(tmp_path):
    from physrna_filter.analysis.rnafm_embeddings import is_valid_weights_file

    bad = tmp_path / "RNA-FM_pretrained.pth"
    bad.write_text("<html>403 Forbidden</html>")
    assert not is_valid_weights_file(bad)


def test_rejects_html_error_page(tmp_path):
    from physrna_filter.analysis.rnafm_embeddings import is_valid_weights_file, MIN_WEIGHT_BYTES

    bad = tmp_path / "RNA-FM_pretrained.pth"
    bad.write_bytes(b"<!DOCTYPE html><html>403 Forbidden</html>" + b"x" * MIN_WEIGHT_BYTES)
    assert not is_valid_weights_file(bad)


def test_clean_path_string_strips_quotes():
    from physrna_filter.analysis.rnafm_embeddings import _clean_path_string

    assert _clean_path_string('"C:\\weights\\RNA-FM_pretrained.pth"') == (
        "C:\\weights\\RNA-FM_pretrained.pth"
    )


def test_block_rnafm_cdn_prevents_hub(monkeypatch):
    from physrna_filter.analysis import rnafm_embeddings as re

    re._hub_blocked = False
    import fm.pretrained as pretrained

    re._block_rnafm_cdn()
    with pytest.raises(RuntimeError, match="CUHK CDN"):
        pretrained.load_fm_model_and_alphabet_hub("rna_fm_t12")

    with pytest.raises(RuntimeError, match="without model_location"):
        pretrained.rna_fm_t12()


def test_resolve_finds_env_path_with_quotes(monkeypatch, tmp_path):
    from physrna_filter.analysis import rnafm_embeddings as re

    weights = tmp_path / "RNA-FM_pretrained.pth"
    weights.write_bytes(b"\x00" * (re.MIN_WEIGHT_BYTES + 1))

    monkeypatch.setenv("RNAFM_CHECKPOINT", f'"{weights}"')
    assert re.resolve_weights_path() == weights.resolve()


def test_load_checkpoint_uses_weights_only_false(monkeypatch, tmp_path):
    from physrna_filter.analysis.rnafm_embeddings import _load_checkpoint

    path = tmp_path / "test.pth"
    path.write_bytes(b"fake")
    captured = {}

    def fake_load(*args, **kwargs):
        captured.update(kwargs)
        return {"model": {}, "args": None}

    monkeypatch.setattr("physrna_filter.analysis.rnafm_embeddings.torch.load", fake_load)
    _load_checkpoint(path)
    assert captured.get("weights_only") is False


def test_stale_rnafm_cache_rejected(monkeypatch, tmp_path):
    from physrna_filter.analysis import rnafm_embeddings as re

    re._CACHE_DIR = tmp_path
    re._model = object()
    re._load_failed = False

    cache_key = re._cache_key("1abc", "R", "AUGC")
    emb_path = tmp_path / f"{cache_key}.npy"
    idx_path = tmp_path / f"{cache_key}.json"
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np = __import__("numpy")
    np.save(str(emb_path), np.zeros((4, re.RNAFM_FALLBACK), dtype=np.float32))
    idx_path.write_text("[1, 2, 3, 4]")

    monkeypatch.setattr(re, "_load_model", lambda: None)
    monkeypatch.setattr(re, "_fallback_embedding", lambda seq, resnums: {
        rn: torch.zeros(re.RNAFM_FALLBACK) for rn in resnums
    })

    result = re.get_rnafm_embeddings("1abc", "R", "AUGC", [1, 2, 3, 4])
    assert all(v.shape[0] == re.RNAFM_FALLBACK for v in result.values())
    assert not emb_path.exists()
