"""
Verify RNA-FM pretrained weights are present and valid.

Run this after a manual download if training still reports HTTP 403:

    python -m physrna_filter.data.verify_rnafm_weights

Expected file size: ~1.19 GB (1,190,000,000 bytes).
Common failure: browser saved the CUHK 403 HTML page as RNA-FM_pretrained.pth
(only a few KB).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

from physrna_filter.analysis.rnafm_embeddings import (
    MIN_WEIGHT_BYTES,
    RNAFM_FILENAME,
    candidate_weight_paths,
    describe_weight_search,
    is_valid_weights_file,
    resolve_weights_path,
    _WEIGHTS_DIR,
)


def rnafm_weights_available() -> bool:
    """True if valid RNA-FM weights are on disk (no console output)."""
    return resolve_weights_path() is not None


def verify_rnafm_weights(try_load: bool = True, *, quiet: bool = False) -> Path | None:
    if not quiet:
        print("RNA-FM weight verification")
        print("=" * 60)

    env_ckpt = os.environ.get("RNAFM_CHECKPOINT", "")
    if not quiet:
        if env_ckpt:
            print(f"RNAFM_CHECKPOINT = {env_ckpt!r}")
        else:
            print("RNAFM_CHECKPOINT = (not set)")
        print(f"Default directory: {_WEIGHTS_DIR}")
        print(f"Torch hub cache:   {Path(torch.hub.get_dir()) / 'checkpoints' / RNAFM_FILENAME}")
        print()

    found_any = False
    for row in describe_weight_search():
        if quiet:
            if row["valid"]:
                found_any = True
            continue
        status = "OK" if row["valid"] else ("INVALID" if row["exists"] else "missing")
        size_mb = row["size_bytes"] / 1e6
        line = f"  [{status:7}] {row['path']}"
        if row["exists"]:
            line += f"  ({size_mb:.1f} MB)"
        print(line)
        if row.get("reason"):
            print(f"           -> {row['reason']}")
        if row["valid"]:
            found_any = True

    resolved = resolve_weights_path()
    if quiet:
        return resolved

    print()
    if resolved is None:
        print("RESULT: No valid RNA-FM weights found.")
        print()
        print("Fix:")
        print("  1. Download ~1.2 GB checkpoint from Hugging Face:")
        print("       python -m physrna_filter.data.download_rnafm_weights")
        print("  2. Or copy your file and set:")
        print(f"       set RNAFM_CHECKPOINT=C:\\path\\to\\{RNAFM_FILENAME}")
        print(f"  3. Or place file at:")
        print(f"       {_WEIGHTS_DIR / RNAFM_FILENAME}")
        return None

    print(f"RESULT: Valid weights at {resolved}")
    print(f"       Size: {resolved.stat().st_size / 1e9:.3f} GB")

    if try_load:
        try:
            import fm.pretrained as pretrained
            from physrna_filter.analysis.rnafm_embeddings import (
                _block_rnafm_cdn,
                _load_checkpoint,
            )
            _block_rnafm_cdn()
            print("Loading checkpoint (local only, no network) ...")
            model_data = _load_checkpoint(resolved)
            model, alphabet = pretrained.load_model_and_alphabet_core(
                resolved.stem, model_data, regression_data=None, theme="rna"
            )
            del model, alphabet
            print("torch.load: OK")
        except ImportError:
            print("rna-fm not installed — skipping load test (pip install rna-fm)")
        except Exception as e:
            print(f"torch.load FAILED: {e}", file=sys.stderr)
            return None

    print()
    print("Training will use real RNA-FM embeddings (dim=640).")
    print(f"Recommended: set RNAFM_CHECKPOINT={resolved}")
    return resolved


def main():
    path = verify_rnafm_weights()
    raise SystemExit(0 if path else 1)


if __name__ == "__main__":
    main()
