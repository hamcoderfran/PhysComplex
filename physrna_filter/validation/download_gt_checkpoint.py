"""
Ensure the pre-trained PhysGT checkpoint is available for screening.

The public checkpoint ships in the repository at
``physrna_filter/validation/gt_checkpoint.pt``. Fine-tuned weights belong in
``~/.physrna/gt_checkpoint.pt`` (see ``physrna_filter.config``).
"""
from __future__ import annotations

from pathlib import Path

from ..config import shipped_checkpoint, user_checkpoint

DEFAULT_CHECKPOINT = shipped_checkpoint()


def ensure_public_checkpoint(
    checkpoint_path: str | Path | None = None,
) -> Path:
    """
    Return path to a usable checkpoint, raising with instructions if absent.
    """
    path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
    if path.is_file() and path.stat().st_size > 10_000:
        return path

    user = user_checkpoint()
    if user.is_file() and user.stat().st_size > 10_000:
        return user

    msg = f"""
PhysGT checkpoint not found: {path}

Quick fix:
  physrna init
  # or: python -m physrna_filter.cli init

This copies the shipped checkpoint (~4.5 MB) to ~/.physrna/gt_checkpoint.pt.
Fine-tunes are saved there and are never overwritten by git pull.

If you cloned without large files:
  git pull origin main
  # physrna_filter/validation/gt_checkpoint.pt should be present

Or train once:
  python -m physrna_filter.validation.deploy_gt --n-folds 1 --gt-epochs 80 --interface-epochs 15
""".strip()
    raise FileNotFoundError(msg)


def main() -> None:
    import argparse

    from ..config import resolve_gt_checkpoint

    ap = argparse.ArgumentParser(description="Verify pre-trained PhysGT checkpoint")
    ap.add_argument("--checkpoint", default=None, help="Explicit checkpoint path")
    args = ap.parse_args()
    path = resolve_gt_checkpoint(args.checkpoint)
    finetuned = ""
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("af3_panel_finetuned"):
            finetuned = " (AF3 panel fine-tuned)"
    except Exception:
        pass
    print(f"OK: {path} ({path.stat().st_size // 1024} KB){finetuned}")


if __name__ == "__main__":
    main()
