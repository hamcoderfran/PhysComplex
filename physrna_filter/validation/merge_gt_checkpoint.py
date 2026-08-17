"""
Merge a contrastive interface-head checkpoint into the main PhysGT checkpoint.

After ``train_interface_head``:

    python -m physrna_filter.validation.merge_gt_checkpoint

Or with explicit paths:

    python -m physrna_filter.validation.merge_gt_checkpoint \\
        --interface physrna_filter/validation/gt_interface_pretrain.pt \\
        --final physrna_filter/validation/gt_checkpoint.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

CHECKPOINT_DIR = Path(__file__).parent
DEFAULT_INTERFACE = CHECKPOINT_DIR / "gt_interface_pretrain.pt"
DEFAULT_FINAL = CHECKPOINT_DIR / "gt_checkpoint.pt"

_MERGE_KEY_TAGS = ("interface_head", "cross_attn", "edge_gates", "physics_bias", "interface_ddg_coupling")


def merge_interface_into_checkpoint(
    interface_path: str | Path,
    final_path: str | Path,
    *,
    mark_deployed: bool = True,
) -> int:
    iface = torch.load(interface_path, map_location="cpu", weights_only=False)
    final = torch.load(final_path, map_location="cpu", weights_only=False)

    merged = 0
    for key, val in iface.get("model_state", {}).items():
        if any(tag in key for tag in _MERGE_KEY_TAGS):
            final.setdefault("model_state", {})[key] = val
            merged += 1

    final["interface_head_trained"] = True
    if mark_deployed:
        final["deployed"] = True

    torch.save(final, final_path)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge interface-head weights into gt_checkpoint.pt")
    ap.add_argument("--interface", default=str(DEFAULT_INTERFACE))
    ap.add_argument("--final", default=str(DEFAULT_FINAL))
    ap.add_argument("--no-deployed-flag", action="store_true")
    args = ap.parse_args()

    iface_path = Path(args.interface)
    final_path = Path(args.final)
    if not iface_path.exists():
        raise SystemExit(f"Interface checkpoint not found: {iface_path}")
    if not final_path.exists():
        raise SystemExit(f"Final checkpoint not found: {final_path}")

    n = merge_interface_into_checkpoint(
        iface_path,
        final_path,
        mark_deployed=not args.no_deployed_flag,
    )
    print(f"Merged {n} interface/cross-attn keys into {final_path}")
    print("interface_head_trained=True — GT branch ready for AF3 screening.")


if __name__ == "__main__":
    main()
