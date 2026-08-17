"""
Pre-flight checks before PhysGT training or AF3 pipeline scoring.
"""
from __future__ import annotations

import os
from pathlib import Path

from ..analysis.gt_constants import CHECKPOINT_SCHEMA_VERSION, EDGE_DIM, PHYSICS_SUMMARY_DIM


def check_rnafm_weights(use_rnafm: bool = True) -> tuple[bool, str]:
    if not use_rnafm:
        return True, "RNA-FM disabled"
    from ..analysis.rnafm_embeddings import announce_rnafm_mode, effective_rnafm_feature_dim

    mode = announce_rnafm_mode()
    dim = effective_rnafm_feature_dim(True)
    if dim < 100:
        return False, (
            f"RNA-FM fallback active (dim={dim}). "
            "Run: python -m physrna_filter.data.verify_rnafm_weights"
        )
    return True, f"RNA-FM OK ({mode}, dim={dim})"


def check_esm_available(use_esm: bool = True) -> tuple[bool, str]:
    if not use_esm:
        return True, "ESM-2 disabled"
    try:
        import esm  # noqa: F401
        from ..analysis.esm_embeddings import ESM_DIM
        return True, f"ESM-2 available (dim={ESM_DIM})"
    except ImportError:
        return False, "fair-esm not installed (pip install fair-esm)"


def check_torch_geometric() -> tuple[bool, str]:
    try:
        import torch_geometric  # noqa: F401
        return True, "torch-geometric installed"
    except ImportError:
        return False, "torch-geometric missing — FallbackMLP only"


def validate_checkpoint(payload: dict) -> tuple[bool, list[str]]:
    """Return (ok, warnings) for a loaded checkpoint dict."""
    warnings: list[str] = []
    schema = payload.get("schema_version", 1)
    if schema < CHECKPOINT_SCHEMA_VERSION:
        warnings.append(
            f"Checkpoint schema v{schema} is older than current v{CHECKPOINT_SCHEMA_VERSION}; "
            "re-run deploy_gt for extended physics features."
        )
    if not payload.get("interface_head_trained", False):
        warnings.append("interface_head_trained=False — AF3 scoring uses physics fallback.")
    edge_dim = payload.get("edge_dim", 9)
    if edge_dim < EDGE_DIM:
        warnings.append(f"edge_dim={edge_dim} < current {EDGE_DIM}")
    if not payload.get("model_state"):
        return False, ["checkpoint missing model_state"]
    return True, warnings


def run_gt_readiness(
    *,
    use_esm: bool = True,
    use_rnafm: bool = True,
    checkpoint_path: str | None = None,
) -> dict:
    """Run all readiness checks; returns report dict with ``ok`` flag."""
    checks: list[dict] = []

    for name, fn in (
        ("esm", lambda: check_esm_available(use_esm)),
        ("rnafm", lambda: check_rnafm_weights(use_rnafm)),
        ("torch_geometric", check_torch_geometric),
    ):
        ok, msg = fn()
        checks.append({"name": name, "ok": ok, "message": msg})

    if checkpoint_path and os.path.exists(checkpoint_path):
        import torch
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        ok, warns = validate_checkpoint(payload)
        checks.append({
            "name": "checkpoint",
            "ok": ok,
            "message": "; ".join(warns) if warns else "checkpoint valid",
        })

    all_ok = all(c["ok"] for c in checks)
    return {"ok": all_ok, "checks": checks, "schema_version": CHECKPOINT_SCHEMA_VERSION}


def print_readiness_report(report: dict) -> None:
    print("PhysGT readiness")
    print("-" * 40)
    for c in report["checks"]:
        status = "OK" if c["ok"] else "FAIL"
        print(f"  [{status}] {c['name']}: {c['message']}")
    print("-" * 40)
    print("READY" if report["ok"] else "NOT READY — fix failures above")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Verify PhysGT training/inference readiness")
    ap.add_argument("--checkpoint", default=None, help="Optional checkpoint to validate")
    ap.add_argument("--no-esm", action="store_true")
    ap.add_argument("--no-rnafm", action="store_true")
    args = ap.parse_args()
    report = run_gt_readiness(
        use_esm=not args.no_esm,
        use_rnafm=not args.no_rnafm,
        checkpoint_path=args.checkpoint,
    )
    print_readiness_report(report)
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
