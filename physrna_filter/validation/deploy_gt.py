"""
Deploy PhysGT for production: contrastive interface training then full ΔΔG training.

Run this before enabling the GT branch in the AF3 pipeline:

    python -m physrna_filter.validation.deploy_gt [--gpu]

Steps:
  1. Contrastive interface-head training (crystal positives vs entropic decoys)
  2. Full PhysGT LOCO training with ESM-2 + RNA-FM + bipartite cross-attention
  3. Merge interface head weights into final checkpoint
  4. Write gt_checkpoint.pt to validation/ for pipeline inference
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from .train_interface_head import train_interface_head
from .train_gt import run_gt_training

CHECKPOINT_DIR = Path(__file__).parent
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "gt_checkpoint.pt"
INTERFACE_CHECKPOINT = CHECKPOINT_DIR / "gt_interface_pretrain.pt"


def deploy_gt(
    max_entries: int | None = None,
    interface_epochs: int = 30,
    gt_epochs: int = 150,
    use_esm: bool = True,
    use_rnafm: bool = True,
    n_folds: int | None = None,
    fast_mutations: bool = False,
    output_path: str | None = None,
    include_nabe: bool = True,
    include_literature: bool = True,
    pronab_only: bool = False,
) -> Path:
    """
    Full deployment pipeline: interface contrastive pretrain → ΔΔG training.
    """
    out = Path(output_path) if output_path else DEFAULT_CHECKPOINT
    interface_out = str(INTERFACE_CHECKPOINT)

    print("\n" + "=" * 60)
    print("Step 0/3: PhysGT readiness checks")
    print("=" * 60)
    from .gt_readiness import run_gt_readiness, print_readiness_report
    report = run_gt_readiness(use_esm=use_esm, use_rnafm=use_rnafm)
    print_readiness_report(report)

    print("\n" + "=" * 60)
    print("Step 1/3: RNA-FM weights (local only — CUHK CDN never contacted)")
    print("=" * 60)
    if use_rnafm:
        from ..analysis.rnafm_embeddings import announce_rnafm_mode, resolve_weights_path
        mode = announce_rnafm_mode()
        skip_dl = os.environ.get("RNAFM_SKIP_DOWNLOAD", "").lower() in ("1", "true", "yes")
        if mode == "fallback" and not resolve_weights_path() and not skip_dl:
            print(
                "No valid local weights — attempting one-time Hugging Face download ..."
            )
            try:
                from ..data.download_rnafm_weights import download_rnafm_weights
                download_rnafm_weights()
                announce_rnafm_mode()
            except SystemExit:
                print(
                    "RNA-FM download failed — continuing with fallback encoding.\n"
                    "  Verify manual download: python -m physrna_filter.data.verify_rnafm_weights\n"
                    "  Skip this step next time: set RNAFM_SKIP_DOWNLOAD=1"
                )
        elif mode == "fallback":
            print(
                "Run: python -m physrna_filter.data.verify_rnafm_weights"
            )

    print("\n" + "=" * 60)
    print("Step 2/3: Contrastive interface-head training")
    print("=" * 60)
    train_interface_head(
        max_entries=max_entries,
        epochs=interface_epochs,
        use_esm=use_esm,
        use_rnafm=use_rnafm,
        output_path=interface_out,
        include_nabe=include_nabe and not pronab_only,
        include_literature=include_literature and not pronab_only,
        pronab_only=pronab_only,
    )

    print("\n" + "=" * 60)
    print("Step 3/3: Full PhysGT LOCO training (ESM-2 + RNA-FM)")
    print("=" * 60)
    run_gt_training(
        max_entries=max_entries,
        epochs=gt_epochs,
        use_esm=use_esm,
        use_rnafm=use_rnafm,
        n_folds=n_folds,
        fast_mutations=fast_mutations,
        output_path=str(out),
        results_csv=str(CHECKPOINT_DIR / "gt_results.csv"),
        include_nabe=include_nabe and not pronab_only,
        include_literature=include_literature and not pronab_only,
        pronab_only=pronab_only,
    )

    # Merge interface-head weights from step 1 into final checkpoint
    if INTERFACE_CHECKPOINT.exists() and out.exists():
        iface_ckpt = torch.load(INTERFACE_CHECKPOINT, map_location="cpu", weights_only=False)
        final_ckpt = torch.load(out, map_location="cpu", weights_only=False)

        iface_state = iface_ckpt.get("model_state", {})
        final_state = final_ckpt.get("model_state", {})

        merged = 0
        for key, val in iface_state.items():
            if any(
                tag in key
                for tag in ("interface_head", "cross_attn", "edge_gates", "physics_bias")
            ):
                final_state[key] = val
                merged += 1

        final_ckpt["model_state"] = final_state
        final_ckpt["interface_head_trained"] = True
        final_ckpt["deployed"] = True
        torch.save(final_ckpt, out)
        print(f"\nMerged {merged} interface/cross-attn weights into {out}")

    print(f"\nDeployment complete: {out}")
    print("GT branch is ready for production inference.")
    return out


def ensure_gt_checkpoint(
    checkpoint_path: str | None = None,
    auto_train: bool = False,
    max_entries: int = 50,
) -> bool:
    """
    Check that a production-ready GT checkpoint exists.

    Returns True if checkpoint has interface_head_trained flag.
    If auto_train=True and missing, runs abbreviated deploy_gt.
    """
    path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
    if not path.exists():
        if auto_train:
            print("No GT checkpoint found — running abbreviated training ...")
            deploy_gt(max_entries=max_entries, interface_epochs=10, gt_epochs=30, n_folds=3)
            return path.exists()
        return False

    payload = torch.load(path, map_location="cpu", weights_only=False)
    return bool(payload.get("interface_head_trained", False))


def main():
    ap = argparse.ArgumentParser(description="Deploy PhysGT for production")
    ap.add_argument("--max-entries", type=int, default=None)
    ap.add_argument("--interface-epochs", type=int, default=30)
    ap.add_argument("--gt-epochs", type=int, default=150)
    ap.add_argument("--n-folds", type=int, default=None)
    ap.add_argument("--no-esm", action="store_true")
    ap.add_argument("--no-rnafm", action="store_true")
    ap.add_argument("--slow-mutations", action="store_true")
    ap.add_argument("--pronab-only", action="store_true",
                    help="Train on ProNAB scrape only (exclude Nabe/literature)")
    ap.add_argument("--no-nabe", action="store_true")
    ap.add_argument("--no-literature", action="store_true")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    deploy_gt(
        max_entries=args.max_entries,
        interface_epochs=args.interface_epochs,
        gt_epochs=args.gt_epochs,
        use_esm=not args.no_esm,
        use_rnafm=not args.no_rnafm,
        n_folds=args.n_folds,
        fast_mutations=not args.slow_mutations,
        output_path=args.output,
        include_nabe=not args.no_nabe and not args.pronab_only,
        include_literature=not args.no_literature and not args.pronab_only,
        pronab_only=args.pronab_only,
    )


if __name__ == "__main__":
    main()
