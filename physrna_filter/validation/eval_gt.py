"""
Evaluate a trained PhysGT checkpoint on merged training data without retraining.

Use this after a fast deploy run (e.g. --n-folds 1) to score the full
mutation set or a held-out test split.

Examples
--------
    # Score all merged entries with your deployed checkpoint (in-sample):
    python -m physrna_filter.validation.eval_gt

    # Faster: reuse cached graphs from training
    python -m physrna_filter.validation.eval_gt ^
        --graph-cache physrna_filter/validation/gt_graphs_merged.pt

    # Unbiased estimate on ~15%% held-out complexes (no retraining):
    python -m physrna_filter.validation.eval_gt --holdout

    # ProNAB-only baseline:
    python -m physrna_filter.validation.eval_gt --holdout --pronab-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr

from ..data.fetch_training_data import fetch_training_data
from ..analysis.gt_inference import load_gt_model
from .train_gt import (
    DEVICE,
    _load_or_build_dataset,
    _per_source_metrics,
    _predict,
    _split_holdout_by_pdb,
)

CHECKPOINT_DIR = Path(__file__).parent
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "gt_checkpoint.pt"
DEFAULT_RESULTS = CHECKPOINT_DIR / "gt_eval_results.csv"


def eval_gt_checkpoint(
    checkpoint_path: str | None = None,
    max_entries: int | None = None,
    graph_cache: str | None = None,
    rebuild_cache: bool = False,
    holdout: bool = False,
    test_fraction: float = 0.15,
    val_fraction: float = 0.15,
    seed: int = 7,
    results_csv: str | None = None,
    include_nabe: bool = True,
    include_literature: bool = True,
    pronab_only: bool = False,
    fast_mutations: bool = True,
) -> pd.DataFrame:
    """
    Run inference with a fixed checkpoint.  Does not update model weights.
    """
    ckpt = checkpoint_path or str(DEFAULT_CHECKPOINT)
    model, meta = load_gt_model(ckpt)
    model = model.to(DEVICE)
    model.eval()

    esm_dim = meta["esm_dim"]
    rnafm_dim = meta["rnafm_dim"]
    use_esm = meta["use_esm"]
    use_rnafm = meta["use_rnafm"]
    target_mean = meta.get("target_mean", 0.0)
    target_std = meta.get("target_std", 1.0)

    print(f"Checkpoint: {meta.get('checkpoint')}")
    print(f"Entries: {'holdout test' if holdout else 'all'}  "
          f"ESM={use_esm}  RNA-FM={use_rnafm}")
    if meta.get("training_sources"):
        print(f"Checkpoint trained on sources: {meta['training_sources']}")

    entries_df = fetch_training_data(
        include_nabe=include_nabe and not pronab_only,
        include_literature=include_literature and not pronab_only,
    )
    if max_entries:
        entries_df = entries_df.head(max_entries)

    dataset, failures = _load_or_build_dataset(
        entries_df=entries_df,
        use_esm=use_esm,
        use_rnafm=use_rnafm,
        esm_dim=esm_dim,
        rnafm_dim=rnafm_dim,
        minimize_structures=False,
        reuse_wt_esm=False,
        fast_mutations=fast_mutations,
        graph_cache=graph_cache,
        rebuild_cache=rebuild_cache,
    )
    print(f"Loaded {len(dataset)} graph pairs ({len(failures)} build failures)")

    if holdout:
        _, _, test_rows = _split_holdout_by_pdb(
            dataset,
            test_fraction=test_fraction,
            val_fraction=val_fraction,
            seed=seed,
        )
        rows = test_rows
        split_label = "holdout test"
    else:
        rows = dataset
        split_label = "all entries"

    eval_data = [(wt, mut, target) for wt, mut, target, _, _, _ in rows]
    pdb_ids = [pid for _, _, _, pid, _, _ in rows]
    mutations = [mut for _, _, _, _, mut, _ in rows]
    sources = [src for _, _, _, _, _, src in rows]

    targets, preds = _predict(model, eval_data, target_mean, target_std)

    results_df = pd.DataFrame({
        "pdb_id": pdb_ids,
        "mutation": mutations,
        "source": sources,
        "experimental_ddg": targets,
        "gt_pred_ddg": preds,
    })

    out = Path(results_csv) if results_csv else DEFAULT_RESULTS
    results_df.to_csv(out, index=False)
    print(f"Saved {len(results_df)} predictions to {out}")

    if len(targets) >= 5:
        r_p, p_p = pearsonr(targets, preds)
        r_s, _ = spearmanr(targets, preds)
        print("\n" + "=" * 55)
        print(f"PhysGT eval ({split_label})  n={len(targets)}")
        print(f"  Pearson  r = {r_p:+.3f}  (p={p_p:.4f})")
        print(f"  Spearman r = {r_s:+.3f}")
        print("=" * 55)
        _per_source_metrics(targets, preds, sources)
        if holdout:
            from .holdout_report import summarize_holdout_csv

            summary_path = out.with_name(out.stem + "_summary.json")
            report = summarize_holdout_csv(out, expected_test_n=len(targets))
            summary_path.write_text(
                json.dumps(report, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"Saved holdout summary to {summary_path}")
        elif not holdout:
            print(
                "Note: all-entry scores are in-sample if the checkpoint was "
                "trained on the same data. Use --holdout or full LOCO for "
                "unbiased estimates."
            )

    return results_df


def main():
    ap = argparse.ArgumentParser(description="Evaluate PhysGT checkpoint on training data")
    ap.add_argument("--checkpoint", default=None,
                    help=f"Path to gt_checkpoint.pt (default: {DEFAULT_CHECKPOINT})")
    ap.add_argument("--max-entries", type=int, default=None)
    ap.add_argument("--graph-cache", default=None,
                    help="Reuse/resume graph builds; checkpoints every 25 pairs when set")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--holdout", action="store_true",
                    help="Evaluate on held-out test complexes only")
    ap.add_argument("--pronab-only", action="store_true")
    ap.add_argument("--no-nabe", action="store_true")
    ap.add_argument("--no-literature", action="store_true")
    ap.add_argument("--test-fraction", type=float, default=0.15)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--results-csv", default=None)
    ap.add_argument(
        "--fast-mutations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use coordinate-level mutations (required for RNA-chain edits; default: on)",
    )
    args = ap.parse_args()

    if args.holdout and not args.graph_cache:
        args.graph_cache = str(CHECKPOINT_DIR / "gt_graphs_merged.pt")

    eval_gt_checkpoint(
        checkpoint_path=args.checkpoint,
        max_entries=args.max_entries,
        graph_cache=args.graph_cache,
        rebuild_cache=args.rebuild_cache,
        holdout=args.holdout,
        test_fraction=args.test_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
        results_csv=args.results_csv,
        include_nabe=not args.no_nabe and not args.pronab_only,
        include_literature=not args.no_literature and not args.pronab_only,
        pronab_only=args.pronab_only,
        fast_mutations=args.fast_mutations,
    )


if __name__ == "__main__":
    main()
