"""
Rank AlphaFold 3 protein–RNA candidates without hand-built panel JSON.

Typical workflow (biologist-friendly)::

    physrna init
    physrna rank af3_predictions --rbp LIN28A

    # Open ranked_candidates_report.html in a browser

Optional manifest CSV columns: job_name, zip_file, rbp_name, rna_sequence,
chrom, genomic_start, genomic_end (for eCLIP). Template::

    python -m physrna_filter.validation.rank_af3_candidates --write-manifest-template candidates.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from ..analysis.gt_inference import GtInferenceContext
from ..data.candidate_manifest import (
    generate_swap_panel,
    load_candidates,
    save_panel_json,
    write_manifest_template,
)
from ..pipeline import run_pipeline
from ..structure.af3_confidence import af3_iptm, af3_ptm, af3_ranking_score
from ..structure.af3_io import is_af3_zip
from ..config import (
    checkpoint_is_finetuned,
    resolve_gt_checkpoint,
    writable_gt_checkpoint,
)
from ..data.af3_eval_panel import contrastive_pairs
from ..validation.finetune_af3_panel import finetune_af3_panel
from ..validation.report_af3 import write_html_report
from ..validation.screen_af3 import _composite_rank_score


def _unified_score(row: dict) -> float:
    """
    Higher is better. Blends AF3 ipTM (structure confidence) with PhysRNA
    composite (lower composite = better, so we negate it).
    """
    iptm = float(row.get("af3_iptm") or 0.0)
    comp = float(row.get("composite_score") or 0.0)
    return iptm - 0.08 * comp


def _parse_int(val: Any) -> int | None:
    if val is None or val == "":
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def rank_af3_candidates(
    zip_dir: str | Path,
    *,
    manifest_csv: str | Path | None = None,
    rbp_name: str | None = None,
    gt_checkpoint: str | Path | None = None,
    output_csv: str | Path = "ranked_candidates.csv",
    output_json: str | Path = "ranked_candidates_metrics.json",
    output_html: str | Path | None = "ranked_candidates_report.html",
    fast_mode: bool = False,
    deep_top: int | None = None,
    require_oxrna: bool = False,
    finetune: bool = False,
    no_finetune: bool = False,
    finetune_epochs: int = 50,
    max_negatives_per_positive: int = 3,
    model_rank: int = 0,
    include_controls: bool = False,
    require_gt_checkpoint: bool = True,
) -> tuple[pd.DataFrame, dict]:
    zip_dir = Path(zip_dir)
    ckpt = resolve_gt_checkpoint(gt_checkpoint)

    from ..validation.download_gt_checkpoint import ensure_public_checkpoint
    ensure_public_checkpoint(ckpt)

    candidates = load_candidates(
        zip_dir, manifest_csv, rbp_name=rbp_name, include_controls=include_controls
    )
    if not candidates:
        hint = (
            f"No AF3 zips matched rbp filter {rbp_name!r} under {zip_dir}. "
            "Use --all to rank every zip, or check job names contain the protein token."
            if rbp_name else
            f"No AF3 zips found in {zip_dir}."
        )
        raise RuntimeError(hint)
    if rbp_name:
        print(
            f"Ranking {len(candidates)} job(s) matching RBP filter {rbp_name!r} "
            f"(wrong-partner controls excluded unless --include-controls)"
        )

    swap_panel = generate_swap_panel(
        candidates, max_negatives_per_positive=max_negatives_per_positive
    )
    n_pairs = len(contrastive_pairs(swap_panel))
    do_finetune = finetune or (
        not no_finetune
        and len(candidates) >= 2
        and n_pairs >= 1
        and not checkpoint_is_finetuned(ckpt)
    )
    if finetune and no_finetune:
        do_finetune = False

    panel_path: Path | None = None
    if do_finetune:
        panel_path = zip_dir / "_physrna_auto_panel.json"
        save_panel_json(swap_panel, panel_path)
        ft_ckpt = writable_gt_checkpoint(gt_checkpoint)
        print(
            f"Contrastive fine-tune: {n_pairs} pair(s), {finetune_epochs} epochs "
            f"-> {ft_ckpt}"
        )
        finetune_af3_panel(
            zip_dir=zip_dir,
            checkpoint_path=ft_ckpt,
            epochs=finetune_epochs,
            model_rank=model_rank,
            panel_path=panel_path,
        )
        ckpt = ft_ckpt
    elif checkpoint_is_finetuned(ckpt):
        print(f"Using fine-tuned checkpoint: {ckpt}")

    ctx = GtInferenceContext(checkpoint_path=str(ckpt) if ckpt.exists() else None)
    if ckpt.exists():
        ctx.ensure_loaded(str(ckpt), require_trained=require_gt_checkpoint)

    rows: list[dict] = []
    for cand in candidates:
        zip_path = Path(cand.get("zip_path") or "")
        if not zip_path.is_file():
            from ..data.candidate_manifest import resolve_zip_path
            found = resolve_zip_path(cand, zip_dir)
            if found is None:
                rows.append({
                    "job_name": cand.get("job_name"),
                    "candidate_id": cand.get("id"),
                    "combined_verdict": "ERROR",
                    "error": "zip not found",
                })
                continue
            zip_path = found

        print(f"\n=== {cand.get('job_name', zip_path.name)} ===")
        try:
            result = run_pipeline(
                pdb_path=str(zip_path),
                verbose=False,
                rbp_name=cand.get("rbp_name") or rbp_name,
                rna_sequence=cand.get("rna_sequence"),
                chrom=cand.get("chrom"),
                genomic_start=_parse_int(cand.get("genomic_start")),
                genomic_end=_parse_int(cand.get("genomic_end")),
                gt_checkpoint=str(ckpt) if ckpt.exists() else None,
                model_rank=model_rank,
                fast_mode=fast_mode and not require_oxrna,
                require_oxrna=require_oxrna,
                require_gt_checkpoint=require_gt_checkpoint,
                inference_context=ctx,
            )
            row = {
                "candidate_id": cand.get("id"),
                "job_name": cand.get("job_name") or zip_path.stem,
                "file": str(zip_path),
                "rbp_name": cand.get("rbp_name") or rbp_name,
                "rna_sequence": cand.get("rna_sequence"),
                "partner_group": cand.get("partner_group"),
                "chrom": cand.get("chrom"),
                "genomic_start": cand.get("genomic_start"),
                "genomic_end": cand.get("genomic_end"),
                "af3_iptm": af3_iptm(zip_path, model_rank=model_rank),
                "af3_ptm": af3_ptm(zip_path, model_rank=model_rank),
                "af3_ranking_score": af3_ranking_score(zip_path, model_rank=model_rank),
                "combined_verdict": result["combined_verdict"],
                "confidence": result["confidence"],
                "gt_score_norm": result["gt_score_norm"],
                "gt_verdict": result["gt_verdict"],
                "bio_verdict": result["bio_verdict"],
                "eclip_supported": result.get("eclip_supported"),
                "clash_n_severe": result["clash_n_severe"],
                "geom_verdict": result["geom_verdict"],
                "simulation_method": result.get("simulation_method"),
                "screen_mode": "fast" if (fast_mode and not require_oxrna) else "deep",
            }
            row["composite_score"] = _composite_rank_score(row)
            row["unified_score"] = _unified_score(row)
            rows.append(row)
            print(
                f"  composite={row['composite_score']:.2f}  "
                f"iptm={row['af3_iptm']}  bio={row['bio_verdict']}  "
                f"combined={row['combined_verdict']}"
            )
        except Exception as exc:
            msg = str(exc).splitlines()[0][:200]
            print(f"  FAILED: {msg}", file=sys.stderr)
            rows.append({
                "job_name": cand.get("job_name"),
                "candidate_id": cand.get("id"),
                "file": str(zip_path),
                "combined_verdict": "ERROR",
                "error": msg,
            })

    df = pd.DataFrame(rows)

    # Optional second pass: oxRNA MD on top-N by fast composite
    if deep_top and deep_top > 0 and fast_mode and not require_oxrna:
        scored = df[df["composite_score"].notna()].copy()
        if not scored.empty:
            top_ids = scored.nsmallest(deep_top, "composite_score")["file"].tolist()
            print(f"\n--- Deep oxRNA pass on top {len(top_ids)} finalist(s) ---")
            for path in top_ids:
                cand = next((c for c in candidates if c.get("zip_path") == path), {})
                try:
                    result = run_pipeline(
                        pdb_path=path,
                        verbose=False,
                        rbp_name=cand.get("rbp_name") or rbp_name,
                        rna_sequence=cand.get("rna_sequence"),
                        chrom=cand.get("chrom"),
                        genomic_start=_parse_int(cand.get("genomic_start")),
                        genomic_end=_parse_int(cand.get("genomic_end")),
                        gt_checkpoint=str(ckpt) if ckpt.exists() else None,
                        model_rank=model_rank,
                        fast_mode=False,
                        require_oxrna=True,
                        require_gt_checkpoint=require_gt_checkpoint,
                        inference_context=ctx,
                    )
                    idx = df.index[df["file"] == path][0]
                    df.loc[idx, "combined_verdict"] = result["combined_verdict"]
                    df.loc[idx, "gt_score_norm"] = result["gt_score_norm"]
                    df.loc[idx, "bio_verdict"] = result["bio_verdict"]
                    df.loc[idx, "simulation_method"] = result.get("simulation_method")
                    row_dict = df.loc[idx].to_dict()
                    df.loc[idx, "composite_score"] = _composite_rank_score(row_dict)
                    df.loc[idx, "unified_score"] = _unified_score(row_dict)
                    df.loc[idx, "screen_mode"] = "deep_finalist"
                except Exception as exc:
                    print(f"  Deep pass failed for {path}: {exc}", file=sys.stderr)

    if "composite_score" in df.columns:
        df = df.sort_values(
            ["composite_score", "unified_score"],
            ascending=[True, False],
            kind="mergesort",
        )
        df.insert(0, "rank", range(1, len(df) + 1))

    metrics: dict[str, Any] = {
        "n_candidates": len(df),
        "n_error": int((df.get("combined_verdict") == "ERROR").sum()),
        "screen_mode": "fast" if fast_mode and not require_oxrna else "deep",
        "deep_finalists": deep_top or 0,
        "finetuned": checkpoint_is_finetuned(ckpt),
        "auto_finetune": do_finetune,
    }
    if "af3_iptm" in df.columns and df["af3_iptm"].notna().any():
        metrics["af3_iptm_mean"] = float(df["af3_iptm"].astype(float).mean())
    if "composite_score" in df.columns and df["composite_score"].notna().any():
        metrics["composite_best"] = float(df["composite_score"].astype(float).min())
    if "bio_verdict" in df.columns:
        metrics["bio_fail_rate"] = float(
            (df["bio_verdict"] == "FAIL").sum() / max(len(df), 1)
        )

    out_csv = Path(output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    with open(output_json, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"\nSaved ranked CSV to {out_csv}")
    print(f"Saved metrics to {output_json}")
    if output_html:
        html_path = write_html_report(
            out_csv,
            output_html,
            title="PhysRNA candidate ranking",
            metrics_json=output_json,
        )
        print(f"Saved HTML report to {html_path}")

    return df, metrics


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Rank AF3 protein–RNA candidates (drop zips in a folder — no panel JSON)"
    )
    ap.add_argument("--zips", default="af3_predictions", help="Folder with AF3 Server zips")
    ap.add_argument("--manifest", default=None, help="Optional candidates.csv manifest")
    ap.add_argument("--rbp-name", default=None,
                    help="Only rank jobs for this RBP (filters by AF3 job name)")
    ap.add_argument("--all", action="store_true",
                    help="Rank every zip in the folder (ignore --rbp-name filter)")
    ap.add_argument("--include-controls", action="store_true",
                    help="Include wrong-partner N* panel jobs in ranking")
    ap.add_argument("--gt-checkpoint", default=None,
                    help="Checkpoint path (default: ~/.physrna or shipped)")
    ap.add_argument("--output", "-o", default="ranked_candidates.csv")
    ap.add_argument("--metrics-json", default="ranked_candidates_metrics.json")
    ap.add_argument("--html", default="ranked_candidates_report.html",
                    help="HTML report path (pass '' to skip)")
    ap.add_argument(
        "--fast",
        action="store_true",
        help="Quick triage only (skip oxRNA MD; use for first-pass screening)",
    )
    ap.add_argument(
        "--deep-top",
        type=int,
        default=3,
        help="With --fast: re-run oxRNA on top N finalists (default 3)",
    )
    ap.add_argument("--require-oxrna", action="store_true",
                    help="Fail if oxRNA unavailable (default: internal fallback)")
    ap.add_argument("--finetune", action="store_true",
                    help="Force contrastive fine-tune on your candidates")
    ap.add_argument("--no-finetune", action="store_true",
                    help="Skip auto fine-tune when multiple RNAs share one protein")
    ap.add_argument("--finetune-epochs", type=int, default=50)
    ap.add_argument("--max-negatives", type=int, default=3,
                    help="Wrong-RNA swaps per candidate for auto finetune")
    ap.add_argument("--write-manifest-template", metavar="PATH", default=None)
    ap.add_argument("--model-rank", type=int, default=0)
    ap.add_argument("--allow-physics-only", action="store_true")
    args = ap.parse_args()

    if args.write_manifest_template:
        path = write_manifest_template(args.write_manifest_template)
        print(f"Wrote manifest template to {path}")
        return

    fast = args.fast
    deep_top = args.deep_top if fast else None
    rbp = None if args.all else args.rbp_name

    rank_af3_candidates(
        zip_dir=args.zips,
        manifest_csv=args.manifest,
        rbp_name=rbp,
        include_controls=args.include_controls,
        gt_checkpoint=args.gt_checkpoint,
        output_csv=args.output,
        output_json=args.metrics_json,
        output_html=args.html or None,
        fast_mode=fast,
        deep_top=deep_top,
        require_oxrna=args.require_oxrna and not fast,
        finetune=args.finetune,
        no_finetune=args.no_finetune,
        finetune_epochs=args.finetune_epochs,
        max_negatives_per_positive=args.max_negatives,
        model_rank=args.model_rank,
        require_gt_checkpoint=not args.allow_physics_only,
    )


if __name__ == "__main__":
    main()
