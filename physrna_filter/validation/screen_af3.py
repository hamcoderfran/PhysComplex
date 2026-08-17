"""
Batch-screen AlphaFold 3 (or RoseTTAFold) protein-RNA predictions for hallucinations.

Runs the full PhysRNA-Filter pipeline on each structure and writes a CSV summary.
Use after training/deploying PhysGT (gt_checkpoint.pt).

Examples
--------
    python -m physrna_filter.validation.screen_af3 my_af3_outputs/
    python -m physrna_filter.validation.screen_af3 complex1.pdb complex2.cif
    python -m physrna_filter.validation.screen_af3 fold_6sqn_u1a_hairpin.zip --require-gt-checkpoint
    python -m physrna_filter.validation.screen_af3 af3_pdbs/ --output af3_screen.csv --require-gt-checkpoint

Evaluation panel (P1–P5 / N1–N5) with per-job RBP metadata and partner ranking::

    python -m physrna_filter.validation.screen_af3 af3_predictions/ \\
        --panel-json physrna_filter/data/af3_eval_panel.json \\
        --gt-checkpoint physrna_filter/validation/gt_checkpoint.pt \\
        --output eval_panel.csv \\
        --partner-summary
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ..data.af3_eval_panel import load_af3_eval_panel, match_panel_entry, partner_groups
from ..pipeline import run_pipeline
from ..analysis.gt_inference import GtInferenceContext
from ..structure.af3_io import (
    collect_structure_inputs,
    is_af3_zip,
    is_structure_file,
    resolve_structure_path,
)
from ..structure.af3_confidence import af3_iptm, af3_ptm, af3_ranking_score
from ..config import resolve_gt_checkpoint
from ..validation.download_gt_checkpoint import ensure_public_checkpoint


def _collect_screen_jobs(
    paths: list[str],
    model_rank: int = 0,
) -> list[tuple[str, Path]]:
    """
    Return (display_label, structure_path) for each input.

    AF3 Server zips are expanded to their top-ranked mmCIF; the zip path is kept
    as the display label in the output CSV.
    """
    seen: set[str] = set()
    jobs: list[tuple[str, Path]] = []

    def add(label: str, structure: Path) -> None:
        key = str(structure.resolve())
        if key not in seen:
            seen.add(key)
            jobs.append((label, structure))

    for raw in paths:
        p = Path(raw)
        if not p.exists():
            print(f"WARNING: skipping missing path {p}", file=sys.stderr)
            continue
        if p.is_dir():
            for child in collect_structure_inputs([str(p)], model_rank=model_rank):
                label = str(p / child.name) if child.parent == p else str(child)
                add(label, child)
        elif is_af3_zip(p):
            add(str(p), resolve_structure_path(p, model_rank=model_rank))
        elif is_structure_file(p):
            add(str(p), p.resolve())

    return jobs


def _composite_rank_score(row: dict) -> float:
    """
    Lower is better for ranking AF3 candidates.

    Uses normalized GT score plus light penalties for clashes and failed biology.
    Clashes are scaled by interface size so large native interfaces are not
    automatically penalized relative to compact cross-partner decoys.
    """
    gt = float(row.get("gt_score_norm", row.get("gt_score", 0.0)) or 0.0)
    clash = float(row.get("clash_n_severe", 0) or 0)
    n_iface = int(row.get("n_interface_nucleotides") or 0)
    clash_scale = max(n_iface, 12)
    clash_pen = 0.8 * (clash / clash_scale)
    bio = row.get("bio_verdict", "UNKNOWN")
    bio_pen = 0.0
    if bio == "FAIL":
        bio_pen = 3.0
    elif bio == "WARN":
        bio_pen = 1.0
    return gt + clash_pen + bio_pen


def _print_partner_summary(df: pd.DataFrame) -> None:
    if "partner_group" not in df.columns or "composite_score" not in df.columns:
        return

    print("\nPartner-group ranking (lower composite_score = preferred):")
    for group, sub in df.groupby("partner_group", dropna=True):
        if not str(group).strip():
            continue
        ranked = sub.sort_values("composite_score", kind="mergesort")
        ranked = ranked[ranked["composite_score"].notna()]
        if ranked.empty:
            continue
        print(f"\n  [{group}]")
        for _, row in ranked.iterrows():
            tag = row.get("panel_id", "?")
            label = row.get("panel_label", "")
            print(
                f"    {tag:3s} ({label:8s})  "
                f"composite={row['composite_score']:.2f}  "
                f"gt_norm={row.get('gt_score_norm', float('nan')):.2f}  "
                f"bio={row.get('bio_verdict', '?')}"
            )
        positives = ranked[ranked.get("panel_label", pd.Series(dtype=str)) == "positive"]
        if len(positives):
            best_pos = positives.iloc[0]
            best_overall = ranked.iloc[0]
            if best_overall.get("panel_label") == "positive":
                print(f"    -> positive ranks #1 in group")
            else:
                print(
                    f"    -> WARNING: best overall is {best_overall.get('panel_id')} "
                    f"(negative); positive best composite={best_pos['composite_score']:.2f}"
                )


def _af3_zip_from_label(label: str) -> Path | None:
    p = Path(label)
    if p.suffix.lower() == ".zip" and p.is_file():
        return p
    return None


def _unified_score(row: dict) -> float:
    iptm = float(row.get("af3_iptm") or 0.0)
    comp = float(row.get("composite_score") or 0.0)
    return iptm - 0.08 * comp


def screen_af3_structures(
    inputs: list[str],
    output_csv: str = "af3_screen_results.csv",
    cutoff: float = 5.0,
    gt_checkpoint: str | None = None,
    require_gt_checkpoint: bool = False,
    rbp_name: str | None = None,
    rna_sequence: str | None = None,
    panel_json: str | None = None,
    quiet: bool = False,
    model_rank: int = 0,
    partner_summary: bool = False,
    fast_mode: bool = False,
    require_oxrna: bool = False,
    chrom: str | None = None,
    genomic_start: int | None = None,
    genomic_end: int | None = None,
) -> pd.DataFrame:
    jobs = _collect_screen_jobs(inputs, model_rank=model_rank)
    if not jobs:
        raise RuntimeError(
            "No .pdb / .cif / AF3 .zip files found in inputs"
        )

    panel = load_af3_eval_panel(panel_json) if panel_json else None
    ckpt_path = resolve_gt_checkpoint(gt_checkpoint)
    if require_gt_checkpoint:
        ensure_public_checkpoint(ckpt_path)
    ctx = GtInferenceContext(checkpoint_path=str(ckpt_path) if ckpt_path.exists() else None)
    if ckpt_path.exists():
        try:
            ctx.ensure_loaded(str(ckpt_path), require_trained=require_gt_checkpoint)
        except RuntimeError as exc:
            if require_gt_checkpoint:
                raise
            print(f"WARNING: {exc}", file=sys.stderr)
    rows: list[dict] = []

    mode_label = "fast" if fast_mode and not require_oxrna else "deep"
    print(f"Screening {len(jobs)} structure(s) [{mode_label}] ...")
    for i, (label, path) in enumerate(jobs, start=1):
        print(f"\n[{i}/{len(jobs)}] {Path(label).name}")

        entry = match_panel_entry(label, panel) if panel else None
        job_rbp = (entry or {}).get("rbp_name") or rbp_name
        job_rna = (entry or {}).get("rna_sequence") or rna_sequence
        job_chrom = (entry or {}).get("chrom") if entry else chrom
        job_start = (entry or {}).get("genomic_start") if entry else genomic_start
        job_end = (entry or {}).get("genomic_end") if entry else genomic_end
        if job_start is not None and not isinstance(job_start, int):
            try:
                job_start = int(job_start)
            except (TypeError, ValueError):
                job_start = genomic_start
        if job_end is not None and not isinstance(job_end, int):
            try:
                job_end = int(job_end)
            except (TypeError, ValueError):
                job_end = genomic_end

        zip_path = _af3_zip_from_label(label)
        try:
            result = run_pipeline(
                pdb_path=str(path),
                cutoff=cutoff,
                verbose=not quiet,
                rbp_name=job_rbp,
                rna_sequence=job_rna,
                chrom=job_chrom,
                genomic_start=job_start,
                genomic_end=job_end,
                require_gt_checkpoint=require_gt_checkpoint,
                require_oxrna=require_oxrna,
                gt_checkpoint=str(ckpt_path) if ckpt_path.exists() else gt_checkpoint,
                model_rank=model_rank,
                fast_mode=fast_mode and not require_oxrna,
                inference_context=ctx,
                allow_physics_only=not require_gt_checkpoint,
            )
            row = {
                "file": label,
                "structure": str(path),
                "panel_id": (entry or {}).get("id"),
                "panel_label": (entry or {}).get("label"),
                "partner_group": (entry or {}).get("partner_group"),
                "rbp_name": job_rbp,
                "af3_iptm": af3_iptm(zip_path, model_rank) if zip_path else None,
                "af3_ptm": af3_ptm(zip_path, model_rank) if zip_path else None,
                "af3_ranking_score": af3_ranking_score(zip_path, model_rank) if zip_path else None,
                "combined_verdict": result["combined_verdict"],
                "confidence": result["confidence"],
                "rmsd_score": result["rmsd_score"],
                "rmsd_verdict": result["rmsd_verdict"],
                "geom_score": result["geom_score"],
                "geom_verdict": result["geom_verdict"],
                "contact_energy": result["contact_energy"],
                "contact_verdict": result["contact_verdict"],
                "clash_n_severe": result["clash_n_severe"],
                "clash_verdict": result["clash_verdict"],
                "gt_score": result["gt_score"],
                "gt_score_raw": result.get("gt_score_raw", result["gt_score"]),
                "gt_score_norm": result.get("gt_score_norm", result["gt_score"]),
                "gt_score_per_nt": result.get("gt_score_per_nt", result["gt_score"]),
                "gt_verdict": result["gt_verdict"],
                "gt_physics_only": result["gt_physics_only"],
                "bio_verdict": result["bio_verdict"],
                "eclip_supported": result.get("eclip_supported"),
                "n_interface_nt": len(result.get("interface_residues", [])),
                "n_prot_rna_edges": result.get("n_prot_rna_edges", 0),
                "screen_mode": mode_label,
            }
            row["composite_score"] = _composite_rank_score(row)
            row["unified_score"] = _unified_score(row)
            rows.append(row)
        except Exception as e:
            msg = str(e).splitlines()[0][:200]
            print(f"  FAILED: {msg}", file=sys.stderr)
            rows.append({
                "file": label,
                "structure": str(path),
                "panel_id": (entry or {}).get("id"),
                "combined_verdict": "ERROR",
                "confidence": 0.0,
                "error": msg,
            })

    df = pd.DataFrame(rows)
    if "composite_score" in df.columns:
        df = df.sort_values("composite_score", kind="mergesort")

    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    n_pass = (df["combined_verdict"] == "PASS").sum() if "combined_verdict" in df else 0
    n_warn = (df["combined_verdict"] == "WARN").sum() if "combined_verdict" in df else 0
    n_fail = (df["combined_verdict"] == "FAIL").sum() if "combined_verdict" in df else 0
    n_err = (df["combined_verdict"] == "ERROR").sum() if "combined_verdict" in df else 0
    print(f"\nSaved {len(df)} rows to {out}")
    print(
        f"Hallucination verdicts: PASS={n_pass}  WARN={n_warn}  "
        f"FAIL={n_fail}  ERROR={n_err}"
    )
    print(
        "  (AF3 mode: combined = PhysGT + biology + true clashes; "
        "use composite_score for partner ranking)"
    )

    if partner_summary:
        _print_partner_summary(df)

    if panel is not None and "panel_label" in df.columns:
        pos = df[df["panel_label"] == "positive"]
        neg = df[df["panel_label"] == "negative"]
        if len(pos) and len(neg) and "gt_score_norm" in df.columns:
            pos_mean = pos["gt_score_norm"].astype(float).mean()
            neg_mean = neg["gt_score_norm"].astype(float).mean()
            print(
                f"\nPanel GT (norm): positives mean={pos_mean:.2f}  "
                f"negatives mean={neg_mean:.2f}  "
                f"(lower = more plausible; want positives < negatives)"
            )

    return df


def main():
    ap = argparse.ArgumentParser(
        description="Screen AF3/RoseTTAFold protein-RNA structures for hallucinations"
    )
    ap.add_argument(
        "inputs",
        nargs="+",
        help="PDB/CIF files, AF3 Server .zip archives, or directories containing them",
    )
    ap.add_argument("--output", default="af3_screen_results.csv")
    ap.add_argument("--cutoff", type=float, default=5.0)
    ap.add_argument("--gt-checkpoint", default=None)
    ap.add_argument("--require-gt-checkpoint", action="store_true")
    ap.add_argument("--rbp-name", default=None, help="Global RBP name (overridden by --panel-json)")
    ap.add_argument("--rna-sequence", default=None, help="Global RNA sequence (overridden by --panel-json)")
    ap.add_argument(
        "--panel-json",
        default=None,
        help="AF3 eval panel JSON for per-job RBP/RNA metadata (P1–P5 / N1–N5)",
    )
    ap.add_argument(
        "--partner-summary",
        action="store_true",
        help="Print per-protein partner-group ranking after screening",
    )
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--fast",
        action="store_true",
        help="Quick triage only (skip oxRNA MD; default is full accuracy)",
    )
    ap.add_argument(
        "--require-oxrna",
        action="store_true",
        help="Fail if oxRNA unavailable (default: internal fallback)",
    )
    ap.add_argument("--chrom", default=None, help="Genomic chromosome for eCLIP overlay")
    ap.add_argument("--genomic-start", type=int, default=None)
    ap.add_argument("--genomic-end", type=int, default=None)
    ap.add_argument("--html", default=None, help="Write HTML report to this path")
    ap.add_argument(
        "--model-rank",
        type=int,
        default=0,
        help="AF3 model rank inside Server zips (0 = highest confidence)",
    )
    args = ap.parse_args()

    fast = args.fast
    screen_af3_structures(
        inputs=args.inputs,
        output_csv=args.output,
        cutoff=args.cutoff,
        gt_checkpoint=args.gt_checkpoint,
        require_gt_checkpoint=args.require_gt_checkpoint,
        rbp_name=args.rbp_name,
        rna_sequence=args.rna_sequence,
        panel_json=args.panel_json,
        quiet=args.quiet,
        model_rank=args.model_rank,
        partner_summary=args.partner_summary,
        fast_mode=fast,
        require_oxrna=args.require_oxrna and not fast,
        chrom=args.chrom,
        genomic_start=args.genomic_start,
        genomic_end=args.genomic_end,
    )
    if args.html:
        from .report_af3 import write_html_report
        write_html_report(args.output, args.html)
        print(f"Wrote HTML report to {args.html}")


if __name__ == "__main__":
    main()
