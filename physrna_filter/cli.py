"""
Unified PhysRNA command-line interface.

Researchers typically need only four commands::

    physrna init
    physrna configure af3 --api-key YOUR_KEY
    physrna predict --protein ... --rna ... --rbp LIN28A
    physrna rank af3_predictions --rbp LIN28A

Full oxRNA MD and contrastive fine-tuning run by default when applicable.
Use ``--fast`` only for quick triage.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _cmd_doctor(_args: argparse.Namespace) -> None:
    from .config import doctor_report

    status = doctor_report()
    print("PhysRNA health check")
    for key, val in status.items():
        print(f"  {key}: {val}")
    if status.get("checkpoint_ok") == "no":
        print("\nRun: physrna init")
        print("    or: python -m physrna_filter.cli init")
        sys.exit(1)
    print("\nTip: if 'physrna' is not found, use: python -m physrna_filter.cli <command>")


def _cmd_init(_args: argparse.Namespace) -> None:
    from .config import init_user_environment

    status = init_user_environment(copy_checkpoint=True)
    print("PhysRNA ready.")
    print(f"  Data directory: {status['data_dir']}")
    print(f"  Checkpoint:     {status['checkpoint']}")
    if status.get("finetuned") == "yes":
        print("  Fine-tuned:     yes (your local weights will be used)")
    print(f"  oxDNA:          {status['oxdna']}")
    if "not found" in status["oxdna"]:
        print(
            "\n  For best accuracy on Linux/Mac, install oxDNA. "
            "On Windows, run scripts/bootstrap_wsl.ps1 and set OXDNA_BIN=wsl:..."
        )
    print("\nNext: predict a complex or rank AF3 zips:")
    print("  physrna configure af3 --api-key YOUR_KEY   # optional, for API mode")
    print("  physrna predict --protein SEQ --rna SEQ --rbp YOUR_PROTEIN")
    print("  physrna rank af3_predictions --rbp YOUR_PROTEIN")
    print("  (or: python -m physrna_filter.cli predict --protein SEQ --rna SEQ)")


def _cmd_configure_af3(args: argparse.Namespace) -> None:
    from .config import (
        af3_api_key_path,
        clear_af3_api_key,
        load_af3_api_key,
        mask_secret,
        save_af3_api_key,
    )

    if args.clear:
        if clear_af3_api_key():
            print(f"Removed API key from {af3_api_key_path()}")
        else:
            print("No saved API key to remove.")
        return

    key = args.api_key
    if args.api_key_file:
        key = Path(args.api_key_file).read_text(encoding="utf-8").strip()

    if args.show:
        existing = load_af3_api_key()
        if existing:
            print(f"API key: configured ({mask_secret(existing)})")
            print(f"  file: {af3_api_key_path()}")
        else:
            print("API key: not configured")
            print("  Set with: physrna configure af3 --api-key YOUR_KEY")
        return

    if not key:
        print("Provide --api-key, --api-key-file, --show, or --clear", file=sys.stderr)
        sys.exit(2)

    path = save_af3_api_key(key)
    print(f"Saved AlphaFold API key to {path}")
    print("Test with: physrna doctor")
    print("Then:      physrna predict --protein ... --rna ... --rbp NAME")


def _resolve_predict_sequences(
    *,
    protein: str | None,
    rna: str | None,
    protein_file: str | None,
    rna_file: str | None,
    af3_zip: str | None,
) -> tuple[str, str, str | None]:
    """Return (protein, rna, job_name_hint) for predict workflows."""
    from .data.af3_client import validate_protein_sequence, validate_rna_sequence
    from .structure.af3_confidence import sequences_from_af3_zip
    from .validation.predict_validate import _read_sequence_arg

    job_hint: str | None = None
    if af3_zip and (not protein and not protein_file or not rna and not rna_file):
        job_hint, zip_protein, zip_rna = sequences_from_af3_zip(af3_zip)
        if not protein and not protein_file and zip_protein:
            protein = zip_protein
        if not rna and not rna_file and zip_rna:
            rna = zip_rna

    protein_seq = ""
    rna_seq = ""
    if protein or protein_file:
        protein_seq = validate_protein_sequence(
            _read_sequence_arg(protein, protein_file, "protein")
        )
    if rna or rna_file:
        rna_seq = validate_rna_sequence(
            _read_sequence_arg(rna, rna_file, "rna")
        )
    return protein_seq, rna_seq, job_hint


def _cmd_predict(args: argparse.Namespace) -> None:
    from .validation.predict_validate import predict_and_validate

    api_key = args.api_key
    if args.api_key_file:
        api_key = Path(args.api_key_file).read_text(encoding="utf-8").strip()
    if api_key:
        os.environ["AF3_API_KEY"] = api_key

    protein, rna, job_hint = _resolve_predict_sequences(
        protein=args.protein,
        rna=args.rna,
        protein_file=args.protein_file,
        rna_file=args.rna_file,
        af3_zip=args.af3_zip,
    )
    job_name = args.job_name
    if job_hint and job_name == "physrna_fold":
        job_name = str(job_hint).replace(" ", "_")

    if not args.af3_zip and not args.af3_job_id and (not protein or not rna):
        print(
            "Provide --protein/--rna (or --protein-file/--rna-file), "
            "or --af3-zip with embedded sequences.",
            file=sys.stderr,
        )
        sys.exit(2)

    mode = "zip" if args.af3_zip else ("api" if args.af3_job_id else args.mode)
    result = predict_and_validate(
        protein_sequence=protein,
        rna_sequence=rna,
        mode=mode,
        job_name=job_name,
        af3_zip=args.af3_zip,
        server_json_path=args.server_json_out,
        api_key=api_key,
        af3_job_id=args.af3_job_id,
        rbp_name=args.rbp_name,
        require_gt_checkpoint=not args.allow_physics_only,
        require_oxrna=args.require_oxrna and not args.fast,
        fast_mode=args.fast,
        gt_checkpoint=args.gt_checkpoint,
        model_rank=args.model_rank,
        verbose=not args.quiet,
        output_json=args.output_json,
        timeout_s=args.timeout,
    )

    if result.get("pipeline"):
        verdict = result["pipeline"]["combined_verdict"]
        print(f"\nPhysRNA verdict: {verdict}")
        sys.exit(0 if verdict == "PASS" else 1)

    if result.get("af3_server_json") and not args.af3_zip:
        print("\nNext: upload the JSON at https://alphafoldserver.com, then:")
        print(f"  physrna predict --protein ... --rna ... --af3-zip fold_{args.job_name}.zip")
        sys.exit(2)

    sys.exit(1)


def _cmd_boltz_prepare(args: argparse.Namespace) -> None:
    from .data.boltz_benchmark import write_boltz_test_bundle

    manifest = write_boltz_test_bundle(
        args.output,
        count=args.count,
        msa_mode=args.msa_mode,
    )
    out = Path(args.output)
    n_yaml = len(list((out / "inputs").glob("*.yaml")))
    print(f"Wrote {n_yaml} Boltz YAML inputs to {out / 'inputs'}")
    print(f"Manifest: {manifest}")
    print("\nNext:")
    print(f"  pip install boltz")
    print(f"  boltz predict {out / 'inputs'} --use_msa_server")
    print(f"  # then copy top-model .cif files into {out / 'predictions'}/")
    print(f"  physrna rank {out / 'predictions'} --manifest {manifest.name} --rbp U1A")


def _cmd_rank(args: argparse.Namespace) -> None:
    from .validation.rank_af3_candidates import rank_af3_candidates

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


def _cmd_panel(args: argparse.Namespace) -> None:
    from .validation.eval_af3_panel import _panel_exit_code, eval_af3_panel

    fast = args.fast
    _, metrics = eval_af3_panel(
        zip_dir=args.zips,
        gt_checkpoint=args.gt_checkpoint,
        panel_json=args.panel_json,
        fast_mode=fast,
        require_oxrna=args.require_oxrna and not fast,
        model_rank=args.model_rank,
        finetune=args.finetune,
        no_finetune=args.no_finetune,
        finetune_epochs=args.finetune_epochs,
        merge_interface=args.merge_interface,
        output_csv=args.output_csv,
        output_json=args.output_json,
        output_html=args.html,
        require_gt_checkpoint=not args.allow_physics_only,
        prepare_missing=args.prepare_missing,
    )

    if args.prepare_missing:
        return

    _panel_exit_code(metrics)


def _cmd_report(args: argparse.Namespace) -> None:
    from .validation.report_af3 import write_html_report

    write_html_report(
        args.csv,
        args.output,
        title=args.title,
        metrics_json=args.metrics_json,
    )
    print(f"Wrote {args.output}")


def _cmd_fetch_dataset(args: argparse.Namespace) -> None:
    from .data.datasets import fetch_all_datasets, fetch_dataset

    names = ["pronab", "rnapedia", "pdb"] if args.fetch_name == "all" else [args.fetch_name]
    for name in names:
        result = fetch_dataset(
            name,
            update=args.update,
            structures=args.structures,
            max_structures=args.max_structures,
            affinity=args.affinity and name == "rnapedia",
            use_sample=args.sample,
        )
        summary = result["summary"]
        print(
            f"{summary.name}: {summary.entries} entries, "
            f"{summary.unique_pdbs} PDB codes ({summary.details})"
        )
        if args.structures and "structures" in result:
            print(f"  structures downloaded: {len(result['structures'])}")


def _cmd_benchmark_pronab(args: argparse.Namespace) -> None:
    from .validation.benchmark_pronab import run_pronab_benchmark

    run_pronab_benchmark(
        max_entries=args.max_targets,
        output_csv=args.output,
        run_full_pipeline=args.full_pipeline,
        minimize_mutations=not args.no_minimize,
        failures_csv=args.failures_csv,
        capped=args.capped,
    )


def _cmd_benchmark_rnapedia(args: argparse.Namespace) -> None:
    from .validation.benchmark_crystals import run_rnapedia_benchmark

    run_rnapedia_benchmark(
        max_targets=args.max_targets,
        output_csv=args.output,
        metrics_json=args.metrics_json,
        fast=args.fast,
        require_oxrna=args.require_oxrna,
        gt_checkpoint=args.gt_checkpoint,
        model_rank=args.model_rank,
        use_sample=args.sample,
    )


def _cmd_benchmark_pdb(args: argparse.Namespace) -> None:
    from .validation.benchmark_crystals import run_pdb_catalog_benchmark

    run_pdb_catalog_benchmark(
        max_targets=args.max_targets,
        output_csv=args.output,
        metrics_json=args.metrics_json,
        fast=args.fast,
        require_oxrna=args.require_oxrna,
        gt_checkpoint=args.gt_checkpoint,
        model_rank=args.model_rank,
        use_sample=args.sample,
    )


def _cmd_test_datasets(args: argparse.Namespace) -> None:
    """Fetch manifests and run fast smoke benchmarks on each dataset."""
    from .data.datasets import fetch_all_datasets, summarize_pdb_catalog, summarize_pronab, summarize_rnapedia
    from .validation.benchmark_crystals import run_pdb_catalog_benchmark, run_rnapedia_benchmark
    from .validation.benchmark_pronab import run_pronab_benchmark

    use_sample = args.sample
    print("Fetching dataset manifests ...")
    fetch_all_datasets(
        update=args.update,
        structures=args.download_structures,
        max_structures=args.max_structures,
        use_sample=use_sample,
    )
    print(f"  ProNAB:   {summarize_pronab()}")
    print(f"  RNApedia: {summarize_rnapedia(use_sample=use_sample)}")
    print(f"  PDB:      {summarize_pdb_catalog(use_sample=use_sample)}")

    max_n = args.max_targets
    print(f"\nRunning smoke benchmarks (max_targets={max_n}) ...")
    run_pronab_benchmark(
        max_entries=max_n,
        output_csv="pronab_smoke.csv",
        minimize_mutations=False,
    )
    run_rnapedia_benchmark(
        max_targets=max_n,
        output_csv="rnapedia_smoke.csv",
        metrics_json="rnapedia_smoke_metrics.json",
        fast=True,
        use_sample=use_sample,
    )
    run_pdb_catalog_benchmark(
        max_targets=max_n,
        output_csv="pdb_smoke.csv",
        metrics_json="pdb_smoke_metrics.json",
        fast=True,
        use_sample=use_sample,
    )
    print("\nDataset smoke tests complete.")


def _cmd_benchmark_foldbench(args: argparse.Namespace) -> None:
    from .data.foldbench import export_af3_job_manifest, fetch_foldbench_manifest
    from .validation.benchmark_foldbench import run_foldbench_benchmark

    if args.update_manifest:
        fetch_foldbench_manifest(update=True)
        print("Updated FoldBench manifest cache.")

    if args.prepare:
        export_af3_job_manifest(
            output_path=args.prepare_output,
            max_jobs=args.max_targets,
        )
        return

    run_foldbench_benchmark(
        predictions_dir=args.predictions,
        output_csv=args.output,
        output_json=args.metrics_json,
        gt_checkpoint=args.gt_checkpoint,
        max_targets=args.max_targets,
        fast_mode=args.fast,
        require_oxrna=args.require_oxrna,
        model_rank=args.model_rank,
        labels_csv=args.labels_csv,
        download_gt=not args.no_download_gt,
        irmsd_threshold=args.irmsd_threshold,
        lrmsd_threshold=args.lrmsd_threshold,
        list_only=args.list,
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="physrna",
        description="Physics-informed validation of AF3 protein–RNA complexes",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="One-time setup (checkpoint + oxDNA check)")
    sub.add_parser("doctor", help="Health check: checkpoint, oxDNA, RNA-FM")

    configure = sub.add_parser("configure", help="Store credentials and preferences")
    configure_sub = configure.add_subparsers(dest="configure_target", required=True)
    af3_cfg = configure_sub.add_parser("af3", help="AlphaFold Server API key")
    af3_cfg.add_argument("--api-key", default=None, help="API key to save locally")
    af3_cfg.add_argument("--api-key-file", default=None, help="Read API key from file")
    af3_cfg.add_argument("--show", action="store_true", help="Show whether a key is configured")
    af3_cfg.add_argument("--clear", action="store_true", help="Remove saved API key")

    predict = sub.add_parser(
        "predict",
        help="Protein + RNA → AlphaFold 3 → PhysRNA validation",
    )
    predict.add_argument("--protein", default=None, help="Protein one-letter sequence")
    predict.add_argument("--protein-file", default=None)
    predict.add_argument("--rna", default=None, help="RNA sequence (A/C/G/U)")
    predict.add_argument("--rna-file", default=None)
    predict.add_argument("--rbp", "--rbp-name", dest="rbp_name", default=None)
    predict.add_argument("--job-name", default="physrna_fold")
    predict.add_argument(
        "--mode",
        choices=["auto", "server-json", "local", "api", "zip"],
        default="auto",
        help="AF3 backend (auto: API if key set, else server-json)",
    )
    predict.add_argument("--api-key", default=None)
    predict.add_argument("--api-key-file", default=None)
    predict.add_argument("--af3-job-id", default=None, help="Resume an in-flight AF3 API job")
    predict.add_argument("--af3-zip", default=None, help="Existing AF3 zip (skip prediction)")
    predict.add_argument("--server-json-out", default=None)
    predict.add_argument("--gt-checkpoint", default=None)
    predict.add_argument("--fast", action="store_true")
    predict.add_argument("--require-oxrna", action="store_true")
    predict.add_argument("--allow-physics-only", action="store_true")
    predict.add_argument("--model-rank", type=int, default=0)
    predict.add_argument("--timeout", type=int, default=7200)
    predict.add_argument("--quiet", action="store_true")
    predict.add_argument("--output-json", default=None)

    boltz = sub.add_parser("boltz", help="Batch Boltz structure prediction helpers")
    boltz_sub = boltz.add_subparsers(dest="boltz_command", required=True)
    boltz_prepare = boltz_sub.add_parser(
        "prepare",
        help="Generate 100 test protein–RNA complexes for Boltz + PhysRNA rank",
    )
    boltz_prepare.add_argument(
        "-o", "--output",
        default="boltz_test_100",
        help="Output directory (default: boltz_test_100)",
    )
    boltz_prepare.add_argument("--count", type=int, default=100)
    boltz_prepare.add_argument(
        "--msa-mode",
        default="empty",
        help="Boltz protein msa field (default: empty; use with --use_msa_server)",
    )

    rank = sub.add_parser(
        "rank",
        help="Rank AF3 candidates — full oxRNA MD by default",
    )
    rank.add_argument(
        "zips",
        nargs="?",
        default="af3_predictions",
        help="Folder with AF3 Server zips (default: af3_predictions)",
    )
    rank.add_argument(
        "--rbp",
        "--rbp-name",
        dest="rbp_name",
        default=None,
        help="Only rank jobs for this protein (filters by AF3 job name)",
    )
    rank.add_argument("--all", action="store_true", help="Rank every zip in folder")
    rank.add_argument("--manifest", default=None, help="Optional candidates.csv")
    rank.add_argument("--include-controls", action="store_true")
    rank.add_argument("--gt-checkpoint", default=None)
    rank.add_argument("-o", "--output", default="ranked_candidates.csv")
    rank.add_argument("--metrics-json", default="ranked_candidates_metrics.json")
    rank.add_argument("--html", default="ranked_candidates_report.html")
    rank.add_argument(
        "--fast",
        action="store_true",
        help="Quick triage only (skip oxRNA MD; not recommended for decisions)",
    )
    rank.add_argument(
        "--deep-top",
        type=int,
        default=3,
        help="With --fast: re-run oxRNA on top N finalists (default 3)",
    )
    rank.add_argument(
        "--require-oxrna",
        action="store_true",
        help="Fail if oxRNA is unavailable (default: fall back with warning)",
    )
    rank.add_argument(
        "--finetune",
        action="store_true",
        help="Force contrastive fine-tune on your candidates",
    )
    rank.add_argument(
        "--no-finetune",
        action="store_true",
        help="Skip auto fine-tune even with multiple RNAs for one protein",
    )
    rank.add_argument("--finetune-epochs", type=int, default=50)
    rank.add_argument("--max-negatives", type=int, default=3)
    rank.add_argument("--model-rank", type=int, default=0)
    rank.add_argument("--allow-physics-only", action="store_true")
    rank.add_argument("--write-manifest-template", metavar="PATH", default=None)

    panel = sub.add_parser("panel", help="Evaluate P/N benchmark panel (20 jobs)")
    panel.add_argument("zips", nargs="?", default="af3_predictions")
    panel.add_argument(
        "--panel-json",
        default=None,
        help="Panel JSON (default: extended 20-job panel)",
    )
    panel.add_argument("--gt-checkpoint", default=None)
    panel.add_argument("--fast", action="store_true")
    panel.add_argument("--require-oxrna", action="store_true")
    panel.add_argument("--finetune", action="store_true")
    panel.add_argument("--no-finetune", action="store_true")
    panel.add_argument("--finetune-epochs", type=int, default=50)
    panel.add_argument("--merge-interface", action="store_true")
    panel.add_argument("--output-csv", default="eval_panel.csv")
    panel.add_argument("--output-json", default="eval_panel_metrics.json")
    panel.add_argument("--html", default="eval_panel_report.html")
    panel.add_argument(
        "--prepare-missing",
        default=None,
        metavar="PATH",
        help="Export AF3 Server JSON for panel jobs missing zips",
    )
    panel.add_argument("--model-rank", type=int, default=0)
    panel.add_argument("--allow-physics-only", action="store_true")

    report = sub.add_parser("report", help="Regenerate HTML from a ranked CSV")
    report.add_argument("csv")
    report.add_argument("-o", "--output", default="report.html")
    report.add_argument("--title", default="PhysRNA report")
    report.add_argument("--metrics-json", default=None)

    benchmark = sub.add_parser(
        "benchmark",
        help="Evaluate PhysRNA on external AF3 benchmarks",
    )
    bench_sub = benchmark.add_subparsers(dest="benchmark_name", required=True)

    foldbench = bench_sub.add_parser(
        "foldbench",
        help="FoldBench protein–RNA benchmark (70 targets)",
    )
    foldbench.add_argument(
        "--predictions",
        default=None,
        help="Folder with AF3 Server zips or mmCIF files",
    )
    foldbench.add_argument("-o", "--output", default="foldbench_benchmark.csv")
    foldbench.add_argument("--metrics-json", default="foldbench_metrics.json")
    foldbench.add_argument("--gt-checkpoint", default=None)
    foldbench.add_argument("--max-targets", type=int, default=None,
                           help="Limit targets (prepare or benchmark run)")
    foldbench.add_argument("--fast", action="store_true")
    foldbench.add_argument("--require-oxrna", action="store_true")
    foldbench.add_argument("--model-rank", type=int, default=0)
    foldbench.add_argument("--labels-csv", default=None)
    foldbench.add_argument("--no-download-gt", action="store_true")
    foldbench.add_argument("--irmsd-threshold", type=float, default=4.0)
    foldbench.add_argument("--lrmsd-threshold", type=float, default=10.0)
    foldbench.add_argument(
        "--prepare",
        action="store_true",
        help="Download crystals and write foldbench_af3_jobs.json for AF3 Server",
    )
    foldbench.add_argument("--prepare-output", default="foldbench_af3_jobs.json")
    foldbench.add_argument("--list", action="store_true", help="Show target coverage")
    foldbench.add_argument("--update-manifest", action="store_true")

    fetch = sub.add_parser("fetch", help="Download ProNAB, RNApedia, or PDB catalogs")
    fetch_sub = fetch.add_subparsers(dest="fetch_name", required=True)
    for name, help_text in (
        ("pronab", "ProNAB / PRA-MutPred mutation ΔΔG set (~710 entries)"),
        ("rnapedia", "RNApedia protein–RNA interfaces (~56k pairs)"),
        ("pdb", "RCSB protein–RNA crystal catalog (~7.8k entries)"),
        ("all", "Fetch all three datasets"),
    ):
        parser = fetch_sub.add_parser(name, help=help_text)
        parser.add_argument("--update", action="store_true", help="Refresh from upstream")
        parser.add_argument(
            "--structures",
            action="store_true",
            help="Download PDB files referenced by the dataset",
        )
        parser.add_argument(
            "--max-structures",
            type=int,
            default=None,
            help="Cap structure downloads (RNApedia/PDB)",
        )
        parser.add_argument(
            "--affinity",
            action="store_true",
            help="Also download RNApedia affinity subset (RNApedia only, ~251 MB)",
        )
        parser.add_argument(
            "--sample",
            action="store_true",
            help="Use bundled sample manifests (offline / CI)",
        )

    test = sub.add_parser("test", help="Run integration smoke tests")
    test_sub = test.add_subparsers(dest="test_name", required=True)
    datasets_test = test_sub.add_parser(
        "datasets",
        help="Fetch + fast benchmark on ProNAB, RNApedia, and PDB",
    )
    datasets_test.add_argument("--update", action="store_true")
    datasets_test.add_argument("--max-targets", type=int, default=3)
    datasets_test.add_argument("--download-structures", action="store_true")
    datasets_test.add_argument("--max-structures", type=int, default=20)
    datasets_test.add_argument(
        "--sample",
        action="store_true",
        help="Use bundled sample manifests for RNApedia/PDB",
    )

    pronab = bench_sub.add_parser(
        "pronab",
        help="ProNAB contact-score ΔΔG benchmark",
    )
    pronab.add_argument("-o", "--output", default="pronab_benchmark.csv")
    pronab.add_argument("--max-targets", type=int, default=None)
    pronab.add_argument("--full-pipeline", action="store_true")
    pronab.add_argument("--no-minimize", action="store_true")
    pronab.add_argument("--failures-csv", default=None)
    pronab.add_argument("--capped", action="store_true")

    rnapedia = bench_sub.add_parser(
        "rnapedia",
        help="Screen RNApedia crystal interfaces (structural positives)",
    )
    rnapedia.add_argument("-o", "--output", default="rnapedia_benchmark.csv")
    rnapedia.add_argument("--metrics-json", default="rnapedia_metrics.json")
    rnapedia.add_argument("--max-targets", type=int, default=None)
    rnapedia.add_argument("--fast", action="store_true")
    rnapedia.add_argument("--require-oxrna", action="store_true")
    rnapedia.add_argument("--gt-checkpoint", default=None)
    rnapedia.add_argument("--model-rank", type=int, default=0)
    rnapedia.add_argument("--sample", action="store_true")

    pdb_bench = bench_sub.add_parser(
        "pdb",
        help="Screen RCSB protein–RNA crystals (structural positives)",
    )
    pdb_bench.add_argument("-o", "--output", default="pdb_catalog_benchmark.csv")
    pdb_bench.add_argument("--metrics-json", default="pdb_catalog_metrics.json")
    pdb_bench.add_argument("--max-targets", type=int, default=None)
    pdb_bench.add_argument("--fast", action="store_true")
    pdb_bench.add_argument("--require-oxrna", action="store_true")
    pdb_bench.add_argument("--gt-checkpoint", default=None)
    pdb_bench.add_argument("--model-rank", type=int, default=0)
    pdb_bench.add_argument("--sample", action="store_true")

    return ap


def main(argv: list[str] | None = None) -> None:
    ap = build_parser()
    args = ap.parse_args(argv)

    if args.command == "init":
        _cmd_init(args)
    elif args.command == "doctor":
        _cmd_doctor(args)
    elif args.command == "configure":
        if args.configure_target == "af3":
            _cmd_configure_af3(args)
        else:
            ap.print_help()
            sys.exit(2)
    elif args.command == "predict":
        _cmd_predict(args)
    elif args.command == "boltz":
        if args.boltz_command == "prepare":
            _cmd_boltz_prepare(args)
        else:
            ap.print_help()
            sys.exit(2)
    elif args.command == "rank":
        if getattr(args, "write_manifest_template", None):
            from .data.candidate_manifest import write_manifest_template

            path = write_manifest_template(args.write_manifest_template)
            print(f"Wrote manifest template to {path}")
            return
        _cmd_rank(args)
    elif args.command == "panel":
        _cmd_panel(args)
    elif args.command == "report":
        _cmd_report(args)
    elif args.command == "fetch":
        _cmd_fetch_dataset(args)
    elif args.command == "test":
        if args.test_name == "datasets":
            _cmd_test_datasets(args)
        else:
            ap.print_help()
            sys.exit(2)
    elif args.command == "benchmark":
        if args.benchmark_name == "foldbench":
            _cmd_benchmark_foldbench(args)
        elif args.benchmark_name == "pronab":
            _cmd_benchmark_pronab(args)
        elif args.benchmark_name == "rnapedia":
            _cmd_benchmark_rnapedia(args)
        elif args.benchmark_name == "pdb":
            _cmd_benchmark_pdb(args)
        else:
            ap.print_help()
            sys.exit(2)
    else:
        ap.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
