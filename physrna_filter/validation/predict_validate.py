"""
End-to-end: protein + RNA sequences → AlphaFold 3 structure → PhysRNA validation.

This is the "Use Case 3" product entry point: one command (or two-step with
Server upload) from sequences to PASS/WARN/FAIL.

Examples
--------
Generate AF3 Server JSON (most Windows users):

    python -m physrna_filter.validation.predict_validate \\
        --protein MKTIIALSYIFCLVFA ... \\
        --rna AUGCAUGCAUGC \\
        --mode server-json

After downloading the AF3 zip from alphafoldserver.com:

    python -m physrna_filter.validation.predict_validate \\
        --protein MKTIIALSYIFCLVFA ... \\
        --rna AUGCAUGCAUGC \\
        --af3-zip fold_myjob.zip \\
        --require-gt-checkpoint

Local AF3 (Linux/WSL + GPU + licensed weights):

    export AF3_MODEL_DIR=~/af3_models AF3_DB_DIR=~/public_databases
    python -m physrna_filter.validation.predict_validate \\
        --protein ... --rna ... --mode local --require-gt-checkpoint
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..data.af3_client import predict_structure, validate_protein_sequence, validate_rna_sequence
from ..pipeline import run_pipeline


def _read_sequence_arg(value: str | None, file_path: str | None, label: str) -> str:
    if file_path:
        text = Path(file_path).read_text(encoding="utf-8")
        return text
    if value:
        return value
    raise ValueError(f"{label} sequence required (--{label} or --{label}-file)")


def predict_and_validate(
    protein_sequence: str,
    rna_sequence: str,
    *,
    mode: str = "auto",
    job_name: str = "physrna_fold",
    af3_zip: str | None = None,
    server_json_path: str | None = None,
    api_key: str | None = None,
    af3_job_id: str | None = None,
    rbp_name: str | None = None,
    require_gt_checkpoint: bool = False,
    require_oxrna: bool = False,
    fast_mode: bool = False,
    gt_checkpoint: str | None = None,
    model_rank: int = 0,
    verbose: bool = True,
    output_json: str | None = None,
    timeout_s: int = 7200,
) -> dict:
    """
    Predict (or accept) an AF3 complex, then run the full PhysRNA pipeline.

    Returns a merged dict with prediction metadata + pipeline scores.
    If only JSON was written (server-json mode), pipeline fields are absent.
    """
    protein_sequence = validate_protein_sequence(protein_sequence) if protein_sequence else ""
    rna_sequence = validate_rna_sequence(rna_sequence) if rna_sequence else ""

    pred = predict_structure(
        protein_sequence,
        rna_sequence,
        mode=mode,
        job_name=job_name,
        server_json_path=server_json_path,
        af3_zip=af3_zip,
        api_key=api_key,
        af3_job_id=af3_job_id,
        timeout_s=timeout_s,
    )

    result: dict = {
        "job_name": job_name,
        "protein_length": len(protein_sequence),
        "rna_length": len(rna_sequence),
        "af3_backend": pred["backend"],
        "af3_structure_path": pred.get("structure_path"),
        "af3_server_json": pred.get("server_json"),
        "af3_notes": pred.get("notes", []),
        "af3_confidence": pred.get("af3_confidence"),
        "pipeline": None,
    }

    if pred.get("structure_path") is None:
        if verbose:
            print("\n--- AlphaFold 3 ---")
            for note in result["af3_notes"]:
                print(f"  {note}")
            if pred.get("server_json"):
                print(f"  JSON written: {pred['server_json']}")
        if output_json:
            Path(output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    structure = pred["structure_path"]
    if verbose:
        print(f"\n--- PhysRNA validation on {structure} ---")

    pipeline_result = run_pipeline(
        structure,
        verbose=verbose,
        rbp_name=rbp_name,
        rna_sequence=rna_sequence,
        require_gt_checkpoint=require_gt_checkpoint,
        require_oxrna=require_oxrna,
        gt_checkpoint=gt_checkpoint,
        model_rank=model_rank,
        fast_mode=fast_mode,
    )
    result["pipeline"] = pipeline_result

    if output_json:
        # FilterResult is not JSON-serializable; keep scalar fields
        serializable = {**result}
        if pipeline_result:
            serializable["pipeline"] = {
                k: v for k, v in pipeline_result.items()
                if k != "filter_result" and k != "per_nucleotide"
            }
        Path(output_json).write_text(
            json.dumps(serializable, indent=2, default=str), encoding="utf-8"
        )

    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Protein + RNA sequences → AF3 → PhysRNA validation"
    )
    ap.add_argument("--protein", default=None, help="Protein one-letter sequence")
    ap.add_argument("--protein-file", default=None, help="File containing protein sequence")
    ap.add_argument("--rna", default=None, help="RNA sequence (A/C/G/U)")
    ap.add_argument("--rna-file", default=None, help="File containing RNA sequence")
    ap.add_argument("--job-name", default="physrna_fold")
    ap.add_argument(
        "--mode",
        choices=["auto", "server-json", "local", "api", "zip"],
        default="auto",
        help="AF3 backend (default: auto)",
    )
    ap.add_argument("--api-key", default=None, help="AlphaFold API key (or use configure af3)")
    ap.add_argument("--af3-job-id", default=None, help="Resume polling an existing AF3 API job")
    ap.add_argument("--af3-zip", default=None,
                    help="Existing AF3 Server zip or .cif (skip prediction)")
    ap.add_argument("--server-json-out", default=None,
                    help="Where to write Server JSON (server-json mode)")
    ap.add_argument("--rbp-name", default=None, help="RBP name for biological branch")
    ap.add_argument("--require-gt-checkpoint", action="store_true")
    ap.add_argument("--require-oxrna", action="store_true")
    ap.add_argument("--fast", action="store_true", help="Skip oxRNA MD during validation")
    ap.add_argument("--gt-checkpoint", default=None)
    ap.add_argument("--model-rank", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=7200, help="AF3 API/local timeout (seconds)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--output-json", default=None, help="Save summary JSON")
    args = ap.parse_args()

    if not args.af3_zip and not args.af3_job_id:
        if (not args.protein and not args.protein_file) or (not args.rna and not args.rna_file):
            ap.error(
                "--protein/--rna (or --protein-file/--rna-file) required unless "
                "--af3-zip or --af3-job-id is provided"
            )

    protein = ""
    rna = ""
    if not args.af3_zip and not args.af3_job_id:
        protein = validate_protein_sequence(
            _read_sequence_arg(args.protein, args.protein_file, "protein")
        )
        rna = validate_rna_sequence(
            _read_sequence_arg(args.rna, args.rna_file, "rna")
        )
    mode = "zip" if args.af3_zip else ("api" if args.af3_job_id else args.mode)
    result = predict_and_validate(
        protein_sequence=protein,
        rna_sequence=rna,
        mode=mode,
        job_name=args.job_name,
        af3_zip=args.af3_zip,
        server_json_path=args.server_json_out,
        api_key=args.api_key,
        af3_job_id=args.af3_job_id,
        rbp_name=args.rbp_name,
        require_gt_checkpoint=args.require_gt_checkpoint,
        require_oxrna=args.require_oxrna,
        fast_mode=args.fast,
        gt_checkpoint=args.gt_checkpoint,
        model_rank=args.model_rank,
        verbose=not args.quiet,
        output_json=args.output_json,
        timeout_s=args.timeout,
    )

    if result.get("pipeline"):
        verdict = result["pipeline"]["combined_verdict"]
        sys.exit(0 if verdict == "PASS" else 1)

    # JSON-only step — not an error, but non-zero so scripts know validation pending
    if result.get("af3_server_json") and not args.af3_zip:
        sys.exit(2)


if __name__ == "__main__":
    main()
