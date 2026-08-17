"""
Main pipeline entry point.

Accepts a PDB/CIF file or AF3 Server .zip of an AF3- or RoseTTAFold-predicted protein-RNA complex
and returns a combined multi-branch verdict on whether the predicted interface
is physically and biologically plausible.

Branches:
  1. Entropic (RMSD vs free-RNA ensemble)
  2. Local geometry (per-nucleotide torsions)
  3. Contact physics (electrostatic/H-bond complementarity)
  4. Steric clashes (geometric overlap)
  5. PhysGT (learned interface plausibility)
  6. Biological (eCLIP + binding motifs, optional)

Usage from CLI:
    python -m physrna_filter.pipeline my_complex.pdb

Usage from Python:
    from physrna_filter.pipeline import run_pipeline
    result = run_pipeline("my_complex.pdb")
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from .structure.parse_complex    import parse_complex, get_rna_sequence
from .structure.partner_selection import select_partner_pair
from .structure.extract_interface import (
    find_interface_residues,
    extract_interface_coords,
)
from .structure.local_geometry   import (
    extract_geometry,
    geometry_to_matrix,
)
from .simulation.run_simulation  import simulate_free_rna
from .analysis.cluster           import cluster_coordinates, cluster_geometry
from .analysis.score             import run_full_scoring, format_report
from .analysis.gt_inference      import GtInferenceContext


def run_pipeline(
    pdb_path: str,
    cutoff: float = 5.0,
    verbose: bool = True,
    rbp_name: str | None = None,
    rna_sequence: str | None = None,
    chrom: str | None = None,
    genomic_start: int | None = None,
    genomic_end: int | None = None,
    require_gt_checkpoint: bool = False,
    require_oxrna: bool = False,
    gt_checkpoint: str | None = None,
    model_rank: int = 0,
    fast_mode: bool = False,
    protein_chain_id: str | None = None,
    rna_chain_id: str | None = None,
    allow_physics_only: bool = True,
    inference_context: GtInferenceContext | None = None,
    reference_native_sequence: str | None = None,
) -> dict:
    """
    Full PhysRNA-Filter pipeline on a single AF3/RoseTTAFold complex.

    Args:
        pdb_path:       path to AF3 output .pdb, .cif, or Server .zip file
        model_rank:     AF3 model rank inside zip (0 = highest confidence)
        cutoff:         interface distance cutoff in Angstroms
        verbose:        print progress and report to stdout
        fast_mode:      skip free-RNA MD (PhysGT + clash + bio only)
        rbp_name:       optional RBP name for eCLIP/motif cross-check
        rna_sequence:   optional RNA sequence for motif scoring
        chrom:          optional genomic chromosome for eCLIP
        genomic_start:  optional genomic start for eCLIP
        genomic_end:    optional genomic end for eCLIP

    Returns:
        dict with all branch scores, combined_verdict, confidence, filter_result
    """
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"Input not found: {pdb_path}")

    _log(verbose, f"\nPhysRNA-Filter (AF3 Augmented)  |  input: {pdb_path}")
    if fast_mode:
        _log(verbose, "      fast mode: skipping free-RNA simulation")

    # ── Step 1: parse complex ─────────────────────────────────────────────────
    _log(verbose, "\n[1/8] Parsing complex ...")
    parsed = parse_complex(pdb_path, model_rank=model_rank)
    protein_chains, rna_chains = select_partner_pair(
        parsed.protein_chains,
        parsed.rna_chains,
        protein_chain_id=protein_chain_id,
        rna_chain_id=rna_chain_id,
    )
    _log(verbose, f"      {len(protein_chains)} protein chain(s), "
                  f"{len(rna_chains)} RNA chain(s) (partner-selected)")

    # ── Step 2: find interface ────────────────────────────────────────────────
    _log(verbose, f"\n[2/8] Finding interface residues (cutoff={cutoff} A) ...")
    interface_residues = find_interface_residues(
        protein_chains, rna_chains, cutoff=cutoff
    )
    _log(verbose, f"      {len(interface_residues)} interface nucleotides")

    af3_coords, residue_order = extract_interface_coords(
        rna_chains, interface_residues
    )

    # ── Step 3: extract AF3 local geometry ────────────────────────────────────
    _log(verbose, "\n[3/8] Extracting AF3 bound-pose local geometry ...")
    af3_geometry = extract_geometry(rna_chains, interface_residues)
    af3_geom_vec = geometry_to_matrix(af3_geometry, residue_order)
    _log(verbose, f"      geometry vector dim: {af3_geom_vec.shape[0]}")

    simulation_method = "skipped" if fast_mode else "pending"

    # ── Step 4: simulate free RNA ─────────────────────────────────────────────
    if fast_mode:
        _log(verbose, "\n[4/8] Skipping free-RNA simulation (fast mode) ...")
        n_atoms = max(len(af3_coords), 1)
        n_geom = max(af3_geom_vec.shape[0], 9)
        coord_snapshots = np.zeros((1, n_atoms, 3))
        geom_snapshots = np.zeros((1, n_geom))
    else:
        _log(verbose, "\n[4/8] Simulating free RNA conformational ensemble ...")
        if require_gt_checkpoint:
            from .validation.deploy_gt import ensure_gt_checkpoint
            if not ensure_gt_checkpoint(gt_checkpoint, auto_train=False):
                raise RuntimeError(
                    "Production GT checkpoint required but not found. "
                    "Run: python -m physrna_filter.validation.deploy_gt"
                )

        coord_snapshots, geom_snapshots, simulation_method = simulate_free_rna(
            rna_chains,
            interface_residues,
            require_oxrna=require_oxrna,
            partner_rna_chains=rna_chains,
        )
        _log(verbose, f"      {len(coord_snapshots)} conformational snapshots collected")

    # ── Step 5: cluster both branches ─────────────────────────────────────────
    _log(verbose, "\n[5/8] Clustering free-RNA ensemble ...")
    if fast_mode:
        coord_labels = np.zeros(len(coord_snapshots), dtype=int)
        coord_medoids = coord_snapshots
        geom_labels = np.zeros(len(geom_snapshots), dtype=int)
        geom_centroids = geom_snapshots
    else:
        coord_labels, coord_medoids = cluster_coordinates(coord_snapshots)
        geom_labels, geom_centroids = cluster_geometry(
            geom_snapshots,
            low_signal=simulation_method == "cg_langevin",
        )

    # ── Step 6: infer RNA sequence if not provided ────────────────────────────
    observed_rna_sequence: str | None = None
    if rna_chains:
        try:
            observed_rna_sequence = get_rna_sequence(rna_chains[0])
        except Exception:
            observed_rna_sequence = None

    if rna_sequence is None:
        rna_sequence = observed_rna_sequence

    # ── Step 7: multi-branch scoring ──────────────────────────────────────────
    _log(verbose, "\n[6/8] Scoring entropic + geometry branches ...")
    _log(verbose, "[7/8] Scoring contact + clash + PhysGT + biological ...")

    if inference_context is None and gt_checkpoint:
        inference_context = GtInferenceContext(checkpoint_path=gt_checkpoint)

    result = run_full_scoring(
        af3_bound_coords=af3_coords,
        af3_geom_vector=af3_geom_vec,
        af3_geometry_per_nuc=af3_geometry,
        coord_medoids=coord_medoids,
        geom_centroids=geom_centroids,
        residue_order=residue_order,
        protein_chains=protein_chains,
        rna_chains=rna_chains,
        pdb_path=pdb_path,
        gt_checkpoint=gt_checkpoint,
        rna_sequence=rna_sequence,
        rbp_name=rbp_name,
        chrom=chrom,
        genomic_start=genomic_start,
        genomic_end=genomic_end,
        fast_mode=fast_mode,
        model_rank=model_rank,
        parsed=parsed,
        inference_context=inference_context,
        require_trained_gt=require_gt_checkpoint,
        allow_physics_only=allow_physics_only,
        simulation_method=simulation_method,
        observed_rna_sequence=observed_rna_sequence,
        interface_cutoff=cutoff,
        reference_native_sequence=reference_native_sequence,
    )

    # ── Step 8: report ────────────────────────────────────────────────────────
    _log(verbose, "\n[8/8] Generating report ...")
    if verbose:
        print(format_report(result, residue_order, af3_mode=True))

    return {
        "rmsd_score":         result.rmsd_score,
        "rmsd_verdict":       result.rmsd_verdict,
        "geom_score":         result.geom_score,
        "geom_verdict":       result.geom_verdict,
        "contact_energy":     result.contact_energy,
        "contact_verdict":    result.contact_verdict,
        "clash_n_severe":     result.clash_n_severe,
        "clash_verdict":      result.clash_verdict,
        "gt_score":           result.gt_score_norm,
        "gt_score_raw":       result.gt_score_raw,
        "gt_score_norm":      result.gt_score_norm,
        "gt_score_per_nt":    result.gt_score_per_nt,
        "gt_verdict":         result.gt_verdict,
        "gt_physics_only":    result.gt_physics_only,
        "n_prot_rna_edges":   result.n_prot_rna_edges,
        "bio_verdict":        result.bio_verdict,
        "bio_motif_hits":     result.bio_motif_hits,
        "eclip_supported":    result.bio_eclip_supported,
        "combined_verdict":   result.combined_verdict,
        "confidence":         result.confidence,
        "per_nucleotide":     result.per_nucleotide,
        "filter_result":      result,
        "interface_residues": residue_order,
        "simulation_method":  simulation_method,
        "protein_chain_id":   protein_chains[0].id if protein_chains else None,
        "rna_chain_id":       rna_chains[0].id if rna_chains else None,
    }


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg)


def _cli() -> None:
    parser = argparse.ArgumentParser(
        prog="physrna-filter",
        description="Physics-informed validation of AI-generated protein-RNA complexes",
    )
    parser.add_argument(
        "pdb",
        help="Path to AF3 / RoseTTAFold output (.pdb, .cif, or AF3 Server .zip)",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=5.0,
        help="Interface distance cutoff in Angstroms (default: 5.0)",
    )
    parser.add_argument(
        "--rbp-name",
        default=None,
        help="RBP name for eCLIP/motif biological cross-check",
    )
    parser.add_argument(
        "--rna-sequence",
        default=None,
        help="RNA sequence for binding-motif scoring",
    )
    parser.add_argument("--chrom", default=None, help="Genomic chromosome for eCLIP")
    parser.add_argument("--genomic-start", type=int, default=None)
    parser.add_argument("--genomic-end", type=int, default=None)
    parser.add_argument(
        "--require-gt-checkpoint",
        action="store_true",
        help="Require trained PhysGT checkpoint (run deploy_gt first)",
    )
    parser.add_argument(
        "--require-oxrna",
        action="store_true",
        help="Require oxRNA/oxDNA for simulation (no CG fallback)",
    )
    parser.add_argument("--gt-checkpoint", default=None)
    parser.add_argument(
        "--model-rank",
        type=int,
        default=0,
        help="AF3 model rank inside a Server zip (0 = highest confidence, default: 0)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip free-RNA MD; score PhysGT + clash + biology only",
    )
    parser.add_argument("--protein-chain", default=None, help="Restrict to protein chain id")
    parser.add_argument("--rna-chain", default=None, help="Restrict to RNA chain id")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    try:
        result = run_pipeline(
            pdb_path=args.pdb,
            cutoff=args.cutoff,
            verbose=not args.quiet,
            rbp_name=args.rbp_name,
            rna_sequence=args.rna_sequence,
            chrom=args.chrom,
            genomic_start=args.genomic_start,
            genomic_end=args.genomic_end,
            require_gt_checkpoint=args.require_gt_checkpoint,
            require_oxrna=args.require_oxrna,
            gt_checkpoint=args.gt_checkpoint,
            model_rank=args.model_rank,
            fast_mode=args.fast,
            protein_chain_id=args.protein_chain,
            rna_chain_id=args.rna_chain,
            allow_physics_only=not args.require_gt_checkpoint,
        )
        verdict = result["combined_verdict"]
        sys.exit(0 if verdict == "PASS" else 1)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    _cli()
