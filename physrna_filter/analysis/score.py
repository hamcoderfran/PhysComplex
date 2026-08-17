"""
Combines all hallucination-detection branches into a single filter verdict.

Branches (literature-motivated, complementary):
  - RMSD:      entropic plausibility (Sulc et al. 2014 oxRNA; Abramson et al. 2024 AF3)
  - Geometry:  per-nucleotide torsion/pucker (Williamson 2000 induced fit)
  - Contact:   electrostatic/stacking/H-bond complementarity (Draper 2004)
  - Clash:     steric overlap (Richardson clash criteria)
  - PhysGT:    learned interface plausibility (heterogeneous GNN / decoy scoring)
  - Biological: eCLIP + motif cross-check (Van Nostrand et al. 2020)

A complex can fail on any branch independently.  For AF3 screening the
combined verdict prioritizes PhysGT, biological partner match, and true steric
overlaps — not bound-vs-free RMSD or interface contact distance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .rmsd import kabsch_rmsd
from .cluster import nearest_geometry_cluster_distance
from .geometry_score import (
    score_per_nucleotide,
    aggregate_geometry_score,
    PerNucleotideScore,
)
from ..structure.local_geometry import NucleotideGeometry
from .thresholds import load_thresholds


def _get_thresholds() -> dict:
    return load_thresholds()


@dataclass
class FilterResult:
    """Complete filter output for a single protein-RNA complex."""

    # RMSD branch
    rmsd_score: float = float("inf")
    rmsd_nearest_cluster: int = -1
    rmsd_verdict: str = "UNKNOWN"

    # Geometry branch
    geom_score: float = float("inf")
    geom_max_score: float = float("inf")
    geom_nearest_cluster: int = -1
    geom_verdict: str = "UNKNOWN"

    # Contact physics branch
    contact_energy: float = 0.0
    contact_n_residues: int = 0
    contact_n_contacts: int = 0
    contact_verdict: str = "UNKNOWN"

    # Clash branch
    clash_n_severe: int = 0
    clash_n_moderate: int = 0
    clash_worst_distance: float = float("inf")
    clash_verdict: str = "UNKNOWN"

    # PhysGT branch
    gt_score: float = 0.0
    gt_score_raw: float = 0.0
    gt_score_norm: float = 0.0
    gt_score_per_nt: float = 0.0
    gt_verdict: str = "UNKNOWN"
    gt_physics_only: bool = True
    n_prot_rna_edges: int = 0

    # Biological branch
    bio_verdict: str = "UNKNOWN"
    bio_motif_hits: list[str] = field(default_factory=list)
    bio_eclip_supported: bool | None = None

    # per-nucleotide detail
    per_nucleotide: list[PerNucleotideScore] = field(default_factory=list)

    # combined
    combined_verdict: str = "UNKNOWN"
    confidence: float = 0.0     # 0–1 confidence in the verdict


def entropic_plausibility_score(
    af3_bound_coords: np.ndarray,
    coord_medoids: np.ndarray,
) -> tuple[float, int, str]:
    """
    RMSD branch: minimum Kabsch-RMSD from AF3 bound coords to any medoid.
    """
    min_rmsd = np.inf
    nearest  = -1

    for i, medoid in enumerate(coord_medoids):
        if medoid.shape != af3_bound_coords.shape:
            continue
        _, rmsd = kabsch_rmsd(mobile=medoid, reference=af3_bound_coords)
        if rmsd < min_rmsd:
            min_rmsd = rmsd
            nearest  = i

    verdict = _rmsd_verdict(float(min_rmsd))
    return float(min_rmsd), nearest, verdict


def geometry_plausibility_score(
    af3_geom_vector: np.ndarray,
    geom_centroids: np.ndarray,
) -> tuple[float, int, str]:
    """
    Geometry branch: angular distance from AF3 geometry vector to nearest
    torsion-space cluster centroid.
    """
    dist, nearest = nearest_geometry_cluster_distance(af3_geom_vector, geom_centroids)
    verdict = _geom_verdict(dist)
    return dist, nearest, verdict


def contact_plausibility_score(
    protein_chains: list,
    rna_chains: list,
    interface_protein_residues: list[tuple[str, int]] | None = None,
    *,
    af3_mode: bool = False,
) -> tuple[float, int, int, str]:
    """
    Contact physics branch: aggregate interface contact energy.

    Returns (total_energy, n_residues, n_contacts, verdict).
  """
    from .contact_score import aggregate_interface_contact_energy

    result = aggregate_interface_contact_energy(
        protein_chains, rna_chains, interface_protein_residues, af3_mode=af3_mode
    )
    return (
        result["total_energy"],
        result["n_interface_residues"],
        result["n_contacts"],
        result["verdict"],
    )


def clash_plausibility_score(
    protein_chains: list,
    rna_chains: list,
    interface_rna_residues: list[tuple[str, int]] | None = None,
    *,
    af3_mode: bool = False,
) -> tuple[int, int, float, str]:
    """
    Clash branch: count steric overlaps at the interface.

    Returns (n_severe, n_moderate, worst_distance, verdict).
    """
    from ..structure.clash_detection import detect_interface_clashes

    result = detect_interface_clashes(
        protein_chains, rna_chains, interface_rna_residues, af3_mode=af3_mode
    )
    return (
        result.n_severe,
        result.n_moderate,
        result.worst_distance,
        result.verdict,
    )


def gt_plausibility_score(
    pdb_path: str,
    checkpoint_path: str | None = None,
    *,
    model_rank: int = 0,
    parsed: object | None = None,
    protein_chains: list | None = None,
    rna_chains: list | None = None,
    inference_context: object | None = None,
    require_trained: bool = False,
    allow_physics_only: bool = True,
) -> dict:
    """
    PhysGT branch: learned + physics interface plausibility.

    Returns dict with raw and size-normalized scores.
    """
    from .gt_inference import score_af3_interface

    return score_af3_interface(
        pdb_path,
        checkpoint_path=checkpoint_path,
        model_rank=model_rank,
        parsed=parsed,
        protein_chains=protein_chains,
        rna_chains=rna_chains,
        inference_context=inference_context,
        require_trained=require_trained,
        allow_physics_only=allow_physics_only,
    )


def biological_plausibility_score(
    rna_sequence: str | None = None,
    rbp_name: str | None = None,
    chrom: str | None = None,
    start: int | None = None,
    end: int | None = None,
    observed_rna_sequence: str | None = None,
    reference_native_sequence: str | None = None,
) -> tuple[str, list[str], bool | None]:
    """
    Biological branch: eCLIP + motif cross-check.

    Returns (verdict, motif_hits, eclip_supported).
    """
    from .biological_plausibility import assess_biological_plausibility

    result = assess_biological_plausibility(
        rna_sequence=rna_sequence,
        rbp_name=rbp_name,
        chrom=chrom,
        start=start,
        end=end,
        observed_rna_sequence=observed_rna_sequence,
        reference_native_sequence=reference_native_sequence,
    )
    return result.verdict, result.motif_hits, result.eclip_supported


def run_full_scoring(
    af3_bound_coords: np.ndarray,
    af3_geom_vector: np.ndarray,
    af3_geometry_per_nuc: dict[tuple[str, int], NucleotideGeometry],
    coord_medoids: np.ndarray,
    geom_centroids: np.ndarray,
    residue_order: list[tuple[str, int]],
    *,
    protein_chains: list | None = None,
    rna_chains: list | None = None,
    pdb_path: str | None = None,
    gt_checkpoint: str | None = None,
    rna_sequence: str | None = None,
    rbp_name: str | None = None,
    chrom: str | None = None,
    genomic_start: int | None = None,
    genomic_end: int | None = None,
    af3_mode: bool = True,
    fast_mode: bool = False,
    model_rank: int = 0,
    parsed: object | None = None,
    inference_context: object | None = None,
    require_trained_gt: bool = False,
    allow_physics_only: bool = True,
    simulation_method: str | None = None,
    observed_rna_sequence: str | None = None,
    interface_cutoff: float = 5.0,
    reference_native_sequence: str | None = None,
) -> FilterResult:
    """
    Runs all scoring branches and returns a combined FilterResult.
    """
    # RMSD branch
    if fast_mode:
        rmsd, rmsd_cluster, rmsd_verdict = float("nan"), -1, "UNKNOWN"
    else:
        rmsd, rmsd_cluster, rmsd_verdict = entropic_plausibility_score(
            af3_bound_coords, coord_medoids
        )
        if af3_mode:
            rmsd_verdict = _rmsd_verdict_af3(rmsd)

    # Geometry branch
    if fast_mode:
        geom_dist, geom_cluster, geom_verdict = float("nan"), -1, "UNKNOWN"
        per_nuc = score_per_nucleotide(
            af3_geometry_per_nuc, geom_centroids, residue_order, af3_mode=af3_mode
        )
        agg = aggregate_geometry_score(per_nuc, af3_mode=af3_mode)
        geom_verdict = agg["verdict"]
        geom_max = agg.get("max_distance", 0.0) or 0.0
        geom_mean = agg.get("mean_distance", 0.0) or 0.0
    else:
        geom_dist, geom_cluster, geom_verdict = geometry_plausibility_score(
            af3_geom_vector, geom_centroids
        )

        per_nuc = score_per_nucleotide(
            af3_geometry_per_nuc, geom_centroids, residue_order, af3_mode=af3_mode
        )
        agg = aggregate_geometry_score(per_nuc, af3_mode=af3_mode)
        if af3_mode:
            geom_verdict = agg["verdict"]
            geom_max     = agg.get("max_distance", 0.0) or 0.0
            geom_mean    = agg.get("mean_distance", 0.0) or 0.0
        else:
            geom_verdict = _worse_verdict(geom_verdict, agg["verdict"])
            geom_max     = agg.get("max_distance", geom_dist) or geom_dist
            geom_mean    = agg.get("mean_distance", geom_dist) or geom_dist

        if (
            af3_mode
            and simulation_method == "cg_langevin"
            and geom_cluster >= 0
        ):
            # Free-ensemble geometry is low-signal under CG fallback; keep intrinsic only.
            pass

    # Contact physics branch
    contact_energy = 0.0
    contact_n_residues = 0
    contact_n_contacts = 0
    contact_verdict = "UNKNOWN"
    if protein_chains is not None and rna_chains is not None:
        from ..structure.extract_interface import find_interface_protein_residues

        iface_protein = find_interface_protein_residues(
            protein_chains, rna_chains, cutoff=interface_cutoff
        )
        contact_energy, contact_n_residues, contact_n_contacts, contact_verdict = (
            contact_plausibility_score(
                protein_chains, rna_chains, iface_protein, af3_mode=af3_mode
            )
        )

    # Clash branch
    clash_n_severe = clash_n_moderate = 0
    clash_worst = float("inf")
    clash_verdict = "UNKNOWN"
    if protein_chains is not None and rna_chains is not None:
        clash_n_severe, clash_n_moderate, clash_worst, clash_verdict = (
            clash_plausibility_score(
                protein_chains, rna_chains, residue_order, af3_mode=af3_mode
            )
        )

    # PhysGT branch
    gt_score = gt_score_raw = gt_score_norm = gt_score_per_nt = 0.0
    gt_verdict = "UNKNOWN"
    gt_physics_only = True
    n_prot_rna_edges = 0
    if pdb_path is not None:
        gt_info = gt_plausibility_score(
            pdb_path,
            checkpoint_path=gt_checkpoint,
            model_rank=model_rank,
            parsed=parsed,
            protein_chains=protein_chains,
            rna_chains=rna_chains,
            inference_context=inference_context,
            require_trained=require_trained_gt,
            allow_physics_only=allow_physics_only,
        )
        gt_score_raw = gt_info["gt_score"]
        gt_score_norm = gt_info["gt_score_norm"]
        gt_score_per_nt = gt_info["gt_score_per_nt"]
        gt_score = gt_score_norm
        gt_verdict = gt_info["gt_verdict"]
        gt_physics_only = gt_info["physics_only"]
        n_prot_rna_edges = gt_info["n_prot_rna_edges"]

    # Biological branch
    bio_verdict, bio_motif_hits, bio_eclip = "UNKNOWN", [], None
    if rna_sequence or rbp_name:
        bio_verdict, bio_motif_hits, bio_eclip = biological_plausibility_score(
            rna_sequence=rna_sequence,
            rbp_name=rbp_name,
            chrom=chrom,
            start=genomic_start,
            end=genomic_end,
            observed_rna_sequence=observed_rna_sequence,
            reference_native_sequence=reference_native_sequence,
        )

    # Combine branches
    if af3_mode:
        combined = _combine_verdicts_af3(
            gt_verdict,
            bio_verdict,
            clash_verdict,
            contact_verdict,
            bio_decisive=bool(rbp_name or rna_sequence),
            gt_physics_only=gt_physics_only,
            require_trained_gt=require_trained_gt,
        )
    else:
        active_verdicts = [
            v for v in [
                rmsd_verdict, geom_verdict, contact_verdict,
                clash_verdict, gt_verdict, bio_verdict,
            ]
            if v != "UNKNOWN"
        ]
        combined = _combine_verdicts(*active_verdicts) if active_verdicts else "UNKNOWN"

    confidence = _estimate_confidence(
        rmsd, geom_mean, per_nuc,
        contact_energy, clash_n_severe, gt_score,
        af3_mode=af3_mode,
    )

    return FilterResult(
        rmsd_score=rmsd,
        rmsd_nearest_cluster=rmsd_cluster,
        rmsd_verdict=rmsd_verdict,
        geom_score=geom_dist,
        geom_max_score=geom_max,
        geom_nearest_cluster=geom_cluster,
        geom_verdict=geom_verdict,
        contact_energy=contact_energy,
        contact_n_residues=contact_n_residues,
        contact_n_contacts=contact_n_contacts,
        contact_verdict=contact_verdict,
        clash_n_severe=clash_n_severe,
        clash_n_moderate=clash_n_moderate,
        clash_worst_distance=clash_worst,
        clash_verdict=clash_verdict,
        gt_score=gt_score,
        gt_score_raw=gt_score_raw,
        gt_score_norm=gt_score_norm,
        gt_score_per_nt=gt_score_per_nt,
        gt_verdict=gt_verdict,
        gt_physics_only=gt_physics_only,
        n_prot_rna_edges=n_prot_rna_edges,
        bio_verdict=bio_verdict,
        bio_motif_hits=bio_motif_hits,
        bio_eclip_supported=bio_eclip,
        per_nucleotide=per_nuc,
        combined_verdict=combined,
        confidence=confidence,
    )


def format_report(result: FilterResult, residue_ids: list, *, af3_mode: bool = True) -> str:
    """Formats the FilterResult as a human-readable text report."""

    verdict_symbol = {"PASS": "✓", "WARN": "~", "FAIL": "✗", "UNKNOWN": "?"}

    lines = [
        "",
        "PhysRNA-Filter Report (AF3 Augmented)",
        "─" * 52,
        f"Interface residues: {' '.join(f'{c}:{r}' for c, r in residue_ids)}",
        "",
        "RMSD Branch (entropic plausibility"
        + (", diagnostic for AF3 bound pose)" if af3_mode else ")")
        ,
        f"  Score:           {result.rmsd_score:.2f} A",
        f"  Nearest cluster: {result.rmsd_nearest_cluster}",
        f"  Verdict:         {verdict_symbol[result.rmsd_verdict]} {result.rmsd_verdict}",
        "",
        "Geometry Branch (local torsion plausibility"
        + (", intrinsic AF3 check; free-ensemble dist diagnostic)" if af3_mode else ")")
        ,
        (f"  Free-ensemble:   {result.geom_score:.1f} deg (diagnostic)"
         if af3_mode else
         f"  Cluster distance:{result.geom_score:.1f} deg"),
        (f"  Intrinsic score: {result.geom_max_score:.1f}"
         if af3_mode else
         f"  Max per-nuc:     {result.geom_max_score:.1f} deg"),
        f"  Nearest cluster: {result.geom_nearest_cluster}",
        f"  Verdict:         {verdict_symbol[result.geom_verdict]} {result.geom_verdict}",
        "",
        "Contact Physics Branch",
        f"  Total energy:    {result.contact_energy:.2f}",
        f"  Interface res:   {result.contact_n_residues}",
        f"  Contacts:        {result.contact_n_contacts}",
        f"  Verdict:         {verdict_symbol[result.contact_verdict]} {result.contact_verdict}",
        "",
        "Clash Branch (steric overlap)",
        f"  Severe clashes:  {result.clash_n_severe}",
        f"  Moderate:        {result.clash_n_moderate}",
        f"  Worst distance:  {result.clash_worst_distance:.2f} A",
        f"  Verdict:         {verdict_symbol[result.clash_verdict]} {result.clash_verdict}",
        "",
        "PhysGT Branch (learned interface plausibility)",
        f"  Score (norm):    {result.gt_score_norm:.2f}",
        f"  Score (raw):     {result.gt_score_raw:.2f}",
        f"  Score / nt:      {result.gt_score_per_nt:.2f}",
        f"  Mode:            {'physics-only' if result.gt_physics_only else 'GT model'}",
        f"  Verdict:         {verdict_symbol[result.gt_verdict]} {result.gt_verdict}",
        "",
        "Biological Branch (eCLIP + motifs)",
        f"  Motif hits:      {', '.join(result.bio_motif_hits) or 'none'}",
        f"  eCLIP support:   {result.bio_eclip_supported}",
        f"  Verdict:         {verdict_symbol[result.bio_verdict]} {result.bio_verdict}",
        "",
        "Per-Nucleotide Breakdown",
        f"  {'Residue':<12} {'Pucker':<12} {'Score':>10} {'Verdict':<8} Flags",
        "  " + "-" * 60,
    ]

    for s in result.per_nucleotide:
        chain, num = s.residue_id
        rid_str = f"{chain}:{num}"
        flag_str = "; ".join(s.flags[:1]) if s.flags else ""
        lines.append(
            f"  {rid_str:<12} {s.sugar_pucker:<12} {s.geometry_cluster_distance:>10.1f} "
            f"{verdict_symbol[s.verdict]} {s.verdict:<6} {flag_str}"
        )

    af3_note = (
        "  (AF3 mode: PhysGT + biology + true clashes; RMSD/geometry diagnostic)"
        if af3_mode else ""
    )
    lines += [
        "",
        "─" * 52,
        f"Hallucination verdict: {verdict_symbol[result.combined_verdict]} {result.combined_verdict}",
        af3_note,
        f"Confidence:        {result.confidence:.2f}",
        "",
    ]

    return "\n".join(lines)


# ── helpers ───────────────────────────────────────────────────────────────────

def _rmsd_verdict(rmsd: float) -> str:
    t = _get_thresholds()
    if rmsd < t["rmsd_pass"]:
        return "PASS"
    if rmsd < t["rmsd_warn"]:
        return "WARN"
    return "FAIL"


def _geom_verdict(dist: float) -> str:
    t = _get_thresholds()
    if dist < t["geom_pass"]:
        return "PASS"
    if dist < t["geom_warn"]:
        return "WARN"
    return "FAIL"


def _rmsd_verdict_af3(rmsd: float) -> str:
    """
    Diagnostic RMSD for bound AF3 pose vs free-RNA ensemble.

    High RMSD is expected (induced fit); only extreme values are flagged FAIL.
    """
    t = _get_thresholds()
    pass_t = t.get("rmsd_af3_pass", 18.0)
    warn_t = t.get("rmsd_af3_warn", 30.0)
    if rmsd < pass_t:
        return "PASS"
    if rmsd < warn_t:
        return "WARN"
    return "FAIL"


def _geom_verdict_af3(dist: float) -> str:
    """Relaxed geometry verdict for AF3 screening (diagnostic branch)."""
    t = _get_thresholds()
    pass_t = t.get("geom_af3_pass", 90.0)
    warn_t = t.get("geom_af3_warn", 140.0)
    if dist < pass_t:
        return "PASS"
    if dist < warn_t:
        return "WARN"
    return "FAIL"


_VERDICT_RANK = {"PASS": 0, "WARN": 1, "FAIL": 2, "UNKNOWN": -1}


def _worse_verdict(a: str, b: str) -> str:
    return a if _VERDICT_RANK[a] >= _VERDICT_RANK[b] else b


def _combine_verdicts(*verdicts: str) -> str:
    worst = "PASS"
    for v in verdicts:
        worst = _worse_verdict(worst, v)
    return worst


def _combine_verdicts_af3(
    gt_verdict: str,
    bio_verdict: str,
    clash_verdict: str,
    contact_verdict: str = "UNKNOWN",
    *,
    bio_decisive: bool = False,
    gt_physics_only: bool = False,
    require_trained_gt: bool = False,
) -> str:
    """
    AF3 hallucination verdict: PhysGT, biological partner match, and true
    steric overlaps.  RMSD, geometry, and contact energy are diagnostic only.
    """
    del contact_verdict  # not used for AF3 combined verdict

    if bio_verdict == "FAIL" or clash_verdict == "FAIL" or gt_verdict == "FAIL":
        return "FAIL"

    if require_trained_gt and gt_verdict == "UNKNOWN":
        return "FAIL"

    primary = [v for v in (gt_verdict, clash_verdict) if v != "UNKNOWN"]
    if not primary:
        if bio_verdict == "FAIL":
            return "FAIL"
        return "WARN"

    if any(v == "WARN" for v in primary):
        return "WARN"
    if gt_physics_only and gt_verdict != "FAIL":
        return "WARN"
    if bio_decisive and bio_verdict == "WARN":
        return "WARN"

    return "PASS"


def _estimate_confidence(
    rmsd: float,
    geom_dist: float,
    per_nuc: list[PerNucleotideScore],
    contact_energy: float = 0.0,
    clash_n_severe: int = 0,
    gt_score: float = 0.0,
    *,
    af3_mode: bool = False,
) -> float:
    """
    Heuristic confidence score (0-1) based on how far the scores are from
    the decision boundaries.
    """
    t = _get_thresholds()
    if af3_mode:
        gt_pass = t.get("gt_pass", -2.0)
        gt_warn = t.get("gt_warn", 0.5)
        gt_margin = (
            min(abs(gt_score - gt_pass), abs(gt_score - gt_warn)) / max(abs(gt_warn - gt_pass), 0.5)
            if gt_score else 0.35
        )
        clash_penalty = min(clash_n_severe * 0.12, 0.5)
        confidence = np.clip(0.75 * gt_margin + 0.25 * (1.0 - clash_penalty), 0.0, 1.0)
        return float(confidence)
    rmsd_margin = min(
        abs(rmsd - t["rmsd_pass"]),
        abs(rmsd - t["rmsd_warn"]),
    ) / t["rmsd_warn"]

    geom_margin = min(
        abs(geom_dist - t["geom_pass"]),
        abs(geom_dist - t["geom_warn"]),
    ) / t["geom_warn"]

    contact_margin = min(abs(contact_energy + 1.5), 3.0) / 3.0 if contact_energy else 0.5
    clash_penalty = clash_n_severe * 0.15
    gt_margin = min(abs(gt_score + 2.0), 4.0) / 4.0 if gt_score else 0.5

    n_fail = sum(1 for s in per_nuc if s.verdict == "FAIL")
    penalty = n_fail * 0.1 + clash_penalty

    confidence = np.clip(
        0.25 * (rmsd_margin + geom_margin + contact_margin + gt_margin) - penalty,
        0.0, 1.0,
    )
    return float(confidence)
