"""
Per-nucleotide local geometry scoring.

This module compares the local geometry of each interface nucleotide in the
AF3 bound pose against the conformational clusters derived from the free-RNA
simulation. It operates in torsion-angle space rather than Cartesian space,
catching hallucinations that look geometrically reasonable at the global level
but involve individual nucleotides locked into backbone conformations that were
never sampled in free solution.

The two most informative single-residue signals are:
  - Sugar pucker: C2'-endo (P ~ 144-190 deg) is rare and energetically costly
    in structured RNA. If an interface nucleotide adopts C2'-endo in the AF3
    prediction but all free-simulation snapshots are C3'-endo, the binding
    geometry is entropically penalized.
  - Chi angle: syn vs. anti glycosidic bond. Anti is normal; syn is rare and
    specific — AF3 forcing a syn conformation at an interface is suspicious
    unless it is a well-documented case (e.g. G-quadruplex, Hoogsteen pair).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cluster import nearest_geometry_cluster_distance, _angular_distance
from ..structure.local_geometry import NucleotideGeometry


@dataclass
class PerNucleotideScore:
    residue_id: tuple[str, int]
    resname: str

    # distance in torsion-angle space to nearest free-ensemble cluster centroid
    geometry_cluster_distance: float

    # individual torsion deviations from nearest centroid (degrees)
    alpha_dev:   float | None
    beta_dev:    float | None
    gamma_dev:   float | None
    delta_dev:   float | None
    epsilon_dev: float | None
    zeta_dev:    float | None
    chi_dev:     float | None

    # sugar pucker
    sugar_pucker: str
    pseudorotation_P: float | None

    verdict: str   # "PASS" | "WARN" | "FAIL"
    flags:   list[str]


def score_per_nucleotide(
    af3_geometry: dict[tuple[str, int], NucleotideGeometry],
    geom_centroids: np.ndarray,
    residue_order: list[tuple[str, int]],
    *,
    af3_mode: bool = False,
) -> list[PerNucleotideScore]:
    """
    Scores each interface nucleotide's local geometry against the free-ensemble
    torsion cluster centroids.

    Args:
        af3_geometry:   per-nucleotide geometry from the AF3 bound pose
        geom_centroids: (n_clusters, n_nucleotides * 9) from cluster_geometry
                        but split per-nucleotide here for per-residue scoring
        residue_order:  ordered list of interface residue IDs

    Returns:
        list of PerNucleotideScore, one per interface nucleotide
    """
    n_nuc      = len(residue_order)
    feat_per   = 9  # features per nucleotide

    # split centroids into per-nucleotide slices
    per_nuc_centroids = _split_centroids_per_nucleotide(
        geom_centroids, n_nuc, feat_per
    )

    results = []

    for i, rid in enumerate(residue_order):
        if rid not in af3_geometry:
            continue

        geom = af3_geometry[rid]
        pucker = geom.sugar_pucker_class()

        if af3_mode:
            dist, devs, flags, verdict = _score_nucleotide_intrinsic(geom)
        else:
            nuc_vec = geom.to_vector()           # (9,)
            centroids_for_nuc = per_nuc_centroids[i]  # (n_clusters, 9)

            dist, nearest_idx = _nearest_centroid_distance(nuc_vec, centroids_for_nuc)
            nearest_centroid  = centroids_for_nuc[nearest_idx]

            devs = _torsion_deviations(nuc_vec, nearest_centroid)
            flags  = _generate_flags(geom, devs, dist)
            verdict = _verdict_from_flags_and_dist(dist, flags)

        results.append(PerNucleotideScore(
            residue_id=rid,
            resname=geom.resname,
            geometry_cluster_distance=dist,
            alpha_dev=devs.get("alpha"),
            beta_dev=devs.get("beta"),
            gamma_dev=devs.get("gamma"),
            delta_dev=devs.get("delta"),
            epsilon_dev=devs.get("epsilon"),
            zeta_dev=devs.get("zeta"),
            chi_dev=devs.get("chi"),
            sugar_pucker=pucker,
            pseudorotation_P=geom.pseudorotation_P,
            verdict=verdict,
            flags=flags,
        ))

    return results


def aggregate_geometry_score(
    per_nuc_scores: list[PerNucleotideScore],
    *,
    af3_mode: bool = False,
) -> dict:
    """
    Aggregates per-nucleotide geometry scores into an interface-level summary.

    Returns:
        mean_distance   — mean torsion-cluster distance across nucleotides
        max_distance    — worst-case nucleotide
        n_warn          — count of nucleotides with WARN verdict
        n_fail          — count of nucleotides with FAIL verdict
        verdict         — "PASS" | "WARN" | "FAIL"
        flagged_residues — list of residue IDs with flags
    """
    if not per_nuc_scores:
        return {"verdict": "UNKNOWN", "mean_distance": None, "max_distance": None}

    distances = [s.geometry_cluster_distance for s in per_nuc_scores]
    verdicts  = [s.verdict for s in per_nuc_scores]
    flagged   = [s.residue_id for s in per_nuc_scores if s.flags]

    mean_dist = float(np.mean(distances))
    max_dist  = float(np.max(distances))
    n_fail    = verdicts.count("FAIL")
    n_warn    = verdicts.count("WARN")

    if af3_mode:
        if n_fail > 0:
            verdict = "FAIL"
        elif n_warn > 0:
            verdict = "WARN"
        else:
            verdict = "PASS"
    elif n_fail > 0:
        verdict = "FAIL"
    elif n_warn > 1 or mean_dist > 45.0:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return {
        "verdict":          verdict,
        "mean_distance":    mean_dist,
        "max_distance":     max_dist,
        "n_warn":           n_warn,
        "n_fail":           n_fail,
        "flagged_residues": flagged,
    }


# ── internal helpers ──────────────────────────────────────────────────────────

def _split_centroids_per_nucleotide(
    centroids: np.ndarray,
    n_nuc: int,
    feat_per: int,
) -> list[np.ndarray]:
    """
    Splits a (n_clusters, n_nuc * feat_per) centroid matrix into a list of
    (n_clusters, feat_per) arrays, one per nucleotide.
    """
    result = []
    for i in range(n_nuc):
        start = i * feat_per
        end   = start + feat_per
        result.append(centroids[:, start:end])
    return result


def _nearest_centroid_distance(
    vec: np.ndarray,
    centroids: np.ndarray,
) -> tuple[float, int]:
    """
    Finds the nearest centroid in torsion-angle space using angular distance.
    Returns (distance, centroid_index).
    """
    min_dist = np.inf
    nearest  = 0

    for i, c in enumerate(centroids):
        d = _angular_distance(vec, c)
        if d < min_dist:
            min_dist = d
            nearest  = i

    return float(min_dist), nearest


def _torsion_deviations(
    vec: np.ndarray,
    centroid: np.ndarray,
) -> dict[str, float]:
    """
    Computes the per-torsion circular deviation from the nearest centroid.

    vec and centroid both have shape (9,):
    [alpha, beta, gamma, delta, epsilon, zeta, chi, P, vmax]
    """
    names = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi", "P", "vmax"]
    devs  = {}

    for i, name in enumerate(names):
        raw = abs(float(vec[i]) - float(centroid[i]))
        dev = min(raw, 360.0 - raw)
        devs[name] = dev

    return devs


def _generate_flags(
    geom: NucleotideGeometry,
    devs: dict[str, float],
    cluster_distance: float,
) -> list[str]:
    """
    Generates human-readable flags for suspicious local geometry features.
    """
    flags = []

    pucker = geom.sugar_pucker_class()
    if pucker == "C2'-endo":
        flags.append(
            f"C2'-endo sugar pucker (P={geom.pseudorotation_P:.1f} deg) — "
            "unusual for RNA, energetically costly"
        )

    if geom.chi is not None:
        if -60 <= geom.chi <= 60:
            flags.append(
                f"syn glycosidic bond (chi={geom.chi:.1f} deg) — "
                "rare, suspect unless at Hoogsteen or G-quadruplex interface"
            )

    for torsion_name in ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi"]:
        dev = devs.get(torsion_name, 0.0)
        if dev > 60.0:
            flags.append(
                f"{torsion_name} deviation {dev:.0f} deg from nearest free-ensemble cluster"
            )

    if cluster_distance > 90.0:
        flags.append(
            f"Torsion-cluster distance {cluster_distance:.0f} deg — "
            "nucleotide geometry not observed in free simulation"
        )

    return flags


def _verdict_from_flags_and_dist(dist: float, flags: list[str]) -> str:
    if dist > 90.0 or any("C2'-endo" in f for f in flags):
        return "FAIL"
    if dist > 45.0 or flags:
        return "WARN"
    return "PASS"


def _score_nucleotide_intrinsic(
    geom: NucleotideGeometry,
) -> tuple[float, dict[str, float | None], list[str], str]:
    """
    AF3 bound-pose intrinsic geometry check (no free-ensemble comparison).

    Bound interfaces routinely deviate from free-solution clusters; we only
    flag chemistries that are rare/suspicious on their own (syn chi, extreme
    pucker outliers).
    """
    flags: list[str] = []
    severity = 0.0

    pucker = geom.sugar_pucker_class()
    if pucker == "C2'-endo" and geom.pseudorotation_P is not None:
        flags.append(
            f"C2'-endo sugar pucker (P={geom.pseudorotation_P:.1f} deg) — "
            "unusual for RNA, review if not induced-fit"
        )
        severity += 25.0

    if geom.chi is not None and -60 <= geom.chi <= 60:
        flags.append(
            f"syn glycosidic bond (chi={geom.chi:.1f} deg) — "
            "rare, suspect unless at Hoogsteen or G-quadruplex interface"
        )
        severity += 40.0

    if geom.pseudorotation_P is not None and geom.pseudorotation_P < -120:
        flags.append(
            f"extreme sugar pucker phase (P={geom.pseudorotation_P:.1f} deg)"
        )
        severity += 20.0

    if any("syn" in f for f in flags):
        verdict = "FAIL"
    elif flags:
        verdict = "WARN"
    else:
        verdict = "PASS"

    devs = {
        "alpha": None, "beta": None, "gamma": None, "delta": None,
        "epsilon": None, "zeta": None, "chi": None,
    }
    return severity, devs, flags, verdict
