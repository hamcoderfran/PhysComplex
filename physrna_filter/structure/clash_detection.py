"""
Steric clash detection at protein-RNA interfaces.

Catches geometric hallucinations where AF3 places atoms too close together —
a failure mode distinct from entropic and torsion-based checks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from Bio.PDB import NeighborSearch

from .parse_complex import RNA_RESIDUE_NAMES, get_all_atoms


# Literature-inspired thresholds (Richardson & Richardson clash criteria adapted)
CLASH_SEVERE_DIST = 2.0    # Å — decoy / same-structure screening
CLASH_MODERATE_DIST = 2.4  # Å — borderline contact
CLASH_AF3_SEVERE_DIST = 1.15   # Å — true heavy-atom overlap at AF3 interfaces
CLASH_AF3_MODERATE_DIST = 1.35
CLASH_PASS_MAX = 0         # no severe clashes
CLASH_WARN_MAX = 2         # up to 2 moderate clashes tolerated
CLASH_FAIL_MAX = 5
CLASH_AF3_FAIL_WORST = 0.95
CLASH_AF3_FAIL_SEVERE = 6


@dataclass
class ClashResult:
    n_severe: int
    n_moderate: int
    n_total: int
    worst_distance: float
    clash_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    verdict: str = "UNKNOWN"


def _residue_label(atom) -> str:
    res = atom.get_parent()
    chain = res.get_parent()
    return f"{chain.id}:{res.id[1]}:{atom.name}"


def _atom_pair_key(a1, a2) -> tuple[str, str]:
    return tuple(sorted([_residue_label(a1), _residue_label(a2)]))


def detect_interface_clashes(
    protein_chains: list,
    rna_chains: list,
    interface_rna_residues: list[tuple[str, int]] | None = None,
    severe_dist: float | None = None,
    moderate_dist: float | None = None,
    *,
    af3_mode: bool = False,
) -> ClashResult:
    """
    Count steric clashes between protein and RNA heavy atoms at the interface.

    Also checks RNA-RNA clashes among interface nucleotides (common AF3 failure
    when the model compresses the binding site).

    Args:
        protein_chains: parsed protein chains
        rna_chains:     parsed RNA chains
        interface_rna_residues: optional list of (chain, resnum) to restrict
            RNA clash checking; if None, all RNA atoms are considered.
        af3_mode: use tighter overlap thresholds for bound AF3 complexes.
            Close protein–RNA contacts (1.3–2.0 Å) are normal at interfaces and
            should not be counted as hallucinations.
    """
    if severe_dist is None:
        severe_dist = CLASH_AF3_SEVERE_DIST if af3_mode else CLASH_SEVERE_DIST
    if moderate_dist is None:
        moderate_dist = CLASH_AF3_MODERATE_DIST if af3_mode else CLASH_MODERATE_DIST
    prot_atoms = [
        a for ch in protein_chains
        for res in ch
        if res.id[0] == " "
        for a in res
        if a.element != "H"
    ]

    if interface_rna_residues is not None:
        iface_set = set(interface_rna_residues)
        rna_atoms = [
            a for ch in rna_chains
            for res in ch
            if (ch.id, res.id[1]) in iface_set
            for a in res
            if a.element != "H"
        ]
    else:
        rna_atoms = get_all_atoms(rna_chains)

    if not prot_atoms or not rna_atoms:
        return ClashResult(0, 0, 0, float("inf"), verdict="PASS")

    ns_prot = NeighborSearch(prot_atoms)
    ns_rna  = NeighborSearch(rna_atoms)

    clash_pairs: list[tuple[str, str, float]] = []
    seen: set[tuple[str, str]] = set()

    # protein ↔ RNA clashes
    for ratom in rna_atoms:
        nearby = ns_prot.search(ratom.coord, moderate_dist, level="A")
        for patom in nearby:
            if patom.element == "H":
                continue
            d = float(np.linalg.norm(patom.coord - ratom.coord))
            if d < 0.5:   # covalent bond, not a clash
                continue
            key = _atom_pair_key(patom, ratom)
            if key in seen:
                continue
            seen.add(key)
            clash_pairs.append((key[0], key[1], d))

    # RNA ↔ RNA clashes at interface
    for a1 in rna_atoms:
        nearby = ns_rna.search(a1.coord, moderate_dist, level="A")
        for a2 in nearby:
            if a1 is a2:
                continue
            if a1.get_parent() is a2.get_parent():
                continue
            d = float(np.linalg.norm(a1.coord - a2.coord))
            if d < 0.5:
                continue
            key = _atom_pair_key(a1, a2)
            if key in seen:
                continue
            seen.add(key)
            clash_pairs.append((key[0], key[1], d))

    n_severe   = sum(1 for _, _, d in clash_pairs if d < severe_dist)
    n_moderate = sum(1 for _, _, d in clash_pairs if severe_dist <= d < moderate_dist)
    worst = min((d for _, _, d in clash_pairs), default=float("inf"))

    verdict = _clash_verdict(n_severe, n_moderate, worst, af3_mode=af3_mode)

    return ClashResult(
        n_severe=n_severe,
        n_moderate=n_moderate,
        n_total=len(clash_pairs),
        worst_distance=worst,
        clash_pairs=sorted(clash_pairs, key=lambda x: x[2])[:20],
        verdict=verdict,
    )


def _clash_verdict(
    n_severe: int,
    n_moderate: int,
    worst: float = float("inf"),
    *,
    af3_mode: bool = False,
) -> str:
    if af3_mode:
        if worst < CLASH_AF3_FAIL_WORST or n_severe > CLASH_AF3_FAIL_SEVERE:
            return "FAIL"
        if n_severe > 0 or n_moderate > CLASH_WARN_MAX:
            return "WARN"
        return "PASS"
    if n_severe > CLASH_PASS_MAX:
        return "FAIL"
    if n_severe + n_moderate > CLASH_WARN_MAX:
        return "WARN" if n_severe == 0 else "FAIL"
    return "PASS"
