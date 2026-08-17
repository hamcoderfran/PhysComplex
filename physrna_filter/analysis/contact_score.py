"""
Per-residue contact energy scoring for protein-RNA interfaces.

This is the primary layer for the ProNAB ΔΔG benchmark. When a protein
residue is mutated (e.g. Y13A in ProNAB), the RNA coordinates barely change
in the experimental structure. What changes is the SET OF CONTACTS that
residue makes with the RNA. The score delta between WT and mutant contact
scores is the ddg proxy that correlates with experimental ΔΔG.

Four interaction types are scored, reflecting the key physics of RNA-protein
interfaces:

  1. Electrostatic — phosphate oxygens (OP1/OP2, charge ~-0.8) against
     Lys (NZ), Arg (NH1/NH2/NE), and His (ND1/NE2). These are the single
     largest contributors — losing a Lys or Arg at an RNA interface is
     worth 3-5 kcal/mol and the dominant signal in ProNAB.

  2. Aromatic stacking — Phe/Tyr/Trp ring atoms against purine/pyrimidine
     ring atoms. Contributes ~1-3 kcal/mol and explains Y/W/F→A mutations.

  3. Hydrogen bonds — N/O donor-acceptor pairs within 3.5 Å. Captures Ser,
     Thr, Asn, Gln mutations and contacts to the 2'-OH of RNA.

  4. Van der Waals base — all other heavy atom contacts, distance-weighted.
     Captures the burial contribution of larger residues (Val→Gly, etc.).

The score for a residue is a weighted sum over all contacts it makes with
RNA within the cutoff. The score delta (WT - mutant) is the ΔΔG proxy.

Weights are physically motivated but would benefit from ProNAB-based
regression training to push Pearson r above 0.6 reliably.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ── atom charge table ─────────────────────────────────────────────────────────
# Partial charges relevant for RNA-protein electrostatics (sign: positive is +)

PROTEIN_CHARGE: dict[str, float] = {
    # Lysine
    "NZ":  +0.80,
    # Arginine
    "NH1": +0.60,
    "NH2": +0.60,
    "NE":  +0.40,
    "CZ":  +0.20,   # Arg CZ is partially positive too
    # Histidine (protonated, positive)
    "ND1": +0.30,
    "NE2": +0.30,
    # Glutamate / Aspartate (negative — penalize contact with RNA phosphate)
    "OE1": -0.40,
    "OE2": -0.40,
    "OD1": -0.40,
    "OD2": -0.40,
}

RNA_CHARGE: dict[str, float] = {
    # Phosphate oxygens — highly negative, key electrostatic partners
    "OP1": -0.80,
    "OP2": -0.80,
    "O1P": -0.80,   # alternate naming
    "O2P": -0.80,
    # 2'-OH oxygen — weaker, but important H-bond acceptor
    "O2'": -0.20,
    # Base ring nitrogens (H-bond acceptors)
    "N1":  -0.15,
    "N3":  -0.15,
    "N7":  -0.15,
}

# ── aromatic atom sets ────────────────────────────────────────────────────────
# Keyed by residue name so atom names like "CG" aren't falsely matched
# in non-aromatic residues (ARG, LYS, MET, etc. all have a CG atom).

PROTEIN_AROMATIC_ATOMS: dict[str, set[str]] = {
    "PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"},
    "TRP": {"CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "HIS": {"CG", "ND1", "CD2", "CE1", "NE2"},
}

RNA_AROMATIC: set[str] = {
    # Purine ring atoms (A, G)
    "C2", "C4", "C5", "C6", "C8",
    "N1", "N3", "N7", "N9",
    # Pyrimidine ring atoms (U, C)
    "C5", "C6",
}

# ── interaction weights ───────────────────────────────────────────────────────
# Tuned to produce score deltas in a kcal/mol-comparable range
# across the ProNAB mutation types.

W_ELECTROSTATIC = 3.0   # dominant for K/R→A mutations
W_STACKING      = 1.8   # dominant for Y/W/F→A mutations
W_HBOND         = 1.5   # captures S/T/N/Q mutations and 2'-OH contacts
W_VDW           = 0.4   # background burial term

HBOND_DIST_CUTOFF = 3.5   # Angstroms
CONTACT_CUTOFF    = 5.5   # Angstroms for all contacts


@dataclass
class ResidueContactScore:
    residue_id: tuple[str, int]
    resname: str
    electrostatic: float
    stacking:      float
    hbond:         float
    vdw:           float
    n_contacts:    int
    total:         float


def score_residue_contacts(
    residue,
    rna_atoms: list,
    cutoff: float = CONTACT_CUTOFF,
    dielectric: bool = False,
) -> ResidueContactScore:
    """
    Scores the contacts one protein residue makes with all RNA atoms.

    Args:
        residue:    Bio.PDB Residue object
        rna_atoms:  flat list of Bio.PDB Atom objects from all RNA chains
        cutoff:     maximum distance to consider
        dielectric: if True, weight electrostatic contacts with a
                    distance-dependent-dielectric Coulomb term
                    (_electrostatic_weight) instead of the generic
                    exponential decay used for the other interaction types.

    Returns:
        ResidueContactScore with per-type breakdown and total
    """
    elec   = 0.0
    stack  = 0.0
    hbond  = 0.0
    vdw    = 0.0
    n_contacts = 0

    for patom in residue:
        if patom.element == "H":
            continue
        pcoord = patom.coord
        p_charge   = PROTEIN_CHARGE.get(patom.name, 0.0)
        _aro_atoms = PROTEIN_AROMATIC_ATOMS.get(residue.resname, set())
        p_aromatic = patom.name in _aro_atoms

        for ratom in rna_atoms:
            if ratom.element == "H":
                continue
            dist = float(np.linalg.norm(pcoord - ratom.coord))
            if dist > cutoff or dist < 0.1:
                continue

            n_contacts += 1
            w = _dist_weight(dist)

            # electrostatic
            r_charge = RNA_CHARGE.get(ratom.name, 0.0)
            if p_charge != 0.0 and r_charge != 0.0:
                elec_w = _electrostatic_weight(dist) if dielectric else w
                # opposite signs → negative product → negative (favorable) contribution
                elec += W_ELECTROSTATIC * elec_w * abs(p_charge * r_charge) * np.sign(p_charge * r_charge)

            # aromatic stacking
            if p_aromatic and ratom.name in RNA_AROMATIC:
                stack -= W_STACKING * w   # negative = favorable

            # hydrogen bond (N/O pairs within hbond cutoff)
            if (
                patom.element in ("N", "O")
                and ratom.element in ("N", "O")
                and dist <= HBOND_DIST_CUTOFF
            ):
                hbond -= W_HBOND * w   # negative = favorable

            # van der Waals base contact
            vdw -= W_VDW * w

    total = elec + stack + hbond + vdw

    return ResidueContactScore(
        residue_id=(residue.get_parent().id, residue.id[1]),
        resname=residue.resname,
        electrostatic=elec,
        stacking=stack,
        hbond=hbond,
        vdw=vdw,
        n_contacts=n_contacts,
        total=total,
    )


def score_residue_contacts_capped(
    residue,
    rna_atoms: list,
    cutoff: float = CONTACT_CUTOFF,
) -> ResidueContactScore:
    """
    Like score_residue_contacts, but caps each interaction type's
    contribution per distinct RNA residue to its single strongest atom-pair
    value, rather than summing over every atom pair.

    Prevents residues that are deeply intercalated between RNA bases (and
    thus have 5-10x the typical atom-pair count, e.g. 1URN F56, 4CIO Y44)
    from producing score deltas an order of magnitude larger than typical
    interface residues.
    """
    best: dict[tuple, dict[str, float]] = {}

    for patom in residue:
        if patom.element == "H":
            continue
        pcoord = patom.coord
        p_charge   = PROTEIN_CHARGE.get(patom.name, 0.0)
        _aro_atoms = PROTEIN_AROMATIC_ATOMS.get(residue.resname, set())
        p_aromatic = patom.name in _aro_atoms

        for ratom in rna_atoms:
            if ratom.element == "H":
                continue
            dist = float(np.linalg.norm(pcoord - ratom.coord))
            if dist > cutoff or dist < 0.1:
                continue

            rna_residue = ratom.get_parent()
            rna_chain   = rna_residue.get_parent()
            key = (rna_chain.id, rna_residue.id[1])
            slot = best.setdefault(key, {"elec": 0.0, "stack": 0.0, "hbond": 0.0, "vdw": 0.0})

            w = _dist_weight(dist)

            r_charge = RNA_CHARGE.get(ratom.name, 0.0)
            if p_charge != 0.0 and r_charge != 0.0:
                contrib = W_ELECTROSTATIC * w * abs(p_charge * r_charge) * np.sign(p_charge * r_charge)
                if abs(contrib) > abs(slot["elec"]):
                    slot["elec"] = contrib

            if p_aromatic and ratom.name in RNA_AROMATIC:
                contrib = -W_STACKING * w
                if contrib < slot["stack"]:
                    slot["stack"] = contrib

            if (
                patom.element in ("N", "O")
                and ratom.element in ("N", "O")
                and dist <= HBOND_DIST_CUTOFF
            ):
                contrib = -W_HBOND * w
                if contrib < slot["hbond"]:
                    slot["hbond"] = contrib

            contrib = -W_VDW * w
            if contrib < slot["vdw"]:
                slot["vdw"] = contrib

    elec  = sum(v["elec"]  for v in best.values())
    stack = sum(v["stack"] for v in best.values())
    hbond = sum(v["hbond"] for v in best.values())
    vdw   = sum(v["vdw"]   for v in best.values())

    return ResidueContactScore(
        residue_id=(residue.get_parent().id, residue.id[1]),
        resname=residue.resname,
        electrostatic=elec,
        stacking=stack,
        hbond=hbond,
        vdw=vdw,
        n_contacts=len(best),
        total=elec + stack + hbond + vdw,
    )


def compute_contact_ddg_proxy(
    wt_pdb_path: str,
    mut_pdb_path: str,
    chain_id: str,
    position: int,
    capped: bool = False,
    dielectric: bool = False,
) -> dict:
    """
    Computes the contact-energy score delta between a wild-type complex and
    its computationally mutated variant.

    score_delta > 0 means mutation weakened the interface (predicts ΔΔG > 0).

    Returns:
        dict with wt_score, mut_score, score_delta, per-type deltas
    """
    from ..structure.parse_complex import parse_complex, get_all_atoms

    wt_parsed  = parse_complex(wt_pdb_path)
    mut_parsed = parse_complex(mut_pdb_path)

    from ..structure.partner_selection import select_partner_rna_chains as _select_partner_rna_chains

    wt_rna_atoms  = get_all_atoms(_select_partner_rna_chains(wt_parsed.protein_chains,  wt_parsed.rna_chains,  chain_id))
    mut_rna_atoms = get_all_atoms(_select_partner_rna_chains(mut_parsed.protein_chains, mut_parsed.rna_chains, chain_id))

    wt_res  = _find_residue(wt_parsed.protein_chains,  chain_id, position)
    mut_res = _find_residue(mut_parsed.protein_chains, chain_id, position)

    if wt_res is None or mut_res is None:
        raise ValueError(f"Residue {chain_id}:{position} not found in one of the structures")

    if capped:
        wt_score  = score_residue_contacts_capped(wt_res,  wt_rna_atoms)
        mut_score = score_residue_contacts_capped(mut_res, mut_rna_atoms)
    else:
        wt_score  = score_residue_contacts(wt_res,  wt_rna_atoms, dielectric=dielectric)
        mut_score = score_residue_contacts(mut_res, mut_rna_atoms, dielectric=dielectric)

    # mut.total > wt.total (less negative) when mutation removes contacts
    # → positive delta = mutation weakened interface = predicts ddG > 0
    delta = mut_score.total - wt_score.total

    return {
        "wt_score":         wt_score.total,
        "mut_score":        mut_score.total,
        "score_delta":      delta,
        # component deltas use the same mut - wt convention as score_delta:
        # positive = mutation removed that favorable contact (predicts ddG > 0)
        "elec_delta":       mut_score.electrostatic - wt_score.electrostatic,
        "stack_delta":      mut_score.stacking      - wt_score.stacking,
        "hbond_delta":      mut_score.hbond         - wt_score.hbond,
        "vdw_delta":        mut_score.vdw           - wt_score.vdw,
        "wt_n_contacts":    wt_score.n_contacts,
        "mut_n_contacts":   mut_score.n_contacts,
    }


def score_full_interface(
    protein_chains: list,
    rna_chains: list,
    interface_residues: list[tuple[str, int]],
    cutoff: float = CONTACT_CUTOFF,
) -> dict[tuple[str, int], ResidueContactScore]:
    """
    Scores every interface protein residue's contribution to the interface.

    Returns dict mapping (chain_id, resnum) -> ResidueContactScore.
    Useful for identifying hot-spot residues in AF3 predictions.
    """
    from ..structure.parse_complex import get_all_atoms

    rna_atoms = get_all_atoms(rna_chains)

    interface_set = set(interface_residues)
    scores = {}

    for chain in protein_chains:
        for residue in chain:
            rid = (chain.id, residue.id[1])
            if rid not in interface_set:
                continue
            scores[rid] = score_residue_contacts(residue, rna_atoms, cutoff)

    return scores


def aggregate_interface_contact_energy(
    protein_chains: list,
    rna_chains: list,
    interface_protein_residues: list[tuple[str, int]] | None = None,
    cutoff: float = CONTACT_CUTOFF,
    *,
    af3_mode: bool = False,
) -> dict:
    """
    Aggregate contact physics over the full protein-RNA interface.

    Returns dict with total energy (negative = favorable), per-type breakdown,
    n_interface_residues, n_contacts, and verdict.

    Hallucinated AF3 interfaces often show weak electrostatic/H-bond complementarity
    despite geometric proximity — this branch catches that failure mode.
    """
    from ..structure.extract_interface import find_interface_protein_residues
    from ..structure.parse_complex import get_all_atoms

    if interface_protein_residues is None:
        interface_protein_residues = find_interface_protein_residues(
            protein_chains, rna_chains, cutoff=cutoff
        )

    rna_atoms = get_all_atoms(rna_chains)
    iface_set = set(interface_protein_residues)

    totals = {"electrostatic": 0.0, "stacking": 0.0, "hbond": 0.0, "vdw": 0.0}
    n_contacts = 0
    n_residues = 0

    for chain in protein_chains:
        for residue in chain:
            rid = (chain.id, residue.id[1])
            if rid not in iface_set:
                continue
            if af3_mode:
                score = score_residue_contacts_capped(residue, rna_atoms, cutoff)
            else:
                score = score_residue_contacts(residue, rna_atoms, cutoff)
            totals["electrostatic"] += score.electrostatic
            totals["stacking"]      += score.stacking
            totals["hbond"]         += score.hbond
            totals["vdw"]           += score.vdw
            n_contacts += score.n_contacts
            n_residues += 1

    total = sum(totals.values())
    verdict = _contact_verdict(total, n_contacts, n_residues)

    return {
        "total_energy": total,
        **totals,
        "n_interface_residues": n_residues,
        "n_contacts": n_contacts,
        "verdict": verdict,
    }


def _contact_verdict(total_energy: float, n_contacts: int, n_residues: int) -> str:
    """More negative total = more favorable interface."""
    from .thresholds import load_thresholds

    t = load_thresholds()
    pass_pr = t.get("contact_pass_per_residue", -1.5)
    warn_pr = t.get("contact_warn_per_residue", -0.3)

    if n_residues == 0 or n_contacts == 0:
        return "FAIL"
    per_residue = total_energy / max(n_residues, 1)
    if per_residue < pass_pr and n_contacts >= 3:
        return "PASS"
    if per_residue < warn_pr or n_contacts >= 2:
        return "WARN"
    return "FAIL"


# ── internal helpers ──────────────────────────────────────────────────────────

def _dist_weight(dist: float, scale: float = 3.5) -> float:
    """
    Exponential distance decay: w(d) = exp(-d / scale).
    At d=3.5 Å (typical contact): w ≈ 0.37
    At d=5.5 Å (cutoff):          w ≈ 0.21
    """
    return float(np.exp(-dist / scale))


def _electrostatic_weight(dist: float) -> float:
    """
    Distance-dependent-dielectric Coulomb weight: w(d) = k / (epsilon(d) * d),
    with a linear distance-dependent dielectric epsilon(d) = 4*d (a standard
    implicit-solvent screening approximation), giving w(d) = k / (4*d^2).

    k is chosen so w(3.5) == _dist_weight(3.5) (~0.368), keeping
    W_ELECTROSTATIC on the same scale as before. Relative to the exponential
    decay, this weights close ion pairs (<3 A) more heavily and contacts
    near the cutoff less heavily, reflecting that electrostatics is screened
    more strongly as charges separate.
    At d=2.0 Å (salt bridge):  w ≈ 1.13  (vs exponential ≈ 0.56)
    At d=3.5 Å (typical):      w ≈ 0.37  (matches exponential)
    At d=5.5 Å (cutoff):       w ≈ 0.15  (vs exponential ≈ 0.21)
    """
    k = _dist_weight(3.5) * 4.0 * 3.5 * 3.5
    return float(k / (4.0 * dist * dist))


def _select_partner_rna_chains(protein_chains: list, rna_chains: list, chain_id: str, cutoff: float = 10.0) -> list:
    """Deprecated alias — use ``structure.partner_selection.select_partner_rna_chains``."""
    from ..structure.partner_selection import select_partner_rna_chains
    return select_partner_rna_chains(protein_chains, rna_chains, chain_id, cutoff=cutoff)


def _find_residue(protein_chains: list, chain_id: str, position: int):
    for chain in protein_chains:
        if chain.id != chain_id:
            continue
        for residue in chain:
            if residue.id[1] == position:
                return residue
    return None
