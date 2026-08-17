"""
Select the protein–RNA chain pair for a deposited or predicted complex.

Crystal structures often contain multiple copies of the same complex (e.g. 1URN
A/B/C × P/Q/R). Scoring must restrict to the biologically relevant pair, not
crystal-packing partners.
"""
from __future__ import annotations

from Bio.PDB import NeighborSearch

from .parse_complex import RNA_RESIDUE_NAMES, get_all_atoms


def _heavy_atoms(chains: list) -> list:
    return [
        a
        for ch in chains
        for res in ch
        if res.id[0] == " "
        for a in res
        if a.element != "H"
    ]


def count_chain_contacts(
    protein_chain,
    rna_chain,
    cutoff: float = 10.0,
) -> int:
    """Count RNA heavy atoms within *cutoff* Å of any protein heavy atom."""
    prot_atoms = [
        a for res in protein_chain
        if res.id[0] == " "
        for a in res
        if a.element != "H"
    ]
    if not prot_atoms:
        return 0
    ns = NeighborSearch(prot_atoms)
    count = 0
    for residue in rna_chain:
        if residue.id[0] != " ":
            continue
        if residue.resname.strip().upper() not in RNA_RESIDUE_NAMES:
            continue
        for atom in residue:
            if atom.element == "H":
                continue
            if ns.search(atom.coord, cutoff):
                count += 1
    return count


def select_partner_pair(
    protein_chains: list,
    rna_chains: list,
    *,
    protein_chain_id: str | None = None,
    rna_chain_id: str | None = None,
    cutoff: float = 10.0,
) -> tuple[list, list]:
    """
    Return (protein_chains, rna_chains) restricted to the best-matching pair.

    When only one protein and one RNA chain exist, returns them unchanged.
    """
    if not protein_chains or not rna_chains:
        return protein_chains, rna_chains

    prot = protein_chains
    rna = rna_chains

    if protein_chain_id is not None:
        prot = [c for c in protein_chains if c.id == protein_chain_id] or protein_chains
    if rna_chain_id is not None:
        rna = [c for c in rna_chains if c.id == rna_chain_id] or rna_chains

    if len(prot) == 1 and len(rna) == 1:
        return prot, rna

    best_score = -1
    best_prot = prot[0]
    best_rna = rna[0]

    for p_chain in prot:
        for r_chain in rna:
            score = count_chain_contacts(p_chain, r_chain, cutoff=cutoff)
            if score > best_score:
                best_score = score
                best_prot = p_chain
                best_rna = r_chain

    if best_score <= 0:
        return prot[:1], rna[:1]
    return [best_prot], [best_rna]


def select_partner_rna_chains(
    protein_chains: list,
    rna_chains: list,
    protein_chain_id: str,
    cutoff: float = 10.0,
) -> list:
    """
    Restrict RNA chains to those paired with *protein_chain_id*.

    Backward-compatible helper used by ProNAB graph building and contact scoring.
    """
    target = next((c for c in protein_chains if c.id == protein_chain_id), None)
    if target is None or len(rna_chains) <= 1:
        return rna_chains

    counts = {ch.id: count_chain_contacts(target, ch, cutoff=cutoff) for ch in rna_chains}
    max_count = max(counts.values(), default=0)
    if max_count == 0:
        return rna_chains

    threshold = max(1, 0.3 * max_count)
    selected = [ch for ch in rna_chains if counts.get(ch.id, 0) >= threshold]
    return selected if selected else rna_chains


def primary_protein_chain_id(protein_chains: list, rna_chains: list) -> str | None:
    """Chain id of the protein with the most RNA contacts (for embedding lookup)."""
    if not protein_chains:
        return None
    if len(protein_chains) == 1:
        return protein_chains[0].id

    best_id = protein_chains[0].id
    best_score = -1
    for p_chain in protein_chains:
        score = sum(count_chain_contacts(p_chain, r, cutoff=10.0) for r in rna_chains)
        if score > best_score:
            best_score = score
            best_id = p_chain.id
    return best_id


def primary_rna_chain_id(rna_chains: list, protein_chains: list) -> str | None:
    """Chain id of the RNA with the most protein contacts."""
    if not rna_chains:
        return None
    if len(rna_chains) == 1:
        return rna_chains[0].id

    prot_atoms = _heavy_atoms(protein_chains)
    if not prot_atoms:
        return rna_chains[0].id

    ns = NeighborSearch(prot_atoms)
    best_id = rna_chains[0].id
    best_score = -1
    for r_chain in rna_chains:
        count = 0
        for residue in r_chain:
            if residue.id[0] != " ":
                continue
            for atom in residue:
                if atom.element == "H":
                    continue
                if ns.search(atom.coord, 10.0):
                    count += 1
        if count > best_score:
            best_score = count
            best_id = r_chain.id
    return best_id
