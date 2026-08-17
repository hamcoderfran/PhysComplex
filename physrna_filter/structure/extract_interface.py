"""
Identifies RNA nucleotides at the protein-RNA interface.

Uses a NeighborSearch on protein heavy atoms to find any RNA residue with an
atom within the cutoff distance. Returns both the residue identifiers and the
C4' backbone coordinates used downstream for RMSD computation.
"""

from __future__ import annotations

import numpy as np
from Bio.PDB import NeighborSearch

from .parse_complex import RNA_RESIDUE_NAMES


def find_interface_residues(
    protein_chains: list,
    rna_chains: list,
    cutoff: float = 5.0,
) -> list[tuple[str, int]]:
    """
    Returns list of (chain_id, residue_seq_number) for every RNA residue
    whose any heavy atom is within `cutoff` Angstroms of any protein atom.
    """
    protein_atoms = []
    for chain in protein_chains:
        for residue in chain:
            for atom in residue:
                if atom.element != "H":
                    protein_atoms.append(atom)

    if not protein_atoms:
        raise ValueError("No protein heavy atoms found — check chain classification.")

    ns = NeighborSearch(protein_atoms)
    interface: set[tuple[str, int]] = set()

    for chain in rna_chains:
        for residue in chain:
            if (
                residue.id[0] != " "
                or residue.resname.strip().upper() not in RNA_RESIDUE_NAMES
                or not _has_coordinate_anchor(residue)
            ):
                continue
            for atom in residue:
                if atom.element == "H":
                    continue
                nearby = ns.search(atom.coord, cutoff, level="R")
                if nearby:
                    interface.add((chain.id, residue.id[1]))

    result = sorted(interface, key=lambda x: (x[0], x[1]))
    print(f"Interface: {len(result)} nucleotides within {cutoff} A of protein")
    return result


def find_interface_protein_residues(
    protein_chains: list,
    rna_chains: list,
    cutoff: float = 5.0,
) -> list[tuple[str, int]]:
    """
    Returns protein residues with any heavy atom within `cutoff` of any RNA atom.
    """
    rna_atoms = []
    for chain in rna_chains:
        for residue in chain:
            if residue.id[0] != " ":
                continue
            for atom in residue:
                if atom.element != "H":
                    rna_atoms.append(atom)

    if not rna_atoms:
        return []

    ns = NeighborSearch(rna_atoms)
    interface: set[tuple[str, int]] = set()

    for chain in protein_chains:
        for residue in chain:
            if residue.id[0] != " ":
                continue
            for atom in residue:
                if atom.element == "H":
                    continue
                nearby = ns.search(atom.coord, cutoff, level="R")
                if nearby:
                    interface.add((chain.id, residue.id[1]))
                    break

    return sorted(interface, key=lambda x: (x[0], x[1]))


def extract_interface_coords(
    rna_chains: list,
    interface_residues: list[tuple[str, int]],
    atom_name: str = "C4'",
) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """
    Extracts coordinates for `atom_name` from each interface nucleotide.

    C4' is the standard backbone atom for RNA RMSD — it captures backbone
    geometry without noise from flexible base rotations.

    Returns:
        coords       — (N, 3) float array
        residue_ids  — ordered list matching rows in coords
    """
    coords = []
    residue_ids = []

    interface_set = set(interface_residues)

    for chain in rna_chains:
        for residue in chain:
            if (
                residue.id[0] != " "
                or residue.resname.strip().upper() not in RNA_RESIDUE_NAMES
            ):
                continue
            rid = (chain.id, residue.id[1])
            if rid not in interface_set:
                continue
            if atom_name in residue:
                coords.append(residue[atom_name].coord.copy())
                residue_ids.append(rid)
            else:
                # fallback to phosphorus if C4' is missing (e.g. terminal residue)
                if "P" in residue:
                    coords.append(residue["P"].coord.copy())
                    residue_ids.append(rid)

    if not coords:
        raise ValueError(
            f"No '{atom_name}' atoms found for interface residues. "
            "Check that residue numbering in interface_residues matches the PDB."
        )

    return np.array(coords, dtype=float), residue_ids


def extract_all_backbone_coords(
    rna_chains: list,
    interface_residues: list[tuple[str, int]],
) -> dict[tuple[str, int], dict[str, np.ndarray]]:
    """
    Extracts all heavy backbone atom coordinates per interface nucleotide.

    Returns dict mapping residue_id -> {atom_name: coord_array}.
    Useful for local geometry computation and hydrogen bond analysis.
    """
    backbone_atoms = {"P", "O5'", "C5'", "C4'", "C3'", "O3'", "C1'", "O4'", "C2'", "O2'"}
    interface_set = set(interface_residues)
    result = {}

    for chain in rna_chains:
        for residue in chain:
            if (
                residue.id[0] != " "
                or residue.resname.strip().upper() not in RNA_RESIDUE_NAMES
            ):
                continue
            rid = (chain.id, residue.id[1])
            if rid not in interface_set:
                continue
            atoms = {}
            for atom in residue:
                if atom.name in backbone_atoms:
                    atoms[atom.name] = atom.coord.copy()
            result[rid] = atoms

    return result


def rna_residue_to_trajectory_indices(
    rna_chains: list,
    interface_residues: list[tuple[str, int]],
) -> list[int]:
    """
    Map interface ``(chain_id, pdb_resnum)`` to 0-based indices along the
    simulated RNA strand (one bead / nucleotide per residue in chain order).
    """
    ordered: list[tuple[str, int]] = []
    for chain in rna_chains:
        for residue in chain:
            if residue.id[0] != " ":
                continue
            if residue.resname.strip().upper() not in RNA_RESIDUE_NAMES:
                continue
            ordered.append((chain.id, int(residue.id[1])))

    res_to_idx = {rid: i for i, rid in enumerate(ordered)}
    indices: list[int] = []
    for rid in interface_residues:
        if rid in res_to_idx:
            indices.append(res_to_idx[rid])
    if not indices:
        raise ValueError(
            "Could not map interface residues to RNA trajectory indices. "
            f"interface={interface_residues[:5]} ordered={ordered[:5]}"
        )
    return indices


def _has_coordinate_anchor(residue) -> bool:
    return "C4'" in residue or "P" in residue
