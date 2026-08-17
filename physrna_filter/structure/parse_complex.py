"""
Parses a PDB file and separates protein and RNA chains.

RNA residues are identified by their standard one- or three-letter residue
names. Modified nucleotides and HETATM residues are handled gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from Bio.PDB import PDBParser, MMCIFParser
from pathlib import Path


RNA_RESIDUE_NAMES = {"A", "U", "G", "C", "ADE", "URA", "GUA", "CYT"}
DNA_RESIDUE_NAMES = {"DA", "DT", "DG", "DC", "DI", "DU"}
PROTEIN_RESIDUE_NAMES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL",
}


@dataclass
class ParsedComplex:
    structure: object          # Bio.PDB Structure
    protein_chains: list       # list of Bio.PDB Chain objects
    rna_chains: list           # list of Bio.PDB Chain objects
    pdb_id: str


def parse_complex(path: str, model_rank: int = 0) -> ParsedComplex:
    """
    Loads a PDB, CIF, or AF3 Server zip and separates protein from RNA chains.

    Identifies chain type by majority residue class — avoids misclassifying
    chains that contain a handful of modified residues.
    """
    from .af3_io import resolve_structure_path

    resolved = resolve_structure_path(path, model_rank=model_rank)
    structure = _load_structure(str(resolved), model_rank=model_rank)
    model = structure[0]

    protein_chains = []
    rna_chains = []

    for chain in model:
        residues = list(chain.get_residues())
        if not residues:
            continue

        rna_count = sum(
            1 for r in residues if r.resname.strip().upper() in RNA_RESIDUE_NAMES
        )
        protein_count = sum(
            1 for r in residues if r.resname.strip().upper() in PROTEIN_RESIDUE_NAMES
        )

        if rna_count > protein_count and rna_count > 0:
            rna_chains.append(chain)
        elif protein_count > 0:
            protein_chains.append(chain)

    stem = Path(resolved).stem
    return ParsedComplex(
        structure=structure,
        protein_chains=protein_chains,
        rna_chains=rna_chains,
        pdb_id=stem,
    )


_PROTEIN_1LETTER = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def get_protein_sequence(chain) -> str:
    """Returns one-letter protein sequence for a chain."""
    seq = []
    for r in chain.get_residues():
        if r.id[0] != " ":
            continue
        name = r.resname.strip().upper()
        if name in _PROTEIN_1LETTER:
            seq.append(_PROTEIN_1LETTER[name])
    return "".join(seq)


def get_rna_sequence(chain) -> str:
    """Returns one-letter RNA sequence for a chain."""
    mapping = {
        "A": "A", "ADE": "A",
        "U": "U", "URA": "U",
        "G": "G", "GUA": "G",
        "C": "C", "CYT": "C",
    }
    seq = []
    for r in chain.get_residues():
        if r.id[0] != " ":
            continue
        name = r.resname.strip().upper()
        if name in mapping:
            seq.append(mapping[name])
    return "".join(seq)


def get_all_atoms(chains: list, include_hetero: bool = False) -> list:
    """Flattens a list of chains into atoms, skipping waters/hetero by default."""
    atoms = []
    for chain in chains:
        for residue in chain:
            if not include_hetero and residue.id[0] != " ":
                continue
            for atom in residue:
                atoms.append(atom)
    return atoms


def _load_structure(path: str, model_rank: int = 0):
    path_obj = Path(path)

    # Mislabeled zip (e.g. old clients passing .zip without extraction)
    with open(path, "rb") as fh:
        magic = fh.read(4)
    if magic[:2] == b"PK":
        from .af3_io import extract_af3_zip
        resolved = extract_af3_zip(path_obj, model_rank=model_rank)
        return _load_structure(str(resolved), model_rank=model_rank)

    suffix = path_obj.suffix.lower()
    is_mmcif = suffix in (".cif", ".mmcif")
    if not is_mmcif:
        with open(path, "rb") as fh:
            start = fh.read(64).lstrip()
        if start.startswith(b"data_"):
            is_mmcif = True

    if is_mmcif:
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    # AF3 mmCIF is UTF-8; Windows defaults to cp1252 and raises UnicodeDecodeError.
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return parser.get_structure("complex", handle)
