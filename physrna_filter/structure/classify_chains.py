"""Modality-aware chain classification for PhysComplex.

This parser is intentionally conservative. It identifies what is present in a
coordinate file; it does not decide whether a ligand is biologically relevant
or whether two chains genuinely bind.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .af3_io import resolve_structure_path
from .parse_complex import DNA_RESIDUE_NAMES, PROTEIN_RESIDUE_NAMES, RNA_RESIDUE_NAMES, _load_structure
from ..physcomplex.contracts import Modality


class ChainKind(StrEnum):
    PROTEIN = "protein"
    RNA = "rna"
    DNA = "dna"
    LIGAND = "ligand"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ClassifiedChain:
    chain_id: str
    kind: ChainKind
    residue_count: int
    heavy_atom_count: int


@dataclass(frozen=True)
class ClassifiedComplex:
    structure_path: str
    chains: tuple[ClassifiedChain, ...]
    modality: Modality | None
    partner_chain_ids: tuple[str, str] | None


_WATER = frozenset({"HOH", "WAT", "DOD"})


def _kind(chain) -> tuple[ChainKind, int, int]:
    residues = list(chain.get_residues())
    polymer = [r for r in residues if r.id[0] == " "]
    names = [r.resname.strip().upper() for r in polymer]
    protein = sum(name in PROTEIN_RESIDUE_NAMES for name in names)
    rna = sum(name in RNA_RESIDUE_NAMES for name in names)
    dna = sum(name in DNA_RESIDUE_NAMES for name in names)
    heavy = sum(
        1 for residue in residues for atom in residue
        if getattr(atom, "element", "").upper() != "H"
    )
    if protein and protein >= max(rna, dna):
        return ChainKind.PROTEIN, len(polymer), heavy
    if rna and rna >= dna:
        return ChainKind.RNA, len(polymer), heavy
    if dna:
        return ChainKind.DNA, len(polymer), heavy
    hetero = [
        residue for residue in residues
        if residue.id[0] != " " and residue.resname.strip().upper() not in _WATER
    ]
    if hetero:
        return ChainKind.LIGAND, len(hetero), heavy
    return ChainKind.UNKNOWN, len(residues), heavy


def infer_modality(chains: tuple[ClassifiedChain, ...]) -> Modality | None:
    """Infer only unambiguous pair types; mixed assemblies remain unclassified."""
    kinds = [chain.kind for chain in chains]
    if ChainKind.PROTEIN in kinds and ChainKind.RNA in kinds:
        return Modality.PROTEIN_RNA
    if ChainKind.PROTEIN in kinds and ChainKind.DNA in kinds:
        return Modality.PROTEIN_DNA
    if kinds.count(ChainKind.PROTEIN) >= 2:
        return Modality.PROTEIN_PROTEIN
    if ChainKind.PROTEIN in kinds and ChainKind.LIGAND in kinds:
        return Modality.PROTEIN_LIGAND
    if kinds.count(ChainKind.RNA) >= 2:
        return Modality.RNA_RNA
    if ChainKind.RNA in kinds and ChainKind.LIGAND in kinds:
        return Modality.RNA_LIGAND
    return None


def _closest_partner_pair(model, chains: tuple[ClassifiedChain, ...]) -> tuple[str, str] | None:
    coords: dict[str, np.ndarray] = {}
    for chain in model:
        xyz = [
            atom.coord for residue in chain for atom in residue
            if getattr(atom, "element", "").upper() != "H"
        ]
        if xyz:
            coords[chain.id] = np.asarray(xyz, dtype=float)
    best: tuple[float, tuple[str, str]] | None = None
    for i, left in enumerate(chains):
        if left.chain_id not in coords:
            continue
        tree = cKDTree(coords[left.chain_id])
        for right in chains[i + 1:]:
            if right.chain_id not in coords or left.kind == right.kind == ChainKind.UNKNOWN:
                continue
            distance = float(np.min(tree.query(coords[right.chain_id], k=1)[0]))
            if best is None or distance < best[0]:
                best = (distance, (left.chain_id, right.chain_id))
    return best[1] if best else None


def classify_complex(path: str | Path, *, model_rank: int = 0) -> ClassifiedComplex:
    """Classify chains in PDB/mmCIF/AF3 ZIP without changing PhysRNA parsing."""
    resolved = resolve_structure_path(path, model_rank=model_rank)
    structure = _load_structure(str(resolved), model_rank=model_rank)
    model = structure[0]
    chains = tuple(
        ClassifiedChain(
            chain_id=chain.id,
            kind=_kind(chain)[0],
            residue_count=_kind(chain)[1],
            heavy_atom_count=_kind(chain)[2],
        )
        for chain in model
        if list(chain.get_residues())
    )
    return ClassifiedComplex(
        structure_path=str(Path(resolved).resolve()),
        chains=chains,
        modality=infer_modality(chains),
        partner_chain_ids=_closest_partner_pair(model, chains),
    )
