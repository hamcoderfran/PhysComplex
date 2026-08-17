"""
Coordinate-aware decoy generation for PhysGT interface contrastive training.

Decoys are built by rigid-body translation of RNA coordinates in a temporary
PDB, then rebuilding the full interface graph so all extended physics edge
features reflect the perturbed geometry (not merely zeroing contact terms).
"""
from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from Bio.PDB import PDBIO

from .graph_features import InterfaceGraph, build_af3_interface_graph
from .gt_constants import EDGE_DIM


def _translate_rna_in_pdb(
    pdb_path: str,
    output_path: str,
    shift_angstrom: float,
    seed: int,
) -> None:
    from ..structure.parse_complex import RNA_RESIDUE_NAMES, parse_complex

    rng = np.random.default_rng(seed)
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction) + 1e-8
    shift = direction * shift_angstrom

    parsed = parse_complex(pdb_path)
    structure = parsed.structure

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.resname.strip().upper() not in RNA_RESIDUE_NAMES:
                    continue
                for atom in residue:
                    atom.coord = atom.coord + shift

    io = PDBIO()
    io.set_structure(structure)
    io.save(output_path)


def make_entropic_decoy_from_pdb(
    pdb_path: str,
    pdb_id: str,
    *,
    shift_angstrom: float = 10.0,
    seed: int = 0,
    esm_embeddings: dict | None = None,
    rnafm_embeddings: dict | None = None,
    esm_dim: int = 320,
    rnafm_dim: int = 640,
) -> InterfaceGraph:
    """Rigid-body RNA shift → rebuild graph with full physics features."""
    with tempfile.TemporaryDirectory(prefix="physgt_decoy_") as tmp:
        decoy_pdb = os.path.join(tmp, "decoy.pdb")
        _translate_rna_in_pdb(pdb_path, decoy_pdb, shift_angstrom, seed)
        return build_af3_interface_graph(
            decoy_pdb,
            pdb_id=f"{pdb_id}_decoy_{seed}",
            esm_embeddings=esm_embeddings,
            rnafm_embeddings=rnafm_embeddings,
            esm_dim=esm_dim,
            rnafm_dim=rnafm_dim,
        )


def make_sequence_decoy_graph(
    positive: InterfaceGraph,
    seed: int = 0,
) -> InterfaceGraph:
    """
    Sequence–structure decoy: permute RNA node features while keeping
    coordinates fixed (wrong base identity at each site).
    """
    rng = random.Random(seed)
    n_rna = positive.x_rna.shape[0]
    if n_rna <= 1:
        return make_entropic_decoy_graph_legacy(positive, seed=seed)

    perm = list(range(n_rna))
    rng.shuffle(perm)
    while perm == list(range(n_rna)) and n_rna > 1:
        rng.shuffle(perm)

    decoy = InterfaceGraph(
        x_protein=positive.x_protein.clone(),
        x_rna=positive.x_rna[perm].clone(),
        edge_index=positive.edge_index.clone(),
        edge_attr=positive.edge_attr.clone(),
        node_types=positive.node_types.clone(),
        mutation_node_idx=positive.mutation_node_idx,
        protein_residues=list(positive.protein_residues),
        rna_residues=[positive.rna_residues[i] for i in perm],
        prot_coords=positive.prot_coords.clone() if positive.prot_coords is not None else None,
        rna_coords=positive.rna_coords.clone() if positive.rna_coords is not None else None,
    )
    return decoy


def make_entropic_decoy_graph_legacy(
    positive: InterfaceGraph,
    shift_angstrom: float = 10.0,
    seed: int = 0,
) -> InterfaceGraph:
    """
    Fast fallback when PDB path is unavailable: zero prot–RNA edge physics.
    Prefer ``make_entropic_decoy_from_pdb`` when a structure file exists.
    """
    decoy = InterfaceGraph(
        x_protein=positive.x_protein.clone(),
        x_rna=positive.x_rna.clone(),
        edge_index=positive.edge_index.clone(),
        edge_attr=positive.edge_attr.clone(),
        node_types=positive.node_types.clone(),
        mutation_node_idx=positive.mutation_node_idx,
        protein_residues=list(positive.protein_residues),
        rna_residues=list(positive.rna_residues),
        prot_coords=positive.prot_coords.clone() if positive.prot_coords is not None else None,
        rna_coords=positive.rna_coords.clone() if positive.rna_coords is not None else None,
    )

    if decoy.edge_attr.numel() > 0:
        prot_rna = decoy.edge_attr[:, 6] > 0.5
        decoy.edge_attr[prot_rna, 2:6] = 0.0
        if decoy.edge_attr.shape[1] >= EDGE_DIM:
            decoy.edge_attr[prot_rna, 9:12] = 0.0
        decoy.edge_attr[prot_rna, 1] = 0.01

    return decoy


def make_entropic_decoy_graph(
    positive: InterfaceGraph,
    shift_angstrom: float = 10.0,
    seed: int = 0,
    pdb_path: str | None = None,
    pdb_id: str = "decoy",
    esm_embeddings: dict | None = None,
    rnafm_embeddings: dict | None = None,
    esm_dim: int = 320,
    rnafm_dim: int = 640,
) -> InterfaceGraph:
    """Dispatch to coordinate-aware or legacy decoy builder."""
    if pdb_path and os.path.exists(pdb_path):
        try:
            return make_entropic_decoy_from_pdb(
                pdb_path,
                pdb_id,
                shift_angstrom=shift_angstrom,
                seed=seed,
                esm_embeddings=esm_embeddings,
                rnafm_embeddings=rnafm_embeddings,
                esm_dim=esm_dim,
                rnafm_dim=rnafm_dim,
            )
        except Exception:
            pass
    return make_entropic_decoy_graph_legacy(positive, shift_angstrom, seed)
