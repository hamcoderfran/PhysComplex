"""Tests for interface residue extraction and trajectory index mapping."""

from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue

import numpy as np

from physrna_filter.structure.extract_interface import rna_residue_to_trajectory_indices


def _atom(name: str, coord) -> Atom:
    return Atom(name, np.array(coord, dtype=float), 1.0, 1.0, " ", name, 1, element="C")


def _residue(res_id, resname: str, atoms: list[Atom]) -> Residue:
    residue = Residue(res_id, resname, " ")
    for atom in atoms:
        residue.add(atom)
    return residue


def _chain(chain_id: str, residues: list[Residue]) -> Chain:
    chain = Chain(chain_id)
    for residue in residues:
        chain.add(residue)
    return chain


class TestTrajectoryIndexMapping:
    def test_maps_pdb_resnum_to_sequence_index(self):
        rna = _chain("R", [
            _residue((" ", 10, " "), "G", [_atom("C4'", [0, 0, 0])]),
            _residue((" ", 11, " "), "A", [_atom("C4'", [1, 0, 0])]),
            _residue((" ", 12, " "), "U", [_atom("C4'", [2, 0, 0])]),
        ])
        indices = rna_residue_to_trajectory_indices([rna], [("R", 11), ("R", 12)])
        assert indices == [1, 2]

    def test_skips_non_rna_residues(self):
        rna = _chain("R", [
            _residue((" ", 1, " "), "ALA", [_atom("CA", [0, 0, 0])]),
            _residue((" ", 2, " "), "A", [_atom("C4'", [1, 0, 0])]),
        ])
        indices = rna_residue_to_trajectory_indices([rna], [("R", 2)])
        assert indices == [0]
