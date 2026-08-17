"""
Tests for filtering waters/hetero residues out of RNA pipeline inputs.
"""

from __future__ import annotations

import numpy as np

from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue

from physrna_filter.structure.extract_interface import find_interface_residues
from physrna_filter.structure.parse_complex import get_all_atoms, get_rna_sequence


def _atom(name: str, coord, element: str) -> Atom:
    return Atom(
        name,
        np.array(coord, dtype=float),
        1.0,
        1.0,
        " ",
        name,
        1,
        element=element,
    )


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


def test_get_rna_sequence_skips_waters_and_unknowns():
    chain = _chain(
        "R",
        [
            _residue((" ", 1, " "), "A", [_atom("C4'", [0, 0, 0], "C")]),
            _residue(("W", 2, " "), "HOH", [_atom("O", [1, 0, 0], "O")]),
            _residue((" ", 3, " "), "MSE", [_atom("C4'", [2, 0, 0], "C")]),
            _residue((" ", 4, " "), "G", [_atom("C4'", [3, 0, 0], "C")]),
        ],
    )

    assert get_rna_sequence(chain) == "AG"


def test_get_all_atoms_skips_hetero_by_default():
    chain = _chain(
        "R",
        [
            _residue((" ", 1, " "), "A", [_atom("C4'", [0, 0, 0], "C")]),
            _residue(("W", 2, " "), "HOH", [_atom("O", [1, 0, 0], "O")]),
        ],
    )

    assert [atom.name for atom in get_all_atoms([chain])] == ["C4'"]
    assert [atom.name for atom in get_all_atoms([chain], include_hetero=True)] == [
        "C4'",
        "O",
    ]


def test_find_interface_residues_ignores_waters_and_anchorless_residues():
    protein = _chain(
        "A",
        [_residue((" ", 1, " "), "ALA", [_atom("CA", [0, 0, 0], "C")])],
    )
    rna = _chain(
        "R",
        [
            _residue((" ", 1, " "), "A", [_atom("C4'", [1, 0, 0], "C")]),
            _residue(("W", 2, " "), "HOH", [_atom("O", [1, 0, 0], "O")]),
            _residue((" ", 3, " "), "C", [_atom("O3'", [1, 0, 0], "O")]),
        ],
    )

    assert find_interface_residues([protein], [rna], cutoff=5.0) == [("R", 1)]
