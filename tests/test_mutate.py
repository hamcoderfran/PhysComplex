"""
Tests for mutation structure helpers.
"""

from __future__ import annotations

import pytest

from physrna_filter.structure.mutate import find_mutation_chain


def _write_minimal_pdb(path, resname: str = "SER") -> None:
    path.write_text(
        "\n".join(
            [
                "ATOM      1  N   %s A 449      11.104  13.207  12.447  1.00 20.00           N" % resname,
                "ATOM      2  CA  %s A 449      12.560  13.207  12.447  1.00 20.00           C" % resname,
                "ATOM      3  C   %s A 449      13.104  14.607  12.447  1.00 20.00           C" % resname,
                "ATOM      4  O   %s A 449      12.504  15.607  12.447  1.00 20.00           O" % resname,
                "TER",
                "END",
                "",
            ]
        )
    )


def test_find_mutation_chain_accepts_engineered_mutant_identity(tmp_path):
    pdb_path = tmp_path / "engineered_mutant.pdb"
    _write_minimal_pdb(pdb_path, resname="SER")

    assert find_mutation_chain(str(pdb_path), 449, "C", alt_aa="S") == "A"


def test_find_mutation_chain_still_rejects_wrong_identity(tmp_path):
    pdb_path = tmp_path / "wrong_identity.pdb"
    _write_minimal_pdb(pdb_path, resname="SER")

    with pytest.raises(ValueError, match="No chain"):
        find_mutation_chain(str(pdb_path), 449, "C")


def test_find_mutation_chain_reads_mmcif(tmp_path):
    from physrna_filter.structure.mutate import _mutate_simple

    cif_path = tmp_path / "engineered.cif"
    cif_path.write_text(
        "\n".join(
            [
                "data_test",
                "loop_",
                "_atom_site.group_PDB",
                "_atom_site.id",
                "_atom_site.type_symbol",
                "_atom_site.label_atom_id",
                "_atom_site.label_alt_id",
                "_atom_site.label_comp_id",
                "_atom_site.label_asym_id",
                "_atom_site.label_entity_id",
                "_atom_site.label_seq_id",
                "_atom_site.pdbx_PDB_ins_code",
                "_atom_site.Cartn_x",
                "_atom_site.Cartn_y",
                "_atom_site.Cartn_z",
                "_atom_site.occupancy",
                "_atom_site.B_iso_or_equiv",
                "_atom_site.pdbx_formal_charge",
                "_atom_site.auth_seq_id",
                "_atom_site.auth_comp_id",
                "_atom_site.auth_asym_id",
                "_atom_site.auth_atom_id",
                "_atom_site.pdbx_PDB_model_num",
                "ATOM 1 N N . SER A 1 449 ? 11.104 13.207 12.447 1.00 20.00 ? 449 SER A N 1",
                "ATOM 2 C CA . SER A 1 449 ? 12.560 13.207 12.447 1.00 20.00 ? 449 SER A CA 1",
                "ATOM 3 C C . SER A 1 449 ? 13.104 14.607 12.447 1.00 20.00 ? 449 SER A C 1",
                "ATOM 4 O O . SER A 1 449 ? 12.504 15.607 12.447 1.00 20.00 ? 449 SER A O 1",
                "#",
            ]
        ),
        encoding="utf-8",
    )
    assert find_mutation_chain(str(cif_path), 449, "S", alt_aa="A") == "A"
    out = _mutate_simple(str(cif_path), "A", 449, "A", str(tmp_path / "mut.pdb"))
    assert out.endswith(".cif")
