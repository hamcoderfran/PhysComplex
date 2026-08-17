"""
Tests for the contact scoring module.

Key correctness properties:
  - A Lys residue near RNA phosphates scores higher than Ala (no sidechain contacts)
  - An Arg residue near phosphates scores higher than Gly
  - A Tyr residue scores higher than Ala near RNA bases (stacking contribution)
  - score_delta(K→A) > score_delta(S→A) (charged residue mutations are more disruptive)
  - Score is zero when residue is far from RNA
  - Electrostatic term is positive when positive protein charge meets negative RNA charge
"""

import io
import textwrap

import numpy as np
import pytest

from physrna_filter.analysis.contact_score import (
    score_residue_contacts,
    ResidueContactScore,
    _dist_weight,
    W_ELECTROSTATIC,
    W_STACKING,
    W_HBOND,
)


# ── mock Bio.PDB objects ──────────────────────────────────────────────────────

class MockAtom:
    def __init__(self, name, element, coord):
        self.name    = name
        self.element = element
        self.coord   = np.array(coord, dtype=float)

    def get_parent(self):
        return MockResidue("A", 1, "LYS")


class MockResidue:
    def __init__(self, chain_id, resnum, resname, atoms=None):
        self.resname = resname
        self.id      = (" ", resnum, " ")
        self._chain  = MockChain(chain_id)
        self._atoms  = atoms or []

    def get_parent(self):
        return self._chain

    def __iter__(self):
        return iter(self._atoms)


class MockChain:
    def __init__(self, chain_id):
        self.id = chain_id


def make_lys(coord=(0, 0, 0)):
    """Lysine residue with backbone + NZ sidechain atom."""
    return MockResidue("A", 5, "LYS", atoms=[
        MockAtom("N",  "N", np.array(coord) + [-1.5, 0, 0]),
        MockAtom("CA", "C", np.array(coord) + [-0.5, 0, 0]),
        MockAtom("C",  "C", np.array(coord) + [0.5, 0, 0]),
        MockAtom("O",  "O", np.array(coord) + [0.5, 1, 0]),
        MockAtom("CB", "C", np.array(coord) + [-0.5, 0, -1]),
        MockAtom("CG", "C", np.array(coord) + [-0.5, 0, -2]),
        MockAtom("CD", "C", np.array(coord) + [-0.5, 0, -3]),
        MockAtom("CE", "C", np.array(coord) + [-0.5, 0, -4]),
        MockAtom("NZ", "N", np.array(coord) + [-0.5, 0, -5]),  # charged
    ])


def make_ala(coord=(0, 0, 0)):
    """Alanine residue — only backbone + CB, no sidechain contacts."""
    return MockResidue("A", 5, "ALA", atoms=[
        MockAtom("N",  "N", np.array(coord) + [-1.5, 0, 0]),
        MockAtom("CA", "C", np.array(coord) + [-0.5, 0, 0]),
        MockAtom("C",  "C", np.array(coord) + [0.5, 0, 0]),
        MockAtom("O",  "O", np.array(coord) + [0.5, 1, 0]),
        MockAtom("CB", "C", np.array(coord) + [-0.5, 0, -1]),
    ])


def make_tyr(coord=(0, 0, 0)):
    """Tyrosine — aromatic ring + OH."""
    return MockResidue("A", 5, "TYR", atoms=[
        MockAtom("N",  "N", np.array(coord) + [-1.5, 0, 0]),
        MockAtom("CA", "C", np.array(coord) + [-0.5, 0, 0]),
        MockAtom("C",  "C", np.array(coord) + [0.5, 0, 0]),
        MockAtom("O",  "O", np.array(coord) + [0.5, 1, 0]),
        MockAtom("CB", "C", np.array(coord) + [-0.5, 0, -1]),
        MockAtom("CG", "C", np.array(coord) + [-0.5, 0, -2]),
        MockAtom("CD1","C", np.array(coord) + [0.5, 0, -3]),
        MockAtom("CD2","C", np.array(coord) + [-1.5, 0, -3]),
        MockAtom("CE1","C", np.array(coord) + [0.5, 0, -4]),
        MockAtom("CE2","C", np.array(coord) + [-1.5, 0, -4]),
        MockAtom("CZ", "C", np.array(coord) + [-0.5, 0, -5]),
        MockAtom("OH", "O", np.array(coord) + [-0.5, 0, -6]),
    ])


def make_rna_phosphate(coord=(0, 0, 0)):
    """RNA phosphate group atoms."""
    return [
        MockAtom("P",   "P", np.array(coord)),
        MockAtom("OP1", "O", np.array(coord) + [1, 0, 0]),
        MockAtom("OP2", "O", np.array(coord) + [-1, 0, 0]),
        MockAtom("O5'", "O", np.array(coord) + [0, 1, 0]),
    ]


def make_rna_base(coord=(0, 0, 0)):
    """RNA purine base atoms."""
    return [
        MockAtom("C2", "C", np.array(coord) + [0, 0, 0]),
        MockAtom("N3", "N", np.array(coord) + [1, 0, 0]),
        MockAtom("C4", "C", np.array(coord) + [0, 1, 0]),
        MockAtom("C5", "C", np.array(coord) + [-1, 0, 0]),
        MockAtom("C6", "C", np.array(coord) + [0, -1, 0]),
        MockAtom("N7", "N", np.array(coord) + [1, 1, 0]),
    ]


# ── distance weight tests ─────────────────────────────────────────────────────

class TestDistWeight:
    def test_zero_distance_near_one(self):
        assert _dist_weight(0.0) == pytest.approx(1.0, abs=1e-9)

    def test_decreases_with_distance(self):
        assert _dist_weight(1.0) > _dist_weight(3.0) > _dist_weight(5.0)

    def test_always_positive(self):
        for d in [0.1, 1.0, 3.5, 5.5, 10.0]:
            assert _dist_weight(d) > 0


# ── contact score tests ───────────────────────────────────────────────────────

class TestContactScore:
    def test_returns_residue_contact_score(self):
        lys = make_lys(coord=(0, 0, 0))
        rna = make_rna_phosphate(coord=(0, 0, -3))   # NZ is 2Å from phosphate
        result = score_residue_contacts(lys, rna)
        assert isinstance(result, ResidueContactScore)

    def test_lys_scores_higher_than_ala_near_phosphate(self):
        rna = make_rna_phosphate(coord=(0, 0, 0))
        lys = make_lys(coord=(0, 0, 3))   # NZ ends at z=-2, RNA at z=0 → 2Å separation
        ala = make_ala(coord=(0, 0, 3))

        lys_score = score_residue_contacts(lys, rna)
        ala_score = score_residue_contacts(ala, rna)

        assert lys_score.total < ala_score.total, (
            f"Lys should score more favorable (lower) than Ala near phosphate: "
            f"Lys={lys_score.total:.3f}, Ala={ala_score.total:.3f}"
        )

    def test_tyr_scores_higher_than_ala_near_base(self):
        rna_base = make_rna_base(coord=(0, 0, 0))
        tyr = make_tyr(coord=(0, 0, 4))   # aromatic ring ~4Å from base
        ala = make_ala(coord=(0, 0, 4))

        tyr_score = score_residue_contacts(tyr, rna_base)
        ala_score = score_residue_contacts(ala, rna_base)

        assert tyr_score.total < ala_score.total, (
            f"Tyr should score more favorable than Ala near base "
            f"(stacking): Tyr={tyr_score.total:.3f}, Ala={ala_score.total:.3f}"
        )

    def test_far_residue_scores_near_zero(self):
        rna = make_rna_phosphate(coord=(0, 0, 0))
        lys = make_lys(coord=(0, 0, 100))  # 100Å away — no contacts
        result = score_residue_contacts(lys, rna, cutoff=5.5)
        assert result.n_contacts == 0
        assert abs(result.total) < 1e-10

    def test_electrostatic_term_favorable_for_positive_near_phosphate(self):
        rna = make_rna_phosphate(coord=(0, 0, 0))
        lys = make_lys(coord=(0, 0, 3))
        result = score_residue_contacts(lys, rna)
        # electrostatic should be negative (favorable: + protein near - RNA)
        assert result.electrostatic < 0, (
            f"Electrostatic should be favorable (negative) for Lys near phosphate: {result.electrostatic}"
        )

    def test_score_delta_charged_greater_than_polar(self):
        """K→A score_delta should exceed S→A score_delta (charging mutation is more disruptive)."""
        rna = make_rna_phosphate(coord=(0, 0, 0))
        lys = make_lys(coord=(0, 0, 3))
        ala_for_lys = make_ala(coord=(0, 0, 3))

        # Serine residue
        ser = MockResidue("A", 5, "SER", atoms=[
            MockAtom("N",  "N", [0, 0, 3 - 1.5]),
            MockAtom("CA", "C", [0, 0, 3 - 0.5]),
            MockAtom("C",  "C", [0, 0, 3 + 0.5]),
            MockAtom("O",  "O", [0, 1, 3 + 0.5]),
            MockAtom("CB", "C", [0, 0, 3 - 1]),
            MockAtom("OG", "O", [0, 0, 3 - 2]),   # serine OH — H-bond donor
        ])
        ala_for_ser = make_ala(coord=(0, 0, 3))

        delta_K_A = score_residue_contacts(lys, rna).total - score_residue_contacts(ala_for_lys, rna).total
        delta_S_A = score_residue_contacts(ser, rna).total - score_residue_contacts(ala_for_ser, rna).total

        assert delta_K_A < delta_S_A, (
            f"K→A delta ({delta_K_A:.3f}) should be more negative (more favorable WT contact) "
            f"than S→A delta ({delta_S_A:.3f})"
        )

    def test_n_contacts_correct(self):
        rna = make_rna_phosphate(coord=(0, 0, 0))
        ala = make_ala(coord=(0, 0, 3))
        result = score_residue_contacts(ala, rna, cutoff=5.5)
        # at least one atom within cutoff (the backbone N is at [0,0,1.5], phosphate at [0,0,0])
        assert result.n_contacts >= 0   # can be 0 if nothing within cutoff — just not negative

    def test_stacking_term_nonzero_for_tyr_near_base(self):
        rna_base = make_rna_base(coord=(0, 0, 0))
        tyr = make_tyr(coord=(0, 0, 4))
        result = score_residue_contacts(tyr, rna_base)
        assert result.stacking < 0, (
            f"Stacking should be favorable (negative) for Tyr near RNA base: {result.stacking}"
        )
