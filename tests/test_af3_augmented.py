"""
Tests for the augmented AF3 filter: PhysGT inference, clash detection,
contact aggregation, and multi-branch scoring.
"""

from __future__ import annotations

import numpy as np
import pytest

from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue

from physrna_filter.analysis.biological_plausibility import (
    assess_biological_plausibility,
    score_binding_motifs,
)
from physrna_filter.analysis.gt_inference import gt_verdict, physics_only_interface_score
from physrna_filter.analysis.graph_features import InterfaceGraph
from physrna_filter.analysis.score import (
    _combine_verdicts,
    _combine_verdicts_af3,
    _rmsd_verdict_af3,
    contact_plausibility_score,
    clash_plausibility_score,
    FilterResult,
    format_report,
)
from physrna_filter.structure.clash_detection import detect_interface_clashes


def _atom(name: str, coord, element: str) -> Atom:
    return Atom(
        name, np.array(coord, dtype=float), 1.0, 1.0, " ",
        name, 1, element=element,
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


class TestBipartiteGT:
    def test_physics_informed_gt_has_cross_attention(self):
        torch = pytest.importorskip("torch")
        tg = pytest.importorskip("torch_geometric")
        del tg  # noqa: F841

        from physrna_filter.analysis.gt_model import PhysicsInformedGT
        from physrna_filter.analysis.gt_constants import EDGE_DIM

        model = PhysicsInformedGT(
            protein_node_dim=24, rna_node_dim=8, hidden_dim=32, n_heads=4, n_layers=2,
            edge_dim=EDGE_DIM,
        )
        assert hasattr(model, "cross_attn")
        assert hasattr(model, "score_interface")

        x_prot = torch.randn(3, 24)
        x_rna  = torch.randn(2, 8)
        edge_index = torch.tensor([[0, 3], [3, 0]], dtype=torch.long)
        edge_attr  = torch.randn(2, EDGE_DIM)

        score = model.score_interface(x_prot, x_rna, edge_index, edge_attr, n_prot=3)
        assert score.shape == torch.Size([])

    def test_fallback_mlp_interface_scoring(self):
        torch = pytest.importorskip("torch")
        from physrna_filter.analysis.gt_model import FallbackMLP

        model = FallbackMLP()
        x_prot = torch.zeros((2, 24))
        x_rna  = torch.zeros((1, 8))
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 9))

        score = model.score_interface(x_prot, x_rna, edge_index, edge_attr, n_prot=2)
        assert score.shape == torch.Size([])


class TestClashDetection:
    def test_no_clash_when_well_separated(self):
        protein = _chain("A", [
            _residue((" ", 1, " "), "ALA", [_atom("CA", [0, 0, 0], "C")]),
        ])
        rna = _chain("R", [
            _residue((" ", 1, " "), "A", [_atom("C4'", [5, 0, 0], "C")]),
        ])
        result = detect_interface_clashes([protein], [rna], [("R", 1)])
        assert result.verdict == "PASS"
        assert result.n_severe == 0

    def test_severe_clash_detected(self):
        protein = _chain("A", [
            _residue((" ", 1, " "), "ALA", [_atom("CA", [0, 0, 0], "C")]),
        ])
        rna = _chain("R", [
            _residue((" ", 1, " "), "A", [_atom("C4'", [1.0, 0, 0], "C")]),
        ])
        result = detect_interface_clashes([protein], [rna], [("R", 1)])
        assert result.n_severe >= 1
        assert result.verdict == "FAIL"

    def test_af3_mode_allows_normal_interface_contact(self):
        protein = _chain("A", [
            _residue((" ", 1, " "), "ALA", [_atom("CA", [0, 0, 0], "C")]),
        ])
        rna = _chain("R", [
            _residue((" ", 1, " "), "A", [_atom("C4'", [1.4, 0, 0], "C")]),
        ])
        result = detect_interface_clashes([protein], [rna], [("R", 1)], af3_mode=True)
        assert result.n_severe == 0
        assert result.verdict in ("PASS", "WARN")

    def test_af3_mode_flags_true_overlap(self):
        protein = _chain("A", [
            _residue((" ", 1, " "), "ALA", [_atom("CA", [0, 0, 0], "C")]),
        ])
        rna = _chain("R", [
            _residue((" ", 1, " "), "A", [_atom("C4'", [0.9, 0, 0], "C")]),
        ])
        result = detect_interface_clashes([protein], [rna], [("R", 1)], af3_mode=True)
        assert result.verdict == "FAIL"


class TestContactAggregation:
    def test_interface_with_contacts_passes_or_warns(self):
        protein = _chain("A", [
            _residue((" ", 1, " "), "LYS", [
                _atom("CA", [0, 0, 0], "C"),
                _atom("NZ", [1.5, 0, 0], "N"),
            ]),
        ])
        rna = _chain("R", [
            _residue((" ", 1, " "), "A", [
                _atom("C4'", [3, 0, 0], "C"),
                _atom("OP1", [2.5, 0.5, 0], "O"),
            ]),
        ])
        energy, n_res, n_contacts, verdict = contact_plausibility_score(
            [protein], [rna]
        )
        assert n_res >= 1
        assert verdict in ("PASS", "WARN", "FAIL")


class TestBiologicalPlausibility:
    def test_u1a_motif_detected(self):
        score, hits = score_binding_motifs("AAUUUUGCAC", rbp_name="U1A")
        assert score >= 1
        assert hits

    def test_no_metadata_returns_unknown(self):
        result = assess_biological_plausibility()
        assert result.verdict == "UNKNOWN"

    def test_partner_mismatch_fails(self):
        result = assess_biological_plausibility(
            rna_sequence="CCGGGGGAUCACCACGG",
            rbp_name="U1A",
        )
        assert result.verdict == "FAIL"
        assert any("MS2" in n for n in result.notes)

    def test_nova2_positive_not_mismatch_on_cac(self):
        result = assess_biological_plausibility(
            rna_sequence="GAGGACCUAGAUCACCCCUC",
            rbp_name="NOVA2",
        )
        assert result.verdict != "FAIL"
        assert not any("MS2" in n for n in result.notes)

    def test_wrong_are_with_u1a_fails(self):
        result = assess_biological_plausibility(
            rna_sequence="UUUUAUUUU",
            rbp_name="U1A",
        )
        assert result.verdict == "FAIL"
        assert any("HUD" in n for n in result.notes)


class TestGTInference:
    def test_gt_verdict_thresholds(self):
        from physrna_filter.analysis.thresholds import load_thresholds
        t = load_thresholds()
        assert gt_verdict(t.get("gt_pass", -2.0) - 1.0) == "PASS"
        assert gt_verdict(t.get("gt_warn", 0.5) - 0.1) == "WARN"
        assert gt_verdict(t.get("gt_warn", 0.5) + 2.0) == "FAIL"

    def test_physics_only_no_edges_is_implausible(self):
        torch = pytest.importorskip("torch")
        graph = InterfaceGraph(
            x_protein=torch.zeros(1, 24),
            x_rna=torch.zeros(1, 8),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 9)),
            node_types=torch.zeros(2, dtype=torch.long),
            mutation_node_idx=0,
        )
        score = physics_only_interface_score(graph)
        assert score > 0


class TestMultiBranchScoring:
    def test_combine_verdicts_takes_worst(self):
        assert _combine_verdicts("PASS", "FAIL") == "FAIL"
        assert _combine_verdicts("PASS", "WARN", "PASS") == "WARN"

    def test_combine_verdicts_af3_primary_branches_only(self):
        assert _combine_verdicts_af3("PASS", "PASS", "PASS") == "PASS"
        assert _combine_verdicts_af3("FAIL", "PASS", "PASS") == "FAIL"
        assert _combine_verdicts_af3("PASS", "FAIL", "PASS") == "FAIL"
        assert _combine_verdicts_af3("PASS", "PASS", "FAIL") == "FAIL"
        assert _combine_verdicts_af3("WARN", "PASS", "PASS") == "WARN"
        assert _combine_verdicts_af3("PASS", "WARN", "PASS") == "PASS"
        assert _combine_verdicts_af3("PASS", "UNKNOWN", "PASS") == "PASS"
        assert _combine_verdicts_af3("UNKNOWN", "UNKNOWN", "UNKNOWN") == "WARN"
        assert _combine_verdicts_af3(
            "UNKNOWN", "UNKNOWN", "UNKNOWN", gt_physics_only=True
        ) == "WARN"
        assert _combine_verdicts_af3(
            "UNKNOWN", "PASS", "UNKNOWN", require_trained_gt=True
        ) == "FAIL"

    def test_rmsd_verdict_af3_relaxed(self):
        assert _rmsd_verdict_af3(10.0) == "PASS"
        assert _rmsd_verdict_af3(22.0) == "WARN"
        assert _rmsd_verdict_af3(35.0) == "FAIL"

    def test_format_report_includes_new_branches(self):
        result = FilterResult(
            rmsd_score=1.5, rmsd_nearest_cluster=0, rmsd_verdict="PASS",
            geom_score=25.0, geom_max_score=40.0, geom_nearest_cluster=1,
            geom_verdict="PASS",
            contact_energy=-5.0, contact_n_residues=3, contact_n_contacts=8,
            contact_verdict="PASS",
            clash_n_severe=0, clash_n_moderate=0, clash_worst_distance=3.5,
            clash_verdict="PASS",
            gt_score=-3.0, gt_verdict="PASS", gt_physics_only=True,
            bio_verdict="UNKNOWN",
            combined_verdict="PASS", confidence=0.8,
        )
        report = format_report(result, [("R", 1)])
        assert "Contact Physics" in report
        assert "PhysGT" in report
        assert "Clash" in report
        assert "Hallucination verdict" in report
