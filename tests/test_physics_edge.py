"""Tests for extended physics edge features."""
import torch

from physrna_filter.analysis.gt_constants import EDGE_DIM, LEGACY_EDGE_DIM, PHYSICS_SUMMARY_DIM
from physrna_filter.analysis.physics_edge import (
    build_edge_feature_vector,
    coerce_edge_attr,
    physics_summary_from_graph,
)


def test_edge_feature_vector_dim():
    feat = build_edge_feature_vector(3.5, -1.0, -0.5, -0.2, -0.1, -0.3, -0.1, -0.4)
    assert feat.shape[0] == EDGE_DIM
    assert feat[6] == 1.0  # prot-rna edge type


def test_edge_feature_vector_legacy_dim():
    feat = build_edge_feature_vector(
        3.5, -1.0, -0.5, -0.2, -0.1, edge_dim=LEGACY_EDGE_DIM
    )
    assert feat.shape[0] == LEGACY_EDGE_DIM
    assert feat[6] == 1.0


def test_coerce_edge_attr_truncates():
    full = torch.randn(4, EDGE_DIM)
    legacy = coerce_edge_attr(full, LEGACY_EDGE_DIM)
    assert legacy.shape == (4, LEGACY_EDGE_DIM)
    assert torch.allclose(legacy, full[:, :LEGACY_EDGE_DIM])


def test_legacy_checkpoint_model_accepts_legacy_edges():
    torch_geometric = __import__("pytest").importorskip("torch_geometric")
    del torch_geometric

    from physrna_filter.analysis.gt_model import PhysicsInformedGT

    model = PhysicsInformedGT(
        protein_node_dim=24,
        rna_node_dim=8,
        hidden_dim=32,
        n_heads=4,
        n_layers=2,
        edge_dim=LEGACY_EDGE_DIM,
        physics_summary_dim=11,
    )
    x_prot = torch.randn(3, 24)
    x_rna = torch.randn(2, 8)
    edge_index = torch.tensor([[0, 3], [3, 0]], dtype=torch.long)
    edge_attr = build_edge_feature_vector(
        3.5, -1.0, -0.5, -0.2, -0.1, edge_dim=LEGACY_EDGE_DIM
    ).unsqueeze(0).expand(2, -1)

    score = model.score_interface(x_prot, x_rna, edge_index, edge_attr, n_prot=3)
    assert score.shape == torch.Size([])


def test_physics_summary_extended_dim():
    x_prot = torch.randn(5, 344)
    edge_attr = torch.randn(8, EDGE_DIM)
    edge_attr[:, 6] = 1.0
    summary = physics_summary_from_graph(x_prot, edge_attr, mutation_idx=2)
    assert summary.shape[0] == PHYSICS_SUMMARY_DIM


def test_physics_summary_legacy_dim():
    x_prot = torch.randn(3, 344)
    edge_attr = torch.randn(4, 9)
    edge_attr[:, 6] = 1.0
    summary = physics_summary_from_graph(x_prot, edge_attr, legacy_dim=11)
    assert summary.shape[0] == 11
