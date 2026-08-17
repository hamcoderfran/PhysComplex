"""Tests for interface-head training dimension consistency."""

from __future__ import annotations

import torch

from physrna_filter.analysis.graph_features import InterfaceGraph
from physrna_filter.validation.train_interface_head import _validate_graph_dims


def _dummy_graph(protein_dim: int, rna_dim: int) -> InterfaceGraph:
    return InterfaceGraph(
        x_protein=torch.zeros(3, protein_dim),
        x_rna=torch.zeros(2, rna_dim),
        edge_index=torch.tensor([[0], [3]], dtype=torch.long),
        edge_attr=torch.zeros(1, 9),
        node_types=torch.tensor([0, 0, 0, 1, 1]),
        mutation_node_idx=0,
        protein_residues=[],
        rna_residues=[],
    )


def test_validate_graph_dims_accepts_uniform_graphs():
    graphs = [_dummy_graph(344, 648), _dummy_graph(344, 648)]
    p_dim, r_dim = _validate_graph_dims(graphs)
    assert p_dim == 344
    assert r_dim == 648


def test_validate_graph_dims_rejects_mixed_rna_widths():
    graphs = [_dummy_graph(344, 28), _dummy_graph(344, 648)]
    try:
        _validate_graph_dims(graphs)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "Inconsistent graph feature dims" in str(e)
