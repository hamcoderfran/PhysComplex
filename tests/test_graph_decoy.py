"""Tests for coordinate-aware decoy generation."""
import torch

from physrna_filter.analysis.graph_features import InterfaceGraph
from physrna_filter.analysis.graph_decoy import (
    make_sequence_decoy_graph,
    make_entropic_decoy_graph_legacy,
)
from physrna_filter.analysis.gt_constants import EDGE_DIM


def _dummy_graph(n_rna: int = 4) -> InterfaceGraph:
    n_prot = 3
    return InterfaceGraph(
        x_protein=torch.randn(n_prot, 24),
        x_rna=torch.randn(n_rna, 8),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.randn(2, EDGE_DIM),
        node_types=torch.tensor([0, 1], dtype=torch.long),
        mutation_node_idx=0,
        protein_residues=[("A", i, "A") for i in range(1, n_prot + 1)],
        rna_residues=[("B", i, "U") for i in range(1, n_rna + 1)],
    )


def test_sequence_decoy_permutes_rna_features():
    g = _dummy_graph()
    decoy = make_sequence_decoy_graph(g, seed=42)
    assert decoy.x_rna.shape == g.x_rna.shape
    assert not torch.allclose(decoy.x_rna, g.x_rna)


def test_legacy_entropic_decoy_zeros_prot_rna_physics():
    g = _dummy_graph()
    g.edge_attr[:, 6] = 1.0
    g.edge_attr[:, 2:6] = 1.0
    decoy = make_entropic_decoy_graph_legacy(g, seed=0)
    prot_rna = decoy.edge_attr[:, 6] > 0.5
    assert torch.all(decoy.edge_attr[prot_rna, 2:6] == 0.0)
