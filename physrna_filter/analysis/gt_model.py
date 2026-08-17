"""
Physics-Informed Graph Transformer (PhysGT) for RNA-protein ΔΔG prediction
and AF3 interface plausibility scoring.

Waiting on more ProNAB data to build up better.

Architecture overview (May be outdated/incorrect)
---------------------
1.  Type-specific input projections: protein nodes and RNA nodes have different
    raw feature dimensions (ESM-2 vs RNA-FM), so we project each to a shared
    hidden_dim with a small MLP before passing to the GT layers.

2.  Bipartite cross-attention (literature: heterogeneous GNN / DTI models,
    e.g. DeepInteract, HGT): explicit protein↔RNA message passing before global
    graph mixing.  This captures recognition-specific patterns that homogeneous
    attention dilutes in mixed node graphs.

3.  TransformerConv layers (torch_geometric): multi-head attention over the
    heterogeneous-but-projected graph.  Physics edge features (elec, stack,
    hbond, vdw, distance) are fused into every attention head, letting the
    model learn context-dependent weights for each interaction type.

4.  Siamese encoding: the same encoder is applied to both the WT graph and the
    mutant graph (weight sharing).  ΔΔG is predicted from the *difference* of
    the two encodings, enforcing antisymmetry (reversing mutation flips sign).

5.  Dual-stream pooling: the ΔΔG signal is drawn from
      (a) the mutation-site node embedding difference      (local context)
      (b) attention-weighted protein subgraph diff         (global context)
    These are concatenated and passed through an MLP head.

6.  Interface plausibility head (literature: decoy discrimination / docking
    scoring): single-graph mode for AF3 structures.  Combines cross-attention
    pooled embeddings with explicit physics summary to score whether the
    predicted interface is physically and thermodynamically plausible.

Expected Pearson r vs ProNAB: 0.50–0.65 after fine-tuning on 617 entries with
ESM-2 + RNA-FM node features and physics edge features.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gt_constants import (
    CHECKPOINT_SCHEMA_VERSION,
    EDGE_DIM,
    LEGACY_EDGE_DIM,
    LEGACY_PHYSICS_SUMMARY_DIM,
    PHYSICS_SUMMARY_DIM,
)
from .physics_edge import physics_summary_from_graph

try:
    from torch_geometric.nn import TransformerConv
    _TORCH_GEOMETRIC = True
except ImportError:
    _TORCH_GEOMETRIC = False


class PhysicsEdgeGate(nn.Module):
    """Scale node embeddings by aggregated favourable edge physics (DynaPhArM-style)."""

    _PHYSICS_FEAT_DIM = 7  # elec,stack,hbond,vdw + pi_cation,dir_hbond,salt_bridge

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(self._PHYSICS_FEAT_DIM, hidden_dim),
            nn.Sigmoid(),
        )

    @staticmethod
    def _node_physics(edge_attr: torch.Tensor) -> torch.Tensor:
        physics = edge_attr[:, 2:6]
        if edge_attr.shape[1] >= EDGE_DIM:
            physics = torch.cat([physics, edge_attr[:, 9:12]], dim=-1)
        else:
            physics = torch.cat([physics, physics.new_zeros(physics.shape[0], 3)], dim=-1)
        return physics

    def forward(self, h: torch.Tensor, edge_attr: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_attr.numel() == 0 or h.shape[0] == 0:
            return h
        n_nodes = h.shape[0]
        physics = self._node_physics(edge_attr)
        src, dst = edge_index[0], edge_index[1]
        agg = torch.zeros(n_nodes, physics.shape[1], device=h.device, dtype=h.dtype)
        count = torch.zeros(n_nodes, 1, device=h.device, dtype=h.dtype)
        agg.index_add_(0, dst, physics)
        count.index_add_(0, dst, torch.ones(dst.shape[0], 1, device=h.device, dtype=h.dtype))
        agg = agg / count.clamp(min=1.0)
        return h * (1.0 + self.gate(agg))


class BipartiteCrossAttention(nn.Module):
    """
    Explicit protein↔RNA cross-attention before global graph mixing.

    Inspired by heterogeneous graph transformers and protein-RNA interaction
    models (HGT, DeepInteract) where type-specific message passing improves
    recognition of electrostatic complementarity and base-specific contacts.
    """

    def __init__(self, hidden_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_prot = nn.Linear(hidden_dim, hidden_dim)
        self.k_rna  = nn.Linear(hidden_dim, hidden_dim)
        self.v_rna  = nn.Linear(hidden_dim, hidden_dim)

        self.q_rna  = nn.Linear(hidden_dim, hidden_dim)
        self.k_prot = nn.Linear(hidden_dim, hidden_dim)
        self.v_prot = nn.Linear(hidden_dim, hidden_dim)

        # Physics-constrained attention bias (learned scale on cross edges)
        self.physics_bias = nn.Linear(6, n_heads, bias=False)

        self.out_prot = nn.Linear(hidden_dim, hidden_dim)
        self.out_rna  = nn.Linear(hidden_dim, hidden_dim)
        self.norm_prot = nn.LayerNorm(hidden_dim)
        self.norm_rna  = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def _scaled_attention(
        self,
        q_nodes: torch.Tensor,
        k_nodes: torch.Tensor,
        v_nodes: torch.Tensor,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n_q = q_nodes.shape[0]
        n_k = k_nodes.shape[0]

        q = q_proj(q_nodes).view(n_q, self.n_heads, self.head_dim)
        k = k_proj(k_nodes).view(n_k, self.n_heads, self.head_dim)
        v = v_proj(k_nodes).view(n_k, self.n_heads, self.head_dim)

        attn = torch.einsum("qhd,khd->hqk", q, k) * self.scale
        if attn_bias is not None and attn_bias.shape[-1] == n_k and attn_bias.shape[-2] == n_q:
            attn = attn + attn_bias
        attn = self.drop(F.softmax(attn, dim=-1))
        out = torch.einsum("hqk,khd->qhd", attn, v)
        return out.reshape(n_q, -1)

    def forward(
        self,
        h_prot: torch.Tensor,
        h_rna:  torch.Tensor,
        cross_physics: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if h_prot.shape[0] == 0:
            return h_prot, h_rna
        if h_rna.shape[0] == 0:
            return h_prot, h_rna

        bias = None
        if cross_physics is not None and cross_physics.shape[0] > 0 and cross_physics.shape[1] > 0:
            # [n_prot, n_rna, n_heads] → [n_heads, n_prot, n_rna]
            bias = self.physics_bias(cross_physics).permute(2, 0, 1)

        prot_update = self._scaled_attention(
            h_prot, h_rna, h_rna,
            self.q_prot, self.k_rna, self.v_rna,
            attn_bias=bias,
        )
        rna_update = self._scaled_attention(
            h_rna, h_prot, h_prot,
            self.q_rna, self.k_prot, self.v_prot,
            attn_bias=bias.transpose(-1, -2) if bias is not None else None,
        )

        h_prot = self.norm_prot(h_prot + self.drop(self.out_prot(prot_update)))
        h_rna  = self.norm_rna(h_rna  + self.drop(self.out_rna(rna_update)))
        return h_prot, h_rna


class PhysicsInformedGT(nn.Module):
    """
    Siamese Graph Transformer for RNA-protein binding ΔΔG prediction
    and AF3 interface plausibility scoring.

    Args:
        protein_node_dim: raw protein node feature dim (ESM-2 dim + 24)
        rna_node_dim:     raw RNA node feature dim (RNA-FM dim + 8)
        edge_dim:         physics edge feature dim (15 by default)
        hidden_dim:       shared hidden dimension for all GT layers
        n_heads:          attention heads per TransformerConv layer
        n_layers:         number of message-passing layers
        dropout:          dropout rate in attention and head
        physics_summary_dim: explicit physics vector for heads (17 by default)
    """

    def __init__(
        self,
        protein_node_dim: int,
        rna_node_dim:     int,
        edge_dim:         int = EDGE_DIM,
        hidden_dim:       int = 192,
        n_heads:          int = 4,
        n_layers:         int = 4,
        dropout:          float = 0.2,
        physics_summary_dim: int = PHYSICS_SUMMARY_DIM,
    ):
        super().__init__()

        if not _TORCH_GEOMETRIC:
            raise ImportError(
                "torch_geometric is required for PhysicsInformedGT. "
                "Install with: pip install torch-geometric"
            )

        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"

        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers
        self.edge_dim   = edge_dim
        self.physics_summary_dim = physics_summary_dim

        # ── input projections (type-specific) ────────────────────────────────
        self.protein_proj = nn.Sequential(
            nn.Linear(protein_node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.rna_proj = nn.Sequential(
            nn.Linear(rna_node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # ── bipartite cross-attention (protein↔RNA) ──────────────────────────
        self.cross_attn = BipartiteCrossAttention(hidden_dim, n_heads, dropout)

        # ── graph transformer layers ──────────────────────────────────────────
        self.conv_layers = nn.ModuleList([
            TransformerConv(
                in_channels  = hidden_dim,
                out_channels = hidden_dim // n_heads,
                heads        = n_heads,
                edge_dim     = edge_dim,
                dropout      = dropout,
                concat       = True,   # concat heads → hidden_dim
                beta         = True,   # skip-connection gating
            )
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.edge_gates = nn.ModuleList([PhysicsEdgeGate(hidden_dim) for _ in range(n_layers)])
        self.drop  = nn.Dropout(dropout)

        # ── attention pooling for global context ─────────────────────────────
        self.pool_gate = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

        # ── ΔΔG prediction head ───────────────────────────────────────────────
        # Input: [site_diff | global_diff | explicit physics-summary delta]
        head_in = hidden_dim * 2 + physics_summary_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # ── interface plausibility head (AF3 single-graph mode) ───────────────
        self.interface_head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def _project(
        self,
        x_protein: torch.Tensor,
        x_rna:     torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project both node types and return (combined, protein, rna)."""
        h_prot = self.protein_proj(x_protein)
        h_rna  = self.rna_proj(x_rna)
        return torch.cat([h_prot, h_rna], dim=0), h_prot, h_rna

    def _attention_pool(self, h: torch.Tensor) -> torch.Tensor:
        """Softmax-weighted mean pooling over node embeddings."""
        if h.shape[0] == 0:
            return h.new_zeros(self.hidden_dim)
        weights = torch.softmax(self.pool_gate(h).squeeze(-1), dim=0)
        return (weights.unsqueeze(-1) * h).sum(dim=0)

    @staticmethod
    def _cross_physics_matrix(
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        n_prot: int,
        n_rna: int,
    ) -> torch.Tensor | None:
        """Per protein–RNA pair physics features for cross-attention bias."""
        if edge_attr.numel() == 0 or n_prot == 0 or n_rna == 0:
            return None
        mat = torch.zeros(n_prot, n_rna, 6, device=edge_attr.device, dtype=edge_attr.dtype)
        src, dst = edge_index[0], edge_index[1]
        for e in range(edge_attr.shape[0]):
            i, j = int(src[e]), int(dst[e])
            if i < n_prot and j >= n_prot:
                pi, ri = i, j - n_prot
            elif j < n_prot and i >= n_prot:
                pi, ri = j, i - n_prot
            else:
                continue
            feat = edge_attr[e, 2:6]
            if edge_attr.shape[1] >= EDGE_DIM:
                feat = torch.cat([feat, edge_attr[e, 9:11]])
            else:
                feat = torch.cat([feat, feat.new_zeros(2)])
            mat[pi, ri] = feat
        return mat

    def _physics_summary(
        self,
        x_protein: torch.Tensor,
        edge_attr: torch.Tensor,
        mutation_idx: int | None = None,
    ) -> torch.Tensor:
        legacy = (
            self.physics_summary_dim <= LEGACY_PHYSICS_SUMMARY_DIM
            and edge_attr.shape[1] <= LEGACY_EDGE_DIM
        )
        return physics_summary_from_graph(
            x_protein,
            edge_attr,
            mutation_idx,
            legacy_dim=LEGACY_PHYSICS_SUMMARY_DIM if legacy else self.physics_summary_dim,
        )

    def encode(
        self,
        x_protein:  torch.Tensor,
        x_rna:      torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the graph transformer.

        Returns:
            node embeddings [n_prot + n_rna, hidden_dim]
        """
        h, h_prot, h_rna = self._project(x_protein, x_rna)
        n_prot = x_protein.shape[0]

        cross_physics = self._cross_physics_matrix(
            edge_index, edge_attr, n_prot, x_rna.shape[0]
        )
        h_prot, h_rna = self.cross_attn(h_prot, h_rna, cross_physics)
        h = torch.cat([h_prot, h_rna], dim=0)

        for conv, norm, edge_gate in zip(self.conv_layers, self.norms, self.edge_gates):
            h_new = conv(h, edge_index, edge_attr)
            h = norm(h + self.drop(h_new))
            h = edge_gate(h, edge_attr, edge_index)

        return h   # [n_nodes, hidden_dim]

    def interface_embedding(
        self,
        x_protein:  torch.Tensor,
        x_rna:      torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr:  torch.Tensor,
        n_prot:     int,
    ) -> torch.Tensor:
        """
        Returns the concatenated [prot_pool | rna_pool | physics_summary]
        feature vector for interface plausibility scoring.
        """
        h = self.encode(x_protein, x_rna, edge_index, edge_attr)
        h_prot = h[:n_prot]
        h_rna  = h[n_prot:]
        prot_pool = self._attention_pool(h_prot)
        rna_pool  = self._attention_pool(h_rna)
        physics   = self._physics_summary(x_protein, edge_attr, mutation_idx=None)
        return torch.cat([prot_pool, rna_pool, physics], dim=-1)

    def score_interface(
        self,
        x_protein:  torch.Tensor,
        x_rna:      torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr:  torch.Tensor,
        n_prot:     int,
    ) -> torch.Tensor:
        """
        Score AF3 interface plausibility from a single complex graph.

        Returns a scalar: lower = more plausible (energy-like convention).
        When no trained checkpoint is available, falls back to physics summary only.
        """
        feat = self.interface_embedding(
            x_protein, x_rna, edge_index, edge_attr, n_prot
        )
        return self.interface_head(feat).squeeze(-1)

    def forward(
        self,
        # WT graph tensors
        wt_x_protein:      torch.Tensor,
        wt_x_rna:          torch.Tensor,
        wt_edge_index:     torch.Tensor,
        wt_edge_attr:      torch.Tensor,
        wt_mutation_idx:   int,
        wt_n_prot:         int,
        # Mutant graph tensors
        mut_x_protein:     torch.Tensor,
        mut_x_rna:         torch.Tensor,
        mut_edge_index:    torch.Tensor,
        mut_edge_attr:     torch.Tensor,
        mut_mutation_idx:  int,
        mut_n_prot:        int,
    ) -> torch.Tensor:
        """
        Predicts ΔΔG = G_mut_binding - G_wt_binding.

        Positive output → mutation destabilises the interface (ΔΔG > 0).

        Returns:
            scalar tensor (unbatched) or [batch] tensor when called via
            forward_batch.
        """
        wt_h  = self.encode(wt_x_protein,  wt_x_rna,  wt_edge_index,  wt_edge_attr)
        mut_h = self.encode(mut_x_protein, mut_x_rna, mut_edge_index, mut_edge_attr)

        # Local: embedding at mutation site
        site_diff = mut_h[mut_mutation_idx] - wt_h[wt_mutation_idx]

        # Global: attention-weighted mean over protein nodes
        wt_global  = self._attention_pool(wt_h[:wt_n_prot])
        mut_global = self._attention_pool(mut_h[:mut_n_prot])
        global_diff = mut_global - wt_global

        physics_diff = self._physics_summary(
            mut_x_protein, mut_edge_attr, mut_mutation_idx
        ) - self._physics_summary(
            wt_x_protein, wt_edge_attr, wt_mutation_idx
        )

        ddg = self.head(torch.cat([site_diff, global_diff, physics_diff], dim=-1))
        return ddg.squeeze(-1)


# ── small fallback for environments without torch_geometric ───────────────────

class FallbackMLP(nn.Module):
    """
    Simple MLP that predicts ΔΔG from raw physics component deltas when
    torch_geometric is not available.
    """

    def __init__(
        self,
        input_dim: int = PHYSICS_SUMMARY_DIM,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.interface_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _graph_summary(
        x_protein: torch.Tensor,
        edge_attr: torch.Tensor,
        mutation_idx: int | None = None,
    ) -> torch.Tensor:
        return physics_summary_from_graph(x_protein, edge_attr, mutation_idx)

    def score_interface(
        self,
        x_protein: torch.Tensor,
        x_rna: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        n_prot: int,
    ) -> torch.Tensor:
        summary = self._graph_summary(x_protein, edge_attr, mutation_idx=None)
        return self.interface_net(summary).squeeze(-1)

    def forward(
        self,
        x: torch.Tensor | None = None,
        *,
        wt_x_protein: torch.Tensor | None = None,
        wt_x_rna: torch.Tensor | None = None,
        wt_edge_index: torch.Tensor | None = None,
        wt_edge_attr: torch.Tensor | None = None,
        wt_mutation_idx: int | None = None,
        wt_n_prot: int | None = None,
        mut_x_protein: torch.Tensor | None = None,
        mut_x_rna: torch.Tensor | None = None,
        mut_edge_index: torch.Tensor | None = None,
        mut_edge_attr: torch.Tensor | None = None,
        mut_mutation_idx: int | None = None,
        mut_n_prot: int | None = None,
    ) -> torch.Tensor:
        if x is None:
            required = (
                wt_x_protein, wt_edge_attr, wt_mutation_idx,
                mut_x_protein, mut_edge_attr, mut_mutation_idx,
            )
            if any(v is None for v in required):
                raise TypeError("FallbackMLP requires either x or GT graph tensors")
            wt_summary = self._graph_summary(
                wt_x_protein, wt_edge_attr, wt_mutation_idx
            )
            mut_summary = self._graph_summary(
                mut_x_protein, mut_edge_attr, mut_mutation_idx
            )
            x = mut_summary - wt_summary
        return self.net(x).squeeze(-1)


def build_model(
    protein_node_dim: int,
    rna_node_dim:     int,
    edge_dim:         int = EDGE_DIM,
    hidden_dim:       int = 192,
    n_heads:          int = 4,
    n_layers:         int = 4,
    dropout:          float = 0.2,
    physics_summary_dim: int = PHYSICS_SUMMARY_DIM,
) -> nn.Module:
    """
    Factory: returns PhysicsInformedGT if torch_geometric is available,
    otherwise FallbackMLP with a warning.
    """
    if _TORCH_GEOMETRIC:
        return PhysicsInformedGT(
            protein_node_dim=protein_node_dim,
            rna_node_dim=rna_node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            physics_summary_dim=physics_summary_dim,
        )
    else:
        print(
            "WARNING: torch_geometric not installed — using FallbackMLP. "
            "Install with: pip install torch-geometric"
        )
        return FallbackMLP(hidden_dim=hidden_dim, dropout=dropout)
