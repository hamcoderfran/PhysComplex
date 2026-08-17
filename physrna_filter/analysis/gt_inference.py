"""
PhysGT inference for AF3/RoseTTAFold interface plausibility scoring.

Loads a trained checkpoint (or falls back to physics-only scoring) and
scores whether a predicted protein-RNA interface is physically plausible.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .graph_features import build_af3_interface_graph, InterfaceGraph
from ..analysis.gt_model import build_model, FallbackMLP
from ..analysis.gt_constants import EDGE_DIM, LEGACY_EDGE_DIM, PHYSICS_SUMMARY_DIM, LEGACY_PHYSICS_SUMMARY_DIM
from .physics_edge import coerce_edge_attr
from .esm_embeddings import get_esm2_embeddings_from_structure, ESM_DIM
from .rnafm_embeddings import (
    get_rnafm_embeddings_from_structure,
    effective_rnafm_feature_dim,
)
from .thresholds import load_thresholds
from ..structure.parse_complex import ParsedComplex, parse_complex
from ..structure.partner_selection import (
    primary_protein_chain_id,
    primary_rna_chain_id,
    select_partner_pair,
)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class GtInferenceContext:
    """
    Reusable PhysGT model + metadata for batch screening.

    Avoids reloading ESM-2 / RNA-FM weights and the GNN checkpoint per structure.
    """

    checkpoint_path: str | None = None
    model: torch.nn.Module | None = None
    meta: dict = field(default_factory=dict)
    _loaded: bool = False

    def ensure_loaded(
        self,
        checkpoint_path: str | None = None,
        *,
        use_esm: bool | None = None,
        use_rnafm: bool | None = None,
        require_trained: bool = False,
    ) -> tuple[torch.nn.Module, dict]:
        path = checkpoint_path or self.checkpoint_path
        if not self._loaded or (path and path != self.meta.get("checkpoint")):
            self.model, self.meta = load_gt_model(
                path, use_esm=use_esm, use_rnafm=use_rnafm
            )
            self._loaded = True
            self.checkpoint_path = path

        if require_trained and self.meta.get("physics_only", True):
            raise RuntimeError(
                "Trained PhysGT interface head required but checkpoint is "
                "physics-only. Run: python -m physrna_filter.validation.deploy_gt"
            )
        return self.model, self.meta


def _default_checkpoint_paths() -> list[Path]:
    here = Path(__file__).resolve().parent
    return [
        here.parent / "validation" / "gt_checkpoint.pt",
        Path("gt_checkpoint.pt"),
    ]


def _resolve_checkpoint(checkpoint_path: str | None) -> str | None:
    if checkpoint_path and os.path.exists(checkpoint_path):
        return checkpoint_path
    for candidate in _default_checkpoint_paths():
        if candidate.exists():
            return str(candidate)
    return None


def load_gt_model(
    checkpoint_path: str | None = None,
    use_esm: bool | None = None,
    use_rnafm: bool | None = None,
) -> tuple[torch.nn.Module, dict]:
    """
    Load PhysGT from checkpoint.  Returns (model, metadata).

    Node dimensions are always taken from the checkpoint when available so
    that graph features match the trained weights.
    """
    ckpt_path = _resolve_checkpoint(checkpoint_path)

    meta: dict = {
        "checkpoint": ckpt_path,
        "physics_only": True,
    }

    if ckpt_path:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        protein_node_dim = payload["protein_node_dim"]
        rna_node_dim     = payload["rna_node_dim"]
        hidden_dim       = payload.get("hidden_dim", 192)
        n_layers         = payload.get("n_layers", 4)
        edge_dim         = payload.get("edge_dim", LEGACY_EDGE_DIM)
        physics_summary_dim = payload.get(
            "physics_summary_dim",
            LEGACY_PHYSICS_SUMMARY_DIM if edge_dim <= LEGACY_EDGE_DIM else PHYSICS_SUMMARY_DIM,
        )
        esm_dim          = payload.get("esm_dim", protein_node_dim - 24)
        rnafm_dim        = payload.get("rnafm_dim", rna_node_dim - 8)

        model = build_model(
            protein_node_dim=protein_node_dim,
            rna_node_dim=rna_node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            physics_summary_dim=physics_summary_dim,
        )
        missing, unexpected = model.load_state_dict(payload["model_state"], strict=False)
        if unexpected:
            print(f"PhysGT checkpoint: ignoring unexpected keys ({len(unexpected)})")
        if missing and len(missing) > 20:
            print(f"PhysGT checkpoint: {len(missing)} missing keys (schema upgrade?)")
        interface_trained = bool(payload.get("interface_head_trained", False))
        meta.update({
            "protein_node_dim": protein_node_dim,
            "rna_node_dim": rna_node_dim,
            "esm_dim": esm_dim,
            "rnafm_dim": rnafm_dim,
            "edge_dim": edge_dim,
            "use_esm": esm_dim > 0,
            "use_rnafm": rnafm_dim > 0,
            "target_mean": payload.get("target_mean", 0.0),
            "target_std":  payload.get("target_std", 1.0),
            "physics_only": not interface_trained,
            "interface_head_trained": interface_trained,
            "af3_panel_finetuned": bool(payload.get("af3_panel_finetuned", False)),
            "schema_version": payload.get("schema_version"),
        })
    else:
        use_esm_flag   = True if use_esm is None else use_esm
        use_rnafm_flag = True if use_rnafm is None else use_rnafm
        esm_dim   = ESM_DIM if use_esm_flag else 0
        rnafm_dim = effective_rnafm_feature_dim(use_rnafm_flag)
        protein_node_dim = 24 + esm_dim
        rna_node_dim     = 8  + rnafm_dim
        model = build_model(
            protein_node_dim=protein_node_dim,
            rna_node_dim=rna_node_dim,
        )
        meta.update({
            "protein_node_dim": protein_node_dim,
            "rna_node_dim": rna_node_dim,
            "esm_dim": esm_dim,
            "rnafm_dim": rnafm_dim,
            "edge_dim": EDGE_DIM,
            "use_esm": use_esm_flag,
            "use_rnafm": use_rnafm_flag,
        })

    model.eval()
    return model.to(_DEVICE), meta


def _graph_to_device(g: InterfaceGraph) -> InterfaceGraph:
    return InterfaceGraph(
        x_protein=g.x_protein.to(_DEVICE),
        x_rna=g.x_rna.to(_DEVICE),
        edge_index=g.edge_index.to(_DEVICE),
        edge_attr=g.edge_attr.to(_DEVICE),
        node_types=g.node_types.to(_DEVICE),
        mutation_node_idx=g.mutation_node_idx,
        protein_residues=g.protein_residues,
        rna_residues=g.rna_residues,
        prot_coords=g.prot_coords.to(_DEVICE) if g.prot_coords is not None else None,
        rna_coords=g.rna_coords.to(_DEVICE) if g.rna_coords is not None else None,
    )


def build_af3_graph(
    pdb_path: str,
    esm_dim: int | None = None,
    rnafm_dim: int | None = None,
    use_esm: bool | None = None,
    use_rnafm: bool | None = None,
    edge_dim: int | None = None,
    *,
    model_rank: int = 0,
    parsed: ParsedComplex | None = None,
    protein_chains: list | None = None,
    rna_chains: list | None = None,
    protein_chain_id: str | None = None,
    rna_chain_id: str | None = None,
) -> InterfaceGraph:
    """Build an AF3 interface graph matching checkpoint dimensions."""
    if parsed is None:
        parsed = parse_complex(pdb_path, model_rank=model_rank)

    if protein_chains is None or rna_chains is None:
        protein_chains, rna_chains = select_partner_pair(
            parsed.protein_chains,
            parsed.rna_chains,
            protein_chain_id=protein_chain_id,
            rna_chain_id=rna_chain_id,
        )

    pdb_id = parsed.pdb_id or Path(pdb_path).stem

    if esm_dim is None:
        esm_dim = ESM_DIM if (use_esm is None or use_esm) else 0
    if rnafm_dim is None:
        rnafm_dim = effective_rnafm_feature_dim(use_rnafm is not False)
    if edge_dim is None:
        edge_dim = EDGE_DIM

    esm_emb = rnafm_emb = None
    prot_id = protein_chain_id or primary_protein_chain_id(protein_chains, rna_chains)
    rna_id  = rna_chain_id or primary_rna_chain_id(rna_chains, protein_chains)

    if esm_dim > 0 and protein_chains and prot_id:
        esm_emb = get_esm2_embeddings_from_structure(
            pdb_id, prot_id, protein_chains
        )

    if rnafm_dim > 0 and rna_chains and rna_id:
        rnafm_emb = get_rnafm_embeddings_from_structure(
            pdb_id, rna_id, rna_chains
        )

    return build_af3_interface_graph(
        pdb_path,
        pdb_id=pdb_id,
        esm_embeddings=esm_emb,
        rnafm_embeddings=rnafm_emb,
        esm_dim=esm_dim,
        rnafm_dim=rnafm_dim,
        edge_dim=edge_dim,
        model_rank=model_rank,
        parsed=parsed,
        protein_chains=protein_chains,
        rna_chains=rna_chains,
    )


def count_prot_rna_edges(graph: InterfaceGraph) -> int:
    if graph.edge_attr.numel() == 0:
        return 0
    return int((graph.edge_attr[:, 6] > 0.5).sum().item())


def normalize_interface_score(
    score: float,
    graph: InterfaceGraph,
    *,
    method: str = "sqrt_edges",
) -> float:
    """
    Size-normalize raw interface scores to reduce bias from large interfaces.

    Methods:
      - sqrt_edges: score / sqrt(n_prot_rna_edges)
      - per_nt:     score / max(1, n_rna_interface_nodes)
    """
    if method == "per_nt":
        denom = max(1, len(graph.rna_residues))
    else:
        denom = max(1.0, float(count_prot_rna_edges(graph))) ** 0.5
    return float(score) / denom


def score_interface_graph(
    model: torch.nn.Module,
    graph: InterfaceGraph,
    edge_dim: int | None = None,
) -> float:
    """Score a pre-built interface graph.  Lower = more plausible."""
    g = _graph_to_device(graph)
    n_prot = g.x_protein.shape[0]
    target_edge_dim = edge_dim or getattr(model, "edge_dim", g.edge_attr.shape[1])
    edge_attr = coerce_edge_attr(g.edge_attr, target_edge_dim)

    with torch.no_grad():
        if hasattr(model, "score_interface"):
            score = model.score_interface(
                g.x_protein, g.x_rna,
                g.edge_index, edge_attr,
                n_prot,
            )
        else:
            raise TypeError("Model does not support interface scoring")

    return float(score.cpu())


def physics_only_interface_score(graph: InterfaceGraph) -> float:
    """
    Pure physics summary score when no trained GT checkpoint is available.

    Sums favorable contact terms from protein-RNA edges.  More negative = better.
    """
    if graph.edge_attr.numel() == 0:
        return 10.0

    prot_rna = graph.edge_attr[:, 6] > 0.5
    if not prot_rna.any():
        return 5.0

    terms = graph.edge_attr[prot_rna, 2:6].sum(dim=0)
    return float(terms.sum().item())


def gt_verdict(score: float) -> str:
    t = load_thresholds()
    if score < t.get("gt_pass", -2.0):
        return "PASS"
    if score < t.get("gt_warn", 0.5):
        return "WARN"
    return "FAIL"


def score_af3_interface(
    pdb_path: str,
    checkpoint_path: str | None = None,
    use_esm: bool | None = None,
    use_rnafm: bool | None = None,
    *,
    model_rank: int = 0,
    parsed: ParsedComplex | None = None,
    protein_chains: list | None = None,
    rna_chains: list | None = None,
    inference_context: GtInferenceContext | None = None,
    require_trained: bool = False,
    allow_physics_only: bool = True,
) -> dict:
    """
    End-to-end AF3 interface plausibility scoring.

    Graph dimensions are matched to the checkpoint automatically.
    """
    if inference_context is not None:
        model, meta = inference_context.ensure_loaded(
            checkpoint_path,
            use_esm=use_esm,
            use_rnafm=use_rnafm,
            require_trained=require_trained,
        )
    else:
        model, meta = load_gt_model(checkpoint_path, use_esm=use_esm, use_rnafm=use_rnafm)
        if require_trained and meta.get("physics_only", True):
            raise RuntimeError(
                "Trained PhysGT checkpoint required but interface head is not trained."
            )

    graph = build_af3_graph(
        pdb_path,
        esm_dim=meta.get("esm_dim"),
        rnafm_dim=meta.get("rnafm_dim"),
        edge_dim=meta.get("edge_dim", EDGE_DIM),
        model_rank=model_rank,
        parsed=parsed,
        protein_chains=protein_chains,
        rna_chains=rna_chains,
    )

    physics_only = bool(meta.get("physics_only") or isinstance(model, FallbackMLP))
    if physics_only and not allow_physics_only and require_trained:
        raise RuntimeError("PhysGT fell back to physics-only scoring.")

    if physics_only:
        gt_score = physics_only_interface_score(graph)
    else:
        gt_score = score_interface_graph(
            model, graph, edge_dim=meta.get("edge_dim", EDGE_DIM)
        )

    prot_rna_edges = count_prot_rna_edges(graph)
    gt_score_norm = normalize_interface_score(gt_score, graph, method="sqrt_edges")
    gt_score_per_nt = normalize_interface_score(gt_score, graph, method="per_nt")

    return {
        "gt_score": gt_score,
        "gt_score_norm": gt_score_norm,
        "gt_score_per_nt": gt_score_per_nt,
        "gt_verdict": gt_verdict(gt_score_norm),
        "n_prot_interface": len(graph.protein_residues),
        "n_rna_interface": len(graph.rna_residues),
        "n_prot_rna_edges": prot_rna_edges,
        "physics_only": physics_only,
        "interface_head_trained": meta.get("interface_head_trained", False),
        "af3_panel_finetuned": meta.get("af3_panel_finetuned", False),
    }
