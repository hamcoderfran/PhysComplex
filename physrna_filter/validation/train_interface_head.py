"""
Contrastive training for the PhysGT interface plausibility head.

Positive examples: crystal-structure interfaces (ProNAB / merged training PDBs).
Negative examples:
  - Entropic decoys: rigid-body RNA translation + full graph rebuild
  - Sequence decoys: permuted RNA node features (wrong sequence at each site)
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.fetch_training_data import fetch_training_data
from ..data.fetch_pdb import download_pdb
from ..config import default_interface_pretrain_path
from ..analysis.graph_features import build_af3_interface_graph, InterfaceGraph
from ..analysis.graph_decoy import (
    make_entropic_decoy_graph,
    make_sequence_decoy_graph,
)
from ..analysis.gt_model import build_model
from ..analysis.gt_constants import CHECKPOINT_SCHEMA_VERSION, EDGE_DIM, PHYSICS_SUMMARY_DIM
from ..analysis.esm_embeddings import get_esm2_embeddings_from_structure, ESM_DIM
from ..analysis.rnafm_embeddings import (
    get_rnafm_embeddings_from_structure,
    effective_rnafm_feature_dim,
    prepare_rnafm_for_training,
)
from ..analysis.gt_inference import count_prot_rna_edges
from ..data.boltz_training_pairs import (
    BoltzTrainingGroup,
    load_boltz_training_graphs,
)
from ..data.af3_decoy_training_pairs import (
    Af3DecoyTrainingGroup,
    load_af3_decoy_training_graphs,
)
try:
    from ..data.foldbench_training_pairs import (
        FoldBenchTrainingGroup,
        load_foldbench_training_graphs,
    )
except ImportError:  # slim starter kit — FoldBench OOD training optional
    from dataclasses import dataclass

    @dataclass
    class FoldBenchTrainingGroup:  # type: ignore[no-redef]
        target_id: str = ""
        native_job: str = ""
        native_path: str = ""
        decoy_jobs: list[str] | None = None
        decoy_paths: list[str] | None = None

    def load_foldbench_training_graphs(*_args, **_kwargs):  # type: ignore[misc]
        return [], {}

from ..structure.parse_complex import parse_complex

OodTrainingGroup = BoltzTrainingGroup | Af3DecoyTrainingGroup | FoldBenchTrainingGroup

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _graph_to_device(g: InterfaceGraph) -> InterfaceGraph:
    return InterfaceGraph(
        x_protein=g.x_protein.to(DEVICE),
        x_rna=g.x_rna.to(DEVICE),
        edge_index=g.edge_index.to(DEVICE),
        edge_attr=g.edge_attr.to(DEVICE),
        node_types=g.node_types.to(DEVICE),
        mutation_node_idx=g.mutation_node_idx,
        protein_residues=g.protein_residues,
        rna_residues=g.rna_residues,
        prot_coords=g.prot_coords.to(DEVICE) if g.prot_coords is not None else None,
        rna_coords=g.rna_coords.to(DEVICE) if g.rna_coords is not None else None,
    )


def _build_positive_graph(
    pdb_id: str,
    use_esm: bool,
    use_rnafm: bool,
    esm_dim: int,
    rnafm_dim: int,
) -> tuple[InterfaceGraph, str] | None:
    pdb_path = download_pdb(pdb_id)
    if pdb_path is None:
        return None

    parsed = parse_complex(pdb_path)

    esm_emb = rnafm_emb = None
    if use_esm and parsed.protein_chains:
        esm_emb = get_esm2_embeddings_from_structure(
            pdb_id, parsed.protein_chains[0].id, parsed.protein_chains
        )
    if use_rnafm and parsed.rna_chains:
        rnafm_emb = get_rnafm_embeddings_from_structure(
            pdb_id, parsed.rna_chains[0].id, parsed.rna_chains
        )

    graph = build_af3_interface_graph(
        pdb_path, pdb_id=pdb_id,
        esm_embeddings=esm_emb, rnafm_embeddings=rnafm_emb,
        esm_dim=esm_dim, rnafm_dim=rnafm_dim,
    )
    return graph, pdb_path


def _validate_graph_dims(graphs: list[InterfaceGraph]) -> tuple[int, int]:
    protein_node_dim = graphs[0].x_protein.shape[1]
    rna_node_dim = graphs[0].x_rna.shape[1]
    for i, g in enumerate(graphs[1:], start=1):
        if g.x_protein.shape[1] != protein_node_dim or g.x_rna.shape[1] != rna_node_dim:
            raise RuntimeError(
                f"Inconsistent graph feature dims at index {i}: "
                f"protein {g.x_protein.shape[1]} vs {protein_node_dim}, "
                f"RNA {g.x_rna.shape[1]} vs {rna_node_dim}. "
                "Re-run: python -m physrna_filter.data.verify_rnafm_weights"
            )
    return protein_node_dim, rna_node_dim


def _sample_negative(
    pos_g: InterfaceGraph,
    pdb_path: str,
    pdb_id: str,
    seed: int,
    *,
    esm_dim: int,
    rnafm_dim: int,
    esm_emb,
    rnafm_emb,
) -> InterfaceGraph:
    rng = random.Random(seed)
    if rng.random() < 0.5:
        return make_sequence_decoy_graph(pos_g, seed=seed)
    return make_entropic_decoy_graph(
        pos_g,
        seed=seed,
        pdb_path=pdb_path,
        pdb_id=pdb_id,
        esm_embeddings=esm_emb,
        rnafm_embeddings=rnafm_emb,
        esm_dim=esm_dim,
        rnafm_dim=rnafm_dim,
    )


def _interface_score(
    model: torch.nn.Module,
    graph: InterfaceGraph,
    *,
    normalized: bool = True,
) -> torch.Tensor:
    raw = model.score_interface(
        graph.x_protein, graph.x_rna, graph.edge_index, graph.edge_attr,
        graph.x_protein.shape[0],
    )
    if not normalized:
        return raw
    denom = max(1.0, float(count_prot_rna_edges(graph))) ** 0.5
    return raw / denom


def _boltz_group_loss(
    model: torch.nn.Module,
    group: OodTrainingGroup,
    graphs: dict[str, InterfaceGraph],
    margin: float,
    *,
    normalized: bool = True,
) -> torch.Tensor:
    native = graphs.get(group.native_job)
    if native is None:
        return torch.zeros((), device=DEVICE)
    pos = _graph_to_device(native)
    score_pos = _interface_score(model, pos, normalized=normalized)

    losses = []
    for job in group.decoy_jobs:
        dec = graphs.get(job)
        if dec is None:
            continue
        neg = _graph_to_device(dec)
        score_neg = _interface_score(model, neg, normalized=normalized)
        losses.append(F.relu(margin + score_pos - score_neg))
    if not losses:
        return score_pos.new_zeros(())
    return torch.stack(losses).mean()


def _eval_boltz_partner_accuracy(
    model: torch.nn.Module,
    groups: list[OodTrainingGroup],
    graphs: dict[str, InterfaceGraph],
    *,
    normalized: bool = True,
) -> tuple[int, int]:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for group in groups:
            native = graphs.get(group.native_job)
            if native is None:
                continue
            candidates = [group.native_job] + [
                j for j in group.decoy_jobs if j in graphs
            ]
            if len(candidates) < 2:
                continue
            scores = {}
            for job in candidates:
                g = _graph_to_device(graphs[job])
                scores[job] = float(_interface_score(model, g, normalized=normalized).cpu())
            best = min(scores, key=scores.get)
            total += 1
            if best == group.native_job:
                correct += 1
    model.train()
    return correct, total


def train_interface_head(
    max_entries: int | None = None,
    epochs: int = 30,
    lr: float = 1e-3,
    margin: float = 1.0,
    use_esm: bool = True,
    use_rnafm: bool = True,
    output_path: str | None = None,
    seed: int = 7,
    include_nabe: bool = True,
    include_literature: bool = True,
    pronab_only: bool = False,
    train_encoder: bool = False,
    boltz_bundle: str | None = None,
    boltz_weight: float = 3.0,
    boltz_partner_groups: list[str] | None = None,
    af3_decoy_bundle: str | None = None,
    af3_predictions_subdir: str = "af3_predictions",
    af3_weight: float = 6.0,
    af3_partner_groups: list[str] | None = None,
    foldbench_predictions: str | None = None,
    foldbench_eval_csv: str | None = None,
    foldbench_dockq_threshold: float = 0.23,
    foldbench_weight: float = 4.0,
    foldbench_max_targets: int | None = None,
    max_crystal_entries: int | None = 30,
    normalized_scores: bool = True,
    eval_every: int = 5,
) -> None:
    """
    Train interface plausibility head with margin ranking loss:
        L = max(0, margin + score(positive) - score(negative))
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    prepare_rnafm_for_training(use_rnafm)
    esm_dim   = ESM_DIM if use_esm else 0
    rnafm_dim = effective_rnafm_feature_dim(use_rnafm)

    boltz_groups: list[BoltzTrainingGroup] = []
    boltz_graphs: dict[str, InterfaceGraph] = {}
    if boltz_bundle:
        print(f"Loading Boltz training graphs from {boltz_bundle} ...")
        boltz_groups, boltz_graphs = load_boltz_training_graphs(
            boltz_bundle,
            use_esm=use_esm,
            use_rnafm=use_rnafm,
            partner_groups=boltz_partner_groups,
        )
        if boltz_partner_groups:
            print(f"  filtered to partner groups: {boltz_partner_groups}")
        print(
            f"  {len(boltz_groups)} partner groups, "
            f"{len(boltz_graphs)} graphs cached"
        )

    af3_groups: list[Af3DecoyTrainingGroup] = []
    af3_graphs: dict[str, InterfaceGraph] = {}
    if af3_decoy_bundle:
        print(f"Loading AF3 decoy training graphs from {af3_decoy_bundle} ...")
        af3_groups, af3_graphs = load_af3_decoy_training_graphs(
            af3_decoy_bundle,
            predictions_subdir=af3_predictions_subdir,
            use_esm=use_esm,
            use_rnafm=use_rnafm,
            partner_groups=af3_partner_groups,
        )
        if af3_partner_groups:
            print(f"  filtered to partner groups: {af3_partner_groups}")
        print(
            f"  {len(af3_groups)} AF3 partner groups, "
            f"{len(af3_graphs)} graphs cached"
        )

    foldbench_groups: list[FoldBenchTrainingGroup] = []
    foldbench_graphs: dict[str, InterfaceGraph] = {}
    if foldbench_predictions:
        print(f"Loading FoldBench failure groups from {foldbench_predictions} ...")
        foldbench_groups, foldbench_graphs = load_foldbench_training_graphs(
            foldbench_predictions,
            dockq_threshold=foldbench_dockq_threshold,
            evaluation_csv=foldbench_eval_csv,
            max_targets=foldbench_max_targets,
            use_esm=use_esm,
            use_rnafm=use_rnafm,
        )
        print(
            f"  {len(foldbench_groups)} FoldBench targets, "
            f"{len(foldbench_graphs)} graphs cached"
        )

    ood_graphs = {**boltz_graphs, **af3_graphs, **foldbench_graphs}
    ood_groups: list[OodTrainingGroup] = [
        *boltz_groups,
        *af3_groups,
        *foldbench_groups,
    ]

    positives: list[InterfaceGraph] = []
    pdb_paths: dict[int, str] = {}
    pdb_ids: dict[int, str] = {}

    if boltz_graphs or af3_graphs:
        for job, graph in ood_graphs.items():
            positives.append(graph)
            pdb_paths[len(positives) - 1] = job
            pdb_ids[len(positives) - 1] = job

    cap = max_entries if max_entries is not None else max_crystal_entries
    if cap != 0:
        training = fetch_training_data(
            include_nabe=include_nabe and not pronab_only,
            include_literature=include_literature and not pronab_only,
        )
        unique_pdbs = list(dict.fromkeys(training["pdb_id"].tolist()))
        if cap is not None:
            unique_pdbs = unique_pdbs[:cap]

        sources = training["source"].value_counts().to_dict() if "source" in training.columns else {}
        print(
            f"Crystal corpus: {len(training)} mutation entries "
            f"({sources}) → {len(unique_pdbs)} unique PDB(s)"
        )

        for n_done, pdb_id in enumerate(unique_pdbs, start=1):
            try:
                result = _build_positive_graph(
                    pdb_id, use_esm, use_rnafm, esm_dim, rnafm_dim,
                )
                if result is None:
                    continue
                g, pdb_path = result
                if g.edge_attr.numel() > 0:
                    idx = len(positives)
                    positives.append(g)
                    pdb_paths[idx] = pdb_path
                    pdb_ids[idx] = pdb_id
            except Exception as e:
                print(f"  SKIP {pdb_id}: {e}")
            if n_done % 10 == 0 or n_done == len(unique_pdbs):
                print(f"  loaded {len(positives)} total graphs ({n_done}/{len(unique_pdbs)} crystal tried)")

    if len(positives) < 5 and not ood_groups:
        raise RuntimeError(f"Only {len(positives)} positive graphs — need at least 5")

    protein_node_dim, rna_node_dim = _validate_graph_dims(positives)
    print(
        f"Training interface head on {len(positives)} positive complexes "
        f"(protein dim={protein_node_dim}, RNA dim={rna_node_dim}, "
        f"edge_dim={EDGE_DIM})"
    )

    model = build_model(
        protein_node_dim=protein_node_dim,
        rna_node_dim=rna_node_dim,
        edge_dim=EDGE_DIM,
    ).to(DEVICE)

    for name, param in model.named_parameters():
        if train_encoder or boltz_bundle or af3_decoy_bundle or foldbench_predictions:
            param.requires_grad = True
        elif any(
            tag in name
            for tag in ("interface_head", "cross_attn", "edge_gates", "physics_bias", "interface_ddg_coupling")
        ):
            param.requires_grad = True
        else:
            param.requires_grad = False

    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    crystal_indices = list(range(len(positives)))
    if ood_graphs:
        crystal_indices = [
            i for i in crystal_indices
            if pdb_ids.get(i, "") not in ood_graphs
        ]

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_steps = 0
        order = list(crystal_indices)
        random.shuffle(order)

        for i in order:
            pos_g = positives[i]
            neg_g = _sample_negative(
                pos_g,
                pdb_paths.get(i, ""),
                pdb_ids.get(i, "decoy"),
                seed=seed + epoch * 1000 + i,
                esm_dim=esm_dim,
                rnafm_dim=rnafm_dim,
                esm_emb=None,
                rnafm_emb=None,
            )
            pos = _graph_to_device(pos_g)
            neg = _graph_to_device(neg_g)

            score_pos = _interface_score(model, pos, normalized=normalized_scores)
            score_neg = _interface_score(model, neg, normalized=normalized_scores)

            loss = F.relu(margin + score_pos - score_neg)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.detach())
            n_steps += 1

        if boltz_groups:
            random.shuffle(boltz_groups)
            for group in boltz_groups:
                boltz_loss = _boltz_group_loss(
                    model, group, ood_graphs, margin,
                    normalized=normalized_scores,
                )
                if boltz_loss.numel() == 0:
                    continue
                weighted = boltz_weight * boltz_loss
                opt.zero_grad()
                weighted.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total_loss += float(weighted.detach())
                n_steps += 1

        if af3_groups:
            random.shuffle(af3_groups)
            for group in af3_groups:
                af3_loss = _boltz_group_loss(
                    model, group, ood_graphs, margin,
                    normalized=normalized_scores,
                )
                if af3_loss.numel() == 0:
                    continue
                weighted = af3_weight * af3_loss
                opt.zero_grad()
                weighted.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total_loss += float(weighted.detach())
                n_steps += 1

        if foldbench_groups:
            random.shuffle(foldbench_groups)
            for group in foldbench_groups:
                fb_loss = _boltz_group_loss(
                    model, group, ood_graphs, margin,
                    normalized=normalized_scores,
                )
                if fb_loss.numel() == 0:
                    continue
                weighted = foldbench_weight * fb_loss
                opt.zero_grad()
                weighted.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total_loss += float(weighted.detach())
                n_steps += 1

        if epoch % eval_every == 0 or epoch == epochs:
            msg = f"  epoch {epoch:3d}/{epochs}  loss={total_loss/max(n_steps,1):.4f}"
            if ood_groups:
                ok, tot = _eval_boltz_partner_accuracy(
                    model, ood_groups, ood_graphs,
                    normalized=normalized_scores,
                )
                msg += f"  ood_partner_acc={ok}/{tot}"
            print(msg)

    out = Path(output_path) if output_path else default_interface_pretrain_path()
    torch.save({
        "model_state": model.state_dict(),
        "protein_node_dim": protein_node_dim,
        "rna_node_dim": rna_node_dim,
        "hidden_dim": 192,
        "n_layers": 4,
        "edge_dim": EDGE_DIM,
        "physics_summary_dim": PHYSICS_SUMMARY_DIM,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "esm_dim": esm_dim,
        "rnafm_dim": rnafm_dim,
        "interface_head_trained": True,
    }, out)
    print(f"Saved interface-head checkpoint to {out}")


def main():
    ap = argparse.ArgumentParser(description="Contrastive interface-head training")
    ap.add_argument(
        "--max-entries", type=int, default=None,
        help="Cap unique PDB structures (default: all merged ProNAB+NABE)",
    )
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--esm", action="store_true", default=True)
    ap.add_argument("--no-esm", action="store_false", dest="esm")
    ap.add_argument("--rnafm", action="store_true", default=True)
    ap.add_argument("--no-rnafm", action="store_false", dest="rnafm")
    ap.add_argument("--pronab-only", action="store_true",
                    help="Exclude NABE-only and literature supplements")
    ap.add_argument("--train-encoder", action="store_true",
                    help="Unfreeze full GT encoder during contrastive training")
    ap.add_argument(
        "--boltz-bundle",
        default=None,
        help="Boltz benchmark dir (e.g. boltz_test_100) for OOD partner ranking loss",
    )
    ap.add_argument("--boltz-weight", type=float, default=3.0)
    ap.add_argument(
        "--boltz-partner-groups",
        nargs="+",
        default=None,
        help="Train Boltz contrastive loss only on these partner groups (e.g. ms2)",
    )
    ap.add_argument(
        "--af3-decoy-bundle",
        default=None,
        help="Bundle with manifest + af3_predictions/ for AF3-structure contrastive training",
    )
    ap.add_argument(
        "--af3-predictions-subdir",
        default="af3_predictions",
        help="Subdir under bundle with AF3 Server zips per manifest job",
    )
    ap.add_argument("--af3-weight", type=float, default=6.0)
    ap.add_argument(
        "--af3-partner-groups",
        nargs="+",
        default=None,
        help="AF3 contrastive loss only on these partner groups",
    )
    ap.add_argument(
        "--foldbench-predictions",
        default=None,
        help="Dir with FoldBench AF3/Boltz prediction zips or mmCIF files",
    )
    ap.add_argument(
        "--foldbench-eval-csv",
        default=None,
        help="FoldBench evaluation CSV with dockq column (optional filter)",
    )
    ap.add_argument("--foldbench-dockq-threshold", type=float, default=0.23)
    ap.add_argument("--foldbench-weight", type=float, default=4.0)
    ap.add_argument("--foldbench-max-targets", type=int, default=None)
    ap.add_argument(
        "--all-crystal",
        action="store_true",
        help="Use all unique PDBs from merged training corpus (not capped at 30)",
    )
    ap.add_argument(
        "--max-crystal-entries",
        type=int,
        default=30,
        help="Cap crystal PDBs (ignored when --all-crystal is set)",
    )
    ap.add_argument("--no-normalized-scores", action="store_true",
                    help="Train on raw interface scores (not recommended)")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument(
        "--output",
        default=None,
        help=f"Output path (default: {default_interface_pretrain_path()})",
    )
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    crystal_cap = None if args.all_crystal else args.max_crystal_entries
    train_interface_head(
        max_entries=args.max_entries,
        epochs=args.epochs,
        lr=args.lr,
        margin=args.margin,
        use_esm=args.esm,
        use_rnafm=args.rnafm,
        output_path=args.output,
        seed=args.seed,
        pronab_only=args.pronab_only,
        train_encoder=args.train_encoder,
        boltz_bundle=args.boltz_bundle,
        boltz_weight=args.boltz_weight,
        boltz_partner_groups=args.boltz_partner_groups,
        af3_decoy_bundle=args.af3_decoy_bundle,
        af3_predictions_subdir=args.af3_predictions_subdir,
        af3_weight=args.af3_weight,
        af3_partner_groups=args.af3_partner_groups,
        foldbench_predictions=args.foldbench_predictions,
        foldbench_eval_csv=args.foldbench_eval_csv,
        foldbench_dockq_threshold=args.foldbench_dockq_threshold,
        foldbench_weight=args.foldbench_weight,
        foldbench_max_targets=args.foldbench_max_targets,
        max_crystal_entries=crystal_cap,
        normalized_scores=not args.no_normalized_scores,
        eval_every=args.eval_every,
    )


if __name__ == "__main__":
    main()
