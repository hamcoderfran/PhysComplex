"""
Training script for the Physics-Informed Graph Transformer (PhysGT).

Workflow
--------
1. Fetch merged training data (fetch_training_data: ProNAB + Nabe + literature)
2. Download PDB structures and build minimised WT + mutant PDB files
3. Pre-compute ESM-2 and RNA-FM embeddings (cached to disk)
4. Build InterfaceGraph objects for each WT/mutant pair
5. Train PhysicsInformedGT with leave-one-complex-out (LOCO) CV
   grouped by PDB ID to avoid data leakage
6. Report per-fold and aggregate Pearson / Spearman r
7. Save the best checkpoint to gt_checkpoint.pt

Usage
-----
    python -m physrna_filter.validation.train_gt [options]

Options
    --max-entries N      cap number of ProNAB entries (default: all)
    --epochs N           training epochs per fold (default: 150)
    --hidden-dim N       GT hidden dimension (default: 192)
    --n-layers N         number of TransformerConv layers (default: 4)
    --lr FLOAT           learning rate (default: 3e-4)
    --no-esm             skip ESM-2 embeddings (use one-hot only)
    --no-rnafm           skip RNA-FM embeddings (use one-hot only)
    --output PATH        checkpoint save path (default: gt_checkpoint.pt)
    --results-csv PATH   per-entry prediction CSV (default: gt_results.csv)
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr

from ..data.mutation_target import resolve_mutation_target, validate_mutation_target_for_structure
from ..data.fetch_training_data import fetch_training_data
from ..data.fetch_pdb import download_pdb
from ..structure.mutate import (
    introduce_mutation,
    find_mutation_chain,
    find_mutation_rna_chain,
    prepare_fixed_structure,
    resolve_rna_mutation_chain,
)
from ..analysis.graph_features import build_interface_graph, InterfaceGraph
from ..analysis.gt_model import build_model
from ..analysis.gt_constants import CHECKPOINT_SCHEMA_VERSION, EDGE_DIM, PHYSICS_SUMMARY_DIM
from ..analysis.esm_embeddings import (
    get_esm2_embeddings_from_structure, ESM_DIM,
)
from ..analysis.rnafm_embeddings import (
    get_rnafm_embeddings_from_structure,
    effective_rnafm_feature_dim,
    prepare_rnafm_for_training,
)

MUTANT_DIR = Path(__file__).parent / "mutant_structures"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# (wt_graph, mut_graph, ddg, pdb_id, mutation, source)
DatasetRow = tuple[InterfaceGraph, InterfaceGraph, float, str, str, str]


def _normalize_dataset_row(
    row: tuple,
) -> DatasetRow:
    """Support legacy 5-tuple graph caches (source defaults to pronab)."""
    if len(row) == 5:
        wt, mut, ddg, pid, mutation = row
        return wt, mut, ddg, pid, mutation, "pronab"
    return row


def _row_source(row: pd.Series) -> str:
    return str(row.get("source", "pronab"))


def _graph_to_device(g: InterfaceGraph) -> InterfaceGraph:
    return InterfaceGraph(
        x_protein       = g.x_protein.to(DEVICE),
        x_rna           = g.x_rna.to(DEVICE),
        edge_index      = g.edge_index.to(DEVICE),
        edge_attr       = g.edge_attr.to(DEVICE),
        node_types      = g.node_types.to(DEVICE),
        mutation_node_idx = g.mutation_node_idx,
        protein_residues  = g.protein_residues,
        rna_residues      = g.rna_residues,
    )


def _build_entry_graphs(
    row:              pd.Series,
    use_esm:          bool,
    use_rnafm:        bool,
    esm_dim:          int,
    rnafm_dim:        int,
    minimize_structures: bool,
    reuse_wt_esm:     bool,
    fast_mutations:   bool,
) -> tuple[InterfaceGraph, InterfaceGraph]:
    """
    Downloads structures, builds minimised WT and mutant PDB files,
    computes embeddings, and returns (wt_graph, mut_graph).
    Raises on any unrecoverable error.
    """
    from ..structure.parse_complex import parse_complex

    pdb_id   = row["pdb_id"]
    pdb_path = download_pdb(pdb_id)
    if pdb_path is None:
        raise RuntimeError(f"Could not download {pdb_id}")

    target = resolve_mutation_target(row)
    chain_id = target.chain_id
    position = target.position
    wt_aa, mut_aa = target.wt, target.mut

    parsed_source = parse_complex(pdb_path)
    protein_chain_ids = {ch.id for ch in parsed_source.protein_chains}
    rna_chain_ids = {ch.id for ch in parsed_source.rna_chains}
    validate_mutation_target_for_structure(
        target,
        protein_chain_ids=protein_chain_ids,
        rna_chain_ids=rna_chain_ids,
    )

    if target.kind == "rna":
        chain_id = resolve_rna_mutation_chain(
            pdb_path,
            chain_id,
            position,
            wt_aa,
            mut_aa,
            protein_chain_ids=protein_chain_ids,
            rna_chain_ids=rna_chain_ids,
        )
    elif not chain_id:
        chain_id = find_mutation_chain(pdb_path, position, wt_aa, alt_aa=mut_aa)

    wt_fixed_path = str(MUTANT_DIR / f"{pdb_id.lower()}_WT{position}.pdb")
    mut_path      = str(MUTANT_DIR / f"{pdb_id.lower()}_{row['mutation']}.pdb")

    if fast_mutations:
        from ..structure.mutate import _mutate_rna_simple, _mutate_simple
        os.makedirs(os.path.dirname(wt_fixed_path) or ".", exist_ok=True)
        shutil.copy(pdb_path, wt_fixed_path)
        if target.kind == "rna":
            mut_path = _mutate_rna_simple(pdb_path, chain_id, position, mut_aa, mut_path)
        else:
            mut_path = _mutate_simple(pdb_path, chain_id, position, mut_aa, mut_path)
    else:
        if target.kind == "rna":
            raise NotImplementedError(
                "Full PDBFixer RNA mutation path not implemented; use fast_mutations"
            )
        prepare_fixed_structure(
            pdb_path=pdb_path, chain_id=chain_id, position=position,
            aa=wt_aa, output_path=wt_fixed_path, minimize=minimize_structures,
        )
        introduce_mutation(
            pdb_path=pdb_path, chain_id=chain_id, position=position,
            mutant_aa=mut_aa, output_path=mut_path, minimize=minimize_structures,
        )

    # ── compute embeddings ──────────────────────────────────────────────────
    wt_parsed  = parse_complex(wt_fixed_path)
    mut_parsed = parse_complex(mut_path)

    from ..structure.partner_selection import primary_protein_chain_id, select_partner_pair
    from ..analysis.contact_score import _select_partner_rna_chains

    if target.kind == "rna":
        protein_chains, rna_chains = select_partner_pair(
            wt_parsed.protein_chains, wt_parsed.rna_chains, rna_chain_id=chain_id,
        )
        protein_chain_id = (
            protein_chains[0].id if protein_chains
            else primary_protein_chain_id(wt_parsed.protein_chains, wt_parsed.rna_chains)
        )
    else:
        protein_chain_id = chain_id
        rna_chains = _select_partner_rna_chains(
            wt_parsed.protein_chains, wt_parsed.rna_chains, chain_id
        )

    wt_esm = get_esm2_embeddings_from_structure(
        pdb_id, protein_chain_id, wt_parsed.protein_chains
    ) if use_esm else None

    if use_esm and reuse_wt_esm and target.kind != "rna":
        mut_esm = wt_esm
    else:
        mut_esm = get_esm2_embeddings_from_structure(
            f"{pdb_id}_{row['mutation']}", protein_chain_id, mut_parsed.protein_chains
        ) if use_esm else None

    rna_chain_id = rna_chains[0].id if rna_chains else chain_id

    wt_rnafm = get_rnafm_embeddings_from_structure(
        pdb_id, rna_chain_id or "X", rna_chains
    ) if (use_rnafm and rna_chain_id) else None

    mut_rna_chains = (
        _select_partner_rna_chains(mut_parsed.protein_chains, mut_parsed.rna_chains, protein_chain_id)
        if target.kind != "rna"
        else select_partner_pair(
            mut_parsed.protein_chains, mut_parsed.rna_chains, rna_chain_id=chain_id,
        )[1]
    )
    if target.kind == "rna":
        mut_rnafm = get_rnafm_embeddings_from_structure(
            f"{pdb_id}_{row['mutation']}", rna_chain_id or chain_id, mut_rna_chains
        ) if (use_rnafm and rna_chain_id) else None
    else:
        mut_rnafm = wt_rnafm

    graph_kwargs = (
        {"mutation_on": "rna", "partner_chain_id": protein_chain_id}
        if target.kind == "rna"
        else {"mutation_on": "protein"}
    )

    # ── build graphs ────────────────────────────────────────────────────────
    wt_graph = build_interface_graph(
        wt_fixed_path, chain_id, position,
        pdb_id=pdb_id,
        esm_embeddings=wt_esm,
        rnafm_embeddings=wt_rnafm,
        esm_dim=esm_dim,
        rnafm_dim=rnafm_dim,
        **graph_kwargs,
    )
    mut_graph = build_interface_graph(
        mut_path, chain_id, position,
        pdb_id=pdb_id + "_mut",
        esm_embeddings=mut_esm,
        rnafm_embeddings=mut_rnafm,
        esm_dim=esm_dim,
        rnafm_dim=rnafm_dim,
        **graph_kwargs,
    )

    return wt_graph, mut_graph


def _save_graph_cache_payload(
    cache_path: Path,
    dataset: list[DatasetRow],
    failures: list[dict],
    *,
    use_esm: bool,
    use_rnafm: bool,
    esm_dim: int,
    rnafm_dim: int,
    minimize_structures: bool,
    reuse_wt_esm: bool,
    fast_mutations: bool,
    n_requested: int,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "dataset": dataset,
            "failures": failures,
            "use_esm": use_esm,
            "use_rnafm": use_rnafm,
            "esm_dim": esm_dim,
            "rnafm_dim": rnafm_dim,
            "minimize_structures": minimize_structures,
            "reuse_wt_esm": reuse_wt_esm,
            "fast_mutations": fast_mutations,
            "n_requested": n_requested,
            "n_built": len(dataset),
        },
        cache_path,
    )


def _cache_config_matches(
    payload: dict,
    *,
    use_esm: bool,
    use_rnafm: bool,
    esm_dim: int,
    rnafm_dim: int,
    minimize_structures: bool,
    reuse_wt_esm: bool,
    fast_mutations: bool,
    n_requested: int,
) -> bool:
    return (
        payload.get("use_esm") == use_esm
        and payload.get("use_rnafm") == use_rnafm
        and payload.get("esm_dim") == esm_dim
        and payload.get("rnafm_dim") == rnafm_dim
        and payload.get("minimize_structures") == minimize_structures
        and payload.get("reuse_wt_esm") == reuse_wt_esm
        and payload.get("fast_mutations") == fast_mutations
        and payload.get("n_requested") == n_requested
    )


def _build_dataset(
    entries_df: pd.DataFrame,
    use_esm: bool,
    use_rnafm: bool,
    esm_dim: int,
    rnafm_dim: int,
    minimize_structures: bool,
    reuse_wt_esm: bool,
    fast_mutations: bool,
    *,
    cache_path: Path | None = None,
    cache_every: int = 25,
    start_index: int = 0,
    dataset: list[DatasetRow] | None = None,
    failures: list[dict] | None = None,
) -> tuple[list[DatasetRow], list[dict]]:
    dataset = list(dataset or [])
    failures = list(failures or [])

    for i, (_, row) in enumerate(entries_df.iterrows()):
        if i < start_index:
            continue
        tag = f"[{i+1}/{len(entries_df)}] {row['pdb_id']} {row['mutation']}"
        print(f"\n{tag}")
        try:
            wt_g, mut_g = _build_entry_graphs(
                row, use_esm, use_rnafm, esm_dim, rnafm_dim,
                minimize_structures, reuse_wt_esm, fast_mutations
            )
            dataset.append((
                wt_g, mut_g, float(row["ddg"]),
                row["pdb_id"], row["mutation"], _row_source(row),
            ))
        except Exception as e:
            msg = str(e).splitlines()[0][:150]
            print(f"  SKIP ({type(e).__name__}): {msg}")
            failures.append({
                "pdb_id": row["pdb_id"],
                "mutation": row["mutation"],
                "source": _row_source(row),
                "error": msg,
            })

        if cache_path and (i + 1) % cache_every == 0:
            _save_graph_cache_payload(
                cache_path,
                dataset,
                failures,
                use_esm=use_esm,
                use_rnafm=use_rnafm,
                esm_dim=esm_dim,
                rnafm_dim=rnafm_dim,
                minimize_structures=minimize_structures,
                reuse_wt_esm=reuse_wt_esm,
                fast_mutations=fast_mutations,
                n_requested=len(entries_df),
            )
            print(f"  checkpointed graph cache ({len(dataset)} pairs) -> {cache_path}")

    return dataset, failures


def _load_or_build_dataset(
    entries_df: pd.DataFrame,
    use_esm: bool,
    use_rnafm: bool,
    esm_dim: int,
    rnafm_dim: int,
    minimize_structures: bool,
    reuse_wt_esm: bool,
    fast_mutations: bool,
    graph_cache: str | None,
    rebuild_cache: bool,
) -> tuple[list[tuple[InterfaceGraph, InterfaceGraph, float, str, str]], list[dict]]:
    cache_path = Path(graph_cache) if graph_cache else None
    dataset: list[DatasetRow] = []
    failures: list[dict] = []
    start_index = 0

    if cache_path and cache_path.exists() and not rebuild_cache:
        print(f"Loading graph cache from {cache_path}")
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if _cache_config_matches(
            payload,
            use_esm=use_esm,
            use_rnafm=use_rnafm,
            esm_dim=esm_dim,
            rnafm_dim=rnafm_dim,
            minimize_structures=minimize_structures,
            reuse_wt_esm=reuse_wt_esm,
            fast_mutations=fast_mutations,
            n_requested=len(entries_df),
        ):
            dataset = [_normalize_dataset_row(r) for r in payload["dataset"]]
            failures = list(payload.get("failures", []))
            start_index = len(dataset) + len(failures)
            if start_index >= len(entries_df):
                print(f"Graph cache complete ({len(dataset)} pairs, {len(failures)} failures)")
                return dataset, failures
            print(
                f"Resuming graph build from row {start_index + 1}/{len(entries_df)} "
                f"({len(dataset)} pairs cached)"
            )
        else:
            print("Graph cache config mismatch; rebuilding from scratch")
            dataset = []
            failures = []
            start_index = 0

    dataset, failures = _build_dataset(
        entries_df, use_esm, use_rnafm, esm_dim, rnafm_dim,
        minimize_structures, reuse_wt_esm, fast_mutations,
        cache_path=cache_path,
        start_index=start_index,
        dataset=dataset,
        failures=failures,
    )
    if cache_path:
        _save_graph_cache_payload(
            cache_path,
            dataset,
            failures,
            use_esm=use_esm,
            use_rnafm=use_rnafm,
            esm_dim=esm_dim,
            rnafm_dim=rnafm_dim,
            minimize_structures=minimize_structures,
            reuse_wt_esm=reuse_wt_esm,
            fast_mutations=fast_mutations,
            n_requested=len(entries_df),
        )
        print(f"Saved graph cache to {cache_path}")

    return dataset, failures


def _train_one_fold(
    model:        nn.Module,
    train_data:   list[tuple[InterfaceGraph, InterfaceGraph, float]],
    val_data:     list[tuple[InterfaceGraph, InterfaceGraph, float]] | None,
    epochs:       int,
    lr:           float,
    target_mean:  float = 0.0,
    target_std:   float = 1.0,
) -> None:
    """
    Trains model for `epochs` epochs.

    If val_data is supplied, the best checkpoint is selected by validation loss
    on training-only holdout complexes and restored before returning.  Held-out
    LOCO test labels must not be passed here.
    """
    model = model.to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.HuberLoss(delta=1.0)   # robust to outliers vs MSE

    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        np.random.shuffle(train_data)   # type: ignore[arg-type]
        total_loss = 0.0

        for wt_g, mut_g, target in train_data:
            wt_g  = _graph_to_device(wt_g)
            mut_g = _graph_to_device(mut_g)
            y     = torch.tensor(
                (target - target_mean) / target_std,
                dtype=torch.float32,
                device=DEVICE,
            )

            opt.zero_grad()
            pred = model(
                wt_x_protein=wt_g.x_protein, wt_x_rna=wt_g.x_rna,
                wt_edge_index=wt_g.edge_index, wt_edge_attr=wt_g.edge_attr,
                wt_mutation_idx=wt_g.mutation_node_idx,
                wt_n_prot=wt_g.x_protein.shape[0],
                mut_x_protein=mut_g.x_protein, mut_x_rna=mut_g.x_rna,
                mut_edge_index=mut_g.edge_index, mut_edge_attr=mut_g.edge_attr,
                mut_mutation_idx=mut_g.mutation_node_idx,
                mut_n_prot=mut_g.x_protein.shape[0],
            )
            loss = loss_fn(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        sched.step()

        # Validate every 10 epochs
        if val_data and (epoch % 10 == 0 or epoch == epochs):
            val_loss = _evaluate_loss(
                model, val_data, loss_fn, target_mean, target_std
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
            print(
                f"  epoch {epoch:4d}/{epochs}  "
                f"train_loss={total_loss/max(len(train_data),1):.4f}  "
                f"val_loss={val_loss:.4f}",
                flush=True,
            )
        elif epoch % 25 == 0:
            print(
                f"  epoch {epoch:4d}/{epochs}  "
                f"train_loss={total_loss/max(len(train_data),1):.4f}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)


def _evaluate_loss(
    model: nn.Module,
    data: list[tuple[InterfaceGraph, InterfaceGraph, float]],
    loss_fn: nn.Module,
    target_mean: float = 0.0,
    target_std: float = 1.0,
) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for wt_g, mut_g, target in data:
            wt_g  = _graph_to_device(wt_g)
            mut_g = _graph_to_device(mut_g)
            y     = torch.tensor(
                (target - target_mean) / target_std,
                dtype=torch.float32,
                device=DEVICE,
            )
            pred  = model(
                wt_x_protein=wt_g.x_protein, wt_x_rna=wt_g.x_rna,
                wt_edge_index=wt_g.edge_index, wt_edge_attr=wt_g.edge_attr,
                wt_mutation_idx=wt_g.mutation_node_idx,
                wt_n_prot=wt_g.x_protein.shape[0],
                mut_x_protein=mut_g.x_protein, mut_x_rna=mut_g.x_rna,
                mut_edge_index=mut_g.edge_index, mut_edge_attr=mut_g.edge_attr,
                mut_mutation_idx=mut_g.mutation_node_idx,
                mut_n_prot=mut_g.x_protein.shape[0],
            )
            total += float(loss_fn(pred, y).detach().cpu())
    return total / max(len(data), 1)


def _split_train_validation_by_pdb(
    train_rows: list[DatasetRow],
    val_fraction: float,
    seed: int,
) -> tuple[
    list[tuple[InterfaceGraph, InterfaceGraph, float]],
    list[tuple[InterfaceGraph, InterfaceGraph, float]],
]:
    """Select internal validation complexes from the training side only."""
    train_rows = [_normalize_dataset_row(r) for r in train_rows]
    unique_pdbs = list(dict.fromkeys(pid for _, _, _, pid, _, _ in train_rows))
    if len(unique_pdbs) < 2:
        return [(w, m, t) for w, m, t, _, _, _ in train_rows], []

    rng = np.random.default_rng(seed)
    shuffled = list(unique_pdbs)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    n_val = min(n_val, len(shuffled) - 1)
    val_pdbs = set(shuffled[:n_val])

    fold_train = [
        (w, m, t) for w, m, t, pid, _, _ in train_rows if pid not in val_pdbs
    ]
    fold_val = [
        (w, m, t) for w, m, t, pid, _, _ in train_rows if pid in val_pdbs
    ]
    return fold_train, fold_val


def _split_holdout_by_pdb(
    dataset: list[DatasetRow],
    test_fraction: float,
    val_fraction: float,
    seed: int,
) -> tuple[list[DatasetRow], list[DatasetRow], list[DatasetRow]]:
    dataset = [_normalize_dataset_row(r) for r in dataset]
    unique_pdbs = list(dict.fromkeys(pid for _, _, _, pid, _, _ in dataset))
    if len(unique_pdbs) < 3:
        raise ValueError("Need at least 3 complexes for holdout evaluation")

    rng = np.random.default_rng(seed)
    shuffled = list(unique_pdbs)
    rng.shuffle(shuffled)

    n_test = max(1, int(round(len(shuffled) * test_fraction)))
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    if n_test + n_val >= len(shuffled):
        n_test = max(1, min(n_test, len(shuffled) - 2))
        n_val = 1

    test_pdbs = set(shuffled[:n_test])
    val_pdbs = set(shuffled[n_test:n_test + n_val])

    train = [row for row in dataset if row[3] not in test_pdbs and row[3] not in val_pdbs]
    val = [row for row in dataset if row[3] in val_pdbs]
    test = [row for row in dataset if row[3] in test_pdbs]
    return train, val, test


def _target_stats(data: list[tuple[InterfaceGraph, InterfaceGraph, float]]) -> tuple[float, float]:
    targets = np.array([t for _, _, t in data], dtype=float)
    mean = float(targets.mean())
    std = float(targets.std())
    return mean, max(std, 1e-6)


def _predict(
    model: nn.Module,
    data:  list[tuple[InterfaceGraph, InterfaceGraph, float]],
    target_mean: float = 0.0,
    target_std: float = 1.0,
) -> tuple[list[float], list[float]]:
    model.eval()
    targets, preds = [], []
    with torch.no_grad():
        for wt_g, mut_g, target in data:
            wt_g  = _graph_to_device(wt_g)
            mut_g = _graph_to_device(mut_g)
            pred  = model(
                wt_x_protein=wt_g.x_protein, wt_x_rna=wt_g.x_rna,
                wt_edge_index=wt_g.edge_index, wt_edge_attr=wt_g.edge_attr,
                wt_mutation_idx=wt_g.mutation_node_idx,
                wt_n_prot=wt_g.x_protein.shape[0],
                mut_x_protein=mut_g.x_protein, mut_x_rna=mut_g.x_rna,
                mut_edge_index=mut_g.edge_index, mut_edge_attr=mut_g.edge_attr,
                mut_mutation_idx=mut_g.mutation_node_idx,
                mut_n_prot=mut_g.x_protein.shape[0],
            )
            targets.append(target)
            preds.append(float(pred.cpu()) * target_std + target_mean)
    return targets, preds


def _per_source_metrics(
    targets: list[float],
    preds: list[float],
    sources: list[str],
) -> None:
    df = pd.DataFrame({
        "experimental_ddg": targets,
        "gt_pred_ddg": preds,
        "source": sources,
    })
    print("\nPer-source metrics:")
    for source, grp in df.groupby("source"):
        if len(grp) < 3:
            print(f"  {source}: n={len(grp)} (too few for correlation)")
            continue
        r_p, _ = pearsonr(grp["experimental_ddg"], grp["gt_pred_ddg"])
        r_s, _ = spearmanr(grp["experimental_ddg"], grp["gt_pred_ddg"])
        print(f"  {source}: n={len(grp)}  Pearson r={r_p:+.3f}  Spearman r={r_s:+.3f}")


def run_gt_training(
    max_entries:    int | None = None,
    epochs:         int        = 150,
    hidden_dim:     int        = 192,
    n_layers:       int        = 4,
    lr:             float      = 3e-4,
    use_esm:        bool       = True,
    use_rnafm:      bool       = True,
    output_path:    str        = "gt_checkpoint.pt",
    results_csv:    str        = "gt_results.csv",
    n_folds:        int | None = None,   # None = full LOCO; int = max folds
    seed:           int        = 7,
    cv_mode:        str        = "loco",
    test_fraction:  float      = 0.15,
    val_fraction:   float      = 0.15,
    graph_cache:    str | None = None,
    rebuild_cache:  bool       = False,
    minimize_structures: bool  = True,
    reuse_wt_esm:   bool       = False,
    fast_mutations: bool       = False,
    include_nabe:     bool       = True,
    include_literature: bool     = True,
    pronab_only:      bool       = False,
) -> pd.DataFrame:
    """
    End-to-end GT training on merged ProNAB (+ optional Nabe/literature).
    Returns a DataFrame of predictions.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    MUTANT_DIR.mkdir(parents=True, exist_ok=True)

    from .gt_readiness import run_gt_readiness, print_readiness_report
    readiness = run_gt_readiness(use_esm=use_esm, use_rnafm=use_rnafm)
    print_readiness_report(readiness)
    if use_rnafm and not any(c["ok"] for c in readiness["checks"] if c["name"] == "rnafm"):
        raise RuntimeError("RNA-FM weights required for production PhysGT training.")

    esm_dim   = ESM_DIM if use_esm else 0
    if use_rnafm:
        prepare_rnafm_for_training(use_rnafm)
    rnafm_dim = effective_rnafm_feature_dim(use_rnafm)

    # protein node dim = 20 (one_hot) + 3 (phys) + 1 (is_site) + esm_dim
    protein_node_dim = 24 + esm_dim
    # RNA node dim = 8 (one_hot) + rnafm_dim
    rna_node_dim     = 8  + rnafm_dim

    print(f"Node dims — protein: {protein_node_dim}  RNA: {rna_node_dim}")
    print(f"Device: {DEVICE}  ESM-2: {use_esm}  RNA-FM: {use_rnafm}")

    pronab_df = fetch_training_data(
        include_nabe=include_nabe and not pronab_only,
        include_literature=include_literature and not pronab_only,
    )
    if max_entries:
        pronab_df = pronab_df.head(max_entries)

    dataset, failures = _load_or_build_dataset(
        entries_df=pronab_df,
        use_esm=use_esm,
        use_rnafm=use_rnafm,
        esm_dim=esm_dim,
        rnafm_dim=rnafm_dim,
        minimize_structures=minimize_structures,
        reuse_wt_esm=reuse_wt_esm,
        fast_mutations=fast_mutations,
        graph_cache=graph_cache,
        rebuild_cache=rebuild_cache,
    )

    print(f"\nBuilt {len(dataset)} graphs ({len(failures)} failures)")

    if not dataset:
        raise RuntimeError("No entries could be processed — check structure downloads.")

    if cv_mode == "holdout":
        train_rows, val_rows, test_rows = _split_holdout_by_pdb(
            dataset, test_fraction=test_fraction, val_fraction=val_fraction, seed=seed
        )
        model = build_model(
            protein_node_dim=protein_node_dim,
            rna_node_dim=rna_node_dim,
            edge_dim=EDGE_DIM,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )
        train_data = [(w, m, t) for w, m, t, _, _, _ in train_rows]
        val_data = [(w, m, t) for w, m, t, _, _, _ in val_rows]
        test_data = [(w, m, t) for w, m, t, _, _, _ in test_rows]
        target_mean, target_std = _target_stats(train_data)

        print(
            f"\nRunning holdout evaluation: train={len(train_data)} "
            f"val={len(val_data)} test={len(test_data)}"
        )
        _train_one_fold(
            model, train_data, val_data, epochs, lr, target_mean, target_std
        )
        all_targets, all_preds = _predict(model, test_data, target_mean, target_std)
        all_pdb_ids = [pid for _, _, _, pid, _, _ in test_rows]
        all_mutations = [mut for _, _, _, _, mut, _ in test_rows]
        all_sources = [src for _, _, _, _, _, src in test_rows]
    else:
        all_targets, all_preds, all_pdb_ids, all_mutations, all_sources = _run_loco_cv(
            dataset=dataset,
            protein_node_dim=protein_node_dim,
            rna_node_dim=rna_node_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            epochs=epochs,
            lr=lr,
            n_folds=n_folds,
            seed=seed,
        )

    # ── aggregate results ───────────────────────────────────────────────────
    results_df = pd.DataFrame({
        "pdb_id":         all_pdb_ids,
        "mutation":       all_mutations,
        "source":         all_sources,
        "experimental_ddg": all_targets,
        "gt_pred_ddg":    all_preds,
    })

    if len(all_targets) >= 5:
        r_p, p_p = pearsonr( all_targets, all_preds)
        r_s, _   = spearmanr(all_targets, all_preds)
        print("\n" + "=" * 55)
        print(f"PhysGT {cv_mode.upper()} evaluation  n={len(all_targets)}")
        print(f"  Pearson  r = {r_p:+.3f}  (p={p_p:.4f})")
        print(f"  Spearman r = {r_s:+.3f}")
        print("=" * 55)
        _per_source_metrics(all_targets, all_preds, all_sources)

    results_df.to_csv(results_csv, index=False)
    print(f"Saved predictions to {results_csv}")

    # Save a final model trained on ALL data for deployment
    final_model = build_model(
        protein_node_dim=protein_node_dim,
        rna_node_dim=rna_node_dim,
        edge_dim=EDGE_DIM,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
    )
    all_data = [(w, m, t) for w, m, t, _, _, _ in dataset]
    final_mean, final_std = _target_stats(all_data)
    _train_one_fold(final_model, all_data, None, epochs, lr, final_mean, final_std)
    torch.save({
        "model_state": final_model.state_dict(),
        "protein_node_dim": protein_node_dim,
        "rna_node_dim":     rna_node_dim,
        "hidden_dim":       hidden_dim,
        "n_layers":         n_layers,
        "edge_dim":         EDGE_DIM,
        "physics_summary_dim": PHYSICS_SUMMARY_DIM,
        "schema_version":   CHECKPOINT_SCHEMA_VERSION,
        "esm_dim":          esm_dim,
        "rnafm_dim":        rnafm_dim,
        "target_mean":      final_mean,
        "target_std":       final_std,
        "interface_head_trained": False,
        "training_sources": sorted(set(pronab_df["source"].unique())),
        "n_training_entries": len(pronab_df),
    }, output_path)
    print(f"Saved final model to {output_path}")

    return results_df


def _run_loco_cv(
    dataset: list[DatasetRow],
    protein_node_dim: int,
    rna_node_dim: int,
    hidden_dim: int,
    n_layers: int,
    epochs: int,
    lr: float,
    n_folds: int | None,
    seed: int,
) -> tuple[list[float], list[float], list[str], list[str], list[str]]:
    dataset = [_normalize_dataset_row(r) for r in dataset]
    # ── LOCO cross-validation ───────────────────────────────────────────────
    pdb_ids = [d[3] for d in dataset]
    unique_pdbs = list(dict.fromkeys(pdb_ids))   # preserve order, unique
    if n_folds:
        unique_pdbs = unique_pdbs[:n_folds]

    all_targets: list[float] = []
    all_preds:   list[float] = []
    all_pdb_ids: list[str]   = []
    all_mutations: list[str] = []
    all_sources: list[str] = []

    print(f"\nRunning LOCO CV over {len(unique_pdbs)} complexes ...")

    for fold_i, test_pdb in enumerate(unique_pdbs):
        train = [
            (wt, mu, t, pid, mut, src)
            for wt, mu, t, pid, mut, src in dataset if pid != test_pdb
        ]
        test  = [
            (wt, mu, t, pid, mut, src)
            for wt, mu, t, pid, mut, src in dataset if pid == test_pdb
        ]

        if not train or not test:
            continue

        # Fresh model for each fold (LOCO = independent test sets)
        model = build_model(
            protein_node_dim=protein_node_dim,
            rna_node_dim=rna_node_dim,
            edge_dim=EDGE_DIM,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )

        train_data, val_data = _split_train_validation_by_pdb(
            train, val_fraction=0.15, seed=seed + fold_i
        )
        test_data  = [(w, m, t) for w, m, t, _, _, _ in test]

        t0 = time.time()
        target_mean, target_std = _target_stats(train_data)
        _train_one_fold(
            model, train_data, val_data, epochs, lr, target_mean, target_std
        )
        fold_targets, fold_preds = _predict(model, test_data, target_mean, target_std)

        elapsed = time.time() - t0
        if len(fold_targets) >= 2:
            r, _ = pearsonr(fold_targets, fold_preds)
            print(f"  Fold {fold_i+1:3d}/{len(unique_pdbs)} [{test_pdb}] "
                  f"n={len(fold_targets)}  r={r:+.3f}  ({elapsed:.0f}s)")

        all_targets.extend(fold_targets)
        all_preds.extend(fold_preds)
        all_pdb_ids.extend([test_pdb] * len(fold_targets))
        all_mutations.extend([m for _, _, _, _, m, _ in test])
        all_sources.extend([s for _, _, _, _, _, s in test])

    return all_targets, all_preds, all_pdb_ids, all_mutations, all_sources


def main():
    ap = argparse.ArgumentParser(description="Train PhysicsInformedGT on ProNAB")
    ap.add_argument("--max-entries", type=int, default=None)
    ap.add_argument("--epochs",      type=int, default=150)
    ap.add_argument("--hidden-dim",  type=int, default=192)
    ap.add_argument("--n-layers",    type=int, default=4)
    ap.add_argument("--lr",          type=float, default=3e-4)
    ap.add_argument("--no-esm",      action="store_true")
    ap.add_argument("--no-rnafm",    action="store_true")
    ap.add_argument("--pronab-only", action="store_true",
                    help="Use ProNAB scrape only (exclude Nabe and literature)")
    ap.add_argument("--no-nabe", action="store_true",
                    help="Exclude Nabe supplemental data")
    ap.add_argument("--no-literature", action="store_true",
                    help="Exclude literature-mined supplemental data")
    ap.add_argument("--n-folds",     type=int, default=None,
                    help="Cap LOCO folds (for quick smoke-test)")
    ap.add_argument("--seed",        type=int, default=7)
    ap.add_argument("--cv-mode",     choices=["loco", "holdout"], default="loco")
    ap.add_argument("--test-fraction", type=float, default=0.15)
    ap.add_argument("--val-fraction",  type=float, default=0.15)
    ap.add_argument("--graph-cache", default=None,
                    help="Path to cache built graph dataset with torch.save")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--no-minimize-structures", action="store_true",
                    help="Skip restrained minimization when building WT/mutant structures")
    ap.add_argument("--fast-mutations", action="store_true",
                    help="Use coordinate-level mutations even when PDBFixer is installed")
    ap.add_argument("--reuse-wt-esm", action="store_true",
                    help="Reuse WT ESM-2 embeddings on mutants (faster, less accurate)")
    ap.add_argument("--output",      default="gt_checkpoint.pt")
    ap.add_argument("--results-csv", default="gt_results.csv")
    args = ap.parse_args()

    run_gt_training(
        max_entries  = args.max_entries,
        epochs       = args.epochs,
        hidden_dim   = args.hidden_dim,
        n_layers     = args.n_layers,
        lr           = args.lr,
        use_esm      = not args.no_esm,
        use_rnafm    = not args.no_rnafm,
        n_folds      = args.n_folds,
        seed         = args.seed,
        cv_mode      = args.cv_mode,
        test_fraction = args.test_fraction,
        val_fraction = args.val_fraction,
        graph_cache  = args.graph_cache,
        rebuild_cache = args.rebuild_cache,
        minimize_structures = not args.no_minimize_structures,
        reuse_wt_esm = args.reuse_wt_esm,
        fast_mutations = args.fast_mutations,
        include_nabe = not args.no_nabe and not args.pronab_only,
        include_literature = not args.no_literature and not args.pronab_only,
        pronab_only = args.pronab_only,
        output_path  = args.output,
        results_csv  = args.results_csv,
    )


if __name__ == "__main__":
    main()
