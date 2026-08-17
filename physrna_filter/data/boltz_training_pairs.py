"""
Load Boltz benchmark structures as contrastive training pairs.

Each partner_group yields one native (positive) and multiple decoy structures
from ``predictions/*.cif``. Used to fine-tune PhysGT on wrong-partner OOD data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..analysis.esm_embeddings import ESM_DIM, get_esm2_embeddings_from_structure
from ..analysis.graph_features import InterfaceGraph, build_af3_interface_graph
from ..analysis.rnafm_embeddings import (
    effective_rnafm_feature_dim,
    get_rnafm_embeddings_from_structure,
)
from ..structure.parse_complex import parse_complex


@dataclass
class BoltzTrainingGroup:
    partner_group: str
    native_job: str
    native_path: str
    decoy_jobs: list[str]
    decoy_paths: list[str]


def load_boltz_training_groups(
    bundle_dir: str | Path,
    *,
    predictions_subdir: str = "predictions",
    partner_groups: list[str] | None = None,
) -> list[BoltzTrainingGroup]:
    root = Path(bundle_dir)
    manifest = pd.read_csv(root / "manifest.csv")
    pred_dir = root / predictions_subdir
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Missing predictions dir: {pred_dir}")

    allowed = {g.lower() for g in partner_groups} if partner_groups else None
    groups: list[BoltzTrainingGroup] = []
    for partner_group, sub in manifest.groupby("partner_group"):
        if allowed is not None and str(partner_group).lower() not in allowed:
            continue
        natives = sub[sub["label"] == "positive"]
        decoys = sub[sub["label"] == "negative"]
        if natives.empty or decoys.empty:
            continue
        native_row = natives.iloc[0]
        native_job = str(native_row["job_name"])
        native_cif = pred_dir / f"{native_job}.cif"
        if not native_cif.is_file():
            continue

        decoy_jobs: list[str] = []
        decoy_paths: list[str] = []
        for _, row in decoys.iterrows():
            job = str(row["job_name"])
            cif = pred_dir / f"{job}.cif"
            if cif.is_file():
                decoy_jobs.append(job)
                decoy_paths.append(str(cif))

        if decoy_jobs:
            groups.append(
                BoltzTrainingGroup(
                    partner_group=str(partner_group),
                    native_job=native_job,
                    native_path=str(native_cif),
                    decoy_jobs=decoy_jobs,
                    decoy_paths=decoy_paths,
                )
            )
    return groups


def build_boltz_interface_graph(
    cif_path: str,
    *,
    use_esm: bool = True,
    use_rnafm: bool = True,
    esm_dim: int | None = None,
    rnafm_dim: int | None = None,
    model_rank: int = 0,
) -> InterfaceGraph | None:
    """Build an interface graph from a Boltz mmCIF prediction."""
    esm_dim = esm_dim if esm_dim is not None else (ESM_DIM if use_esm else 0)
    rnafm_dim = rnafm_dim if rnafm_dim is not None else effective_rnafm_feature_dim(use_rnafm)

    parsed = parse_complex(cif_path, model_rank=model_rank)
    if not parsed.protein_chains or not parsed.rna_chains:
        return None

    pdb_id = Path(cif_path).stem
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
        cif_path,
        pdb_id=pdb_id,
        esm_embeddings=esm_emb,
        rnafm_embeddings=rnafm_emb,
        esm_dim=esm_dim,
        rnafm_dim=rnafm_dim,
        model_rank=model_rank,
        parsed=parsed,
        protein_chains=parsed.protein_chains,
        rna_chains=parsed.rna_chains,
    )
    if graph.edge_attr.numel() == 0:
        return None
    return graph


def load_boltz_training_graphs(
    bundle_dir: str | Path,
    *,
    use_esm: bool = True,
    use_rnafm: bool = True,
    max_decoys_per_group: int | None = None,
    partner_groups: list[str] | None = None,
) -> tuple[list[BoltzTrainingGroup], dict[str, InterfaceGraph]]:
    """
    Return partner groups and a job_name → graph cache for all loadable structures.
    """
    groups = load_boltz_training_groups(bundle_dir, partner_groups=partner_groups)
    esm_dim = ESM_DIM if use_esm else 0
    rnafm_dim = effective_rnafm_feature_dim(use_rnafm)
    graphs: dict[str, InterfaceGraph] = {}

    for group in groups:
        paths = [(group.native_job, group.native_path)]
        for job, path in zip(group.decoy_jobs, group.decoy_paths):
            paths.append((job, path))
        if max_decoys_per_group is not None:
            keep = {group.native_job}
            for job in group.decoy_jobs[:max_decoys_per_group]:
                keep.add(job)
            paths = [(j, p) for j, p in paths if j in keep]

        for job, path in paths:
            if job in graphs:
                continue
            try:
                g = build_boltz_interface_graph(
                    path,
                    use_esm=use_esm,
                    use_rnafm=use_rnafm,
                    esm_dim=esm_dim,
                    rnafm_dim=rnafm_dim,
                )
                if g is not None:
                    graphs[job] = g
            except Exception as exc:
                print(f"  SKIP graph {job}: {exc}")

    valid_groups = [
        g for g in groups
        if g.native_job in graphs
        and any(j in graphs for j in g.decoy_jobs)
    ]
    return valid_groups, graphs
