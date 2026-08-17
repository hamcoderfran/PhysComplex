"""
Load AF3-predicted structures as contrastive training pairs (native vs decoys).

Unlike ``boltz_training_pairs`` (Boltz mmCIF), this module expects AF3 Server
zips or mmCIF files under ``predictions/`` for each manifest job — including
decoy sequences from ``af3_error_decoys`` bundles.
"""
from __future__ import annotations

import re
import zipfile
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
from .af3_client import build_alphafold_server_job


@dataclass
class Af3DecoyTrainingGroup:
    partner_group: str
    native_job: str
    native_path: str
    decoy_jobs: list[str]
    decoy_paths: list[str]


def _norm_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def match_af3_prediction(predictions_dir: Path, job_name: str) -> Path | None:
    """Find an AF3 zip or structure file for ``job_name`` under ``predictions_dir``."""
    if not predictions_dir.is_dir():
        return None
    token = _norm_key(job_name)
    candidates: list[Path] = []
    for pattern in ("*.zip", "*.cif", "*.mmcif", "*.pdb"):
        for path in predictions_dir.rglob(pattern):
            stem = _norm_key(path.stem)
            if stem == token or token in stem or stem.endswith(token):
                if path.suffix.lower() == ".zip":
                    if path.stat().st_size < 128 or not zipfile.is_zipfile(path):
                        continue
                candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: (0 if p.suffix.lower() == ".zip" else 1, len(p.name)))
    return candidates[0]


def load_af3_decoy_training_groups(
    bundle_dir: str | Path,
    *,
    predictions_subdir: str = "af3_predictions",
    partner_groups: list[str] | None = None,
) -> list[Af3DecoyTrainingGroup]:
    root = Path(bundle_dir)
    manifest = pd.read_csv(root / "manifest.csv")
    pred_dir = root / predictions_subdir
    allowed = {g.lower() for g in partner_groups} if partner_groups else None

    groups: list[Af3DecoyTrainingGroup] = []
    for partner_group, sub in manifest.groupby("partner_group"):
        if allowed is not None and str(partner_group).lower() not in allowed:
            continue
        natives = sub[sub["label"] == "positive"]
        decoys = sub[sub["label"] == "negative"]
        if natives.empty or decoys.empty:
            continue
        native_row = natives.iloc[0]
        native_job = str(native_row["job_name"])
        native_path = match_af3_prediction(pred_dir, native_job)
        if native_path is None:
            continue

        decoy_jobs: list[str] = []
        decoy_paths: list[str] = []
        for _, row in decoys.iterrows():
            job = str(row["job_name"])
            path = match_af3_prediction(pred_dir, job)
            if path is not None:
                decoy_jobs.append(job)
                decoy_paths.append(str(path))

        if decoy_jobs:
            groups.append(
                Af3DecoyTrainingGroup(
                    partner_group=str(partner_group),
                    native_job=native_job,
                    native_path=str(native_path),
                    decoy_jobs=decoy_jobs,
                    decoy_paths=decoy_paths,
                )
            )
    return groups


def build_af3_decoy_interface_graph(
    structure_path: str,
    *,
    use_esm: bool = True,
    use_rnafm: bool = True,
    esm_dim: int | None = None,
    rnafm_dim: int | None = None,
    model_rank: int = 0,
) -> InterfaceGraph | None:
    esm_dim = esm_dim if esm_dim is not None else (ESM_DIM if use_esm else 0)
    rnafm_dim = rnafm_dim if rnafm_dim is not None else effective_rnafm_feature_dim(use_rnafm)

    parsed = parse_complex(structure_path, model_rank=model_rank)
    if not parsed.protein_chains or not parsed.rna_chains:
        return None

    pdb_id = Path(structure_path).stem
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
        structure_path,
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


def load_af3_decoy_training_graphs(
    bundle_dir: str | Path,
    *,
    predictions_subdir: str = "af3_predictions",
    partner_groups: list[str] | None = None,
    use_esm: bool = True,
    use_rnafm: bool = True,
    max_decoys_per_group: int | None = None,
) -> tuple[list[Af3DecoyTrainingGroup], dict[str, InterfaceGraph]]:
    groups = load_af3_decoy_training_groups(
        bundle_dir,
        predictions_subdir=predictions_subdir,
        partner_groups=partner_groups,
    )
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
                g = build_af3_decoy_interface_graph(
                    path,
                    use_esm=use_esm,
                    use_rnafm=use_rnafm,
                    esm_dim=esm_dim,
                    rnafm_dim=rnafm_dim,
                )
                if g is not None:
                    graphs[job] = g
            except Exception as exc:
                print(f"  SKIP AF3 graph {job}: {exc}")

    valid = [
        g for g in groups
        if g.native_job in graphs and any(j in graphs for j in g.decoy_jobs)
    ]
    return valid, graphs


def export_af3_jobs_from_bundle(
    bundle_dir: str | Path,
    output_path: str | Path,
    *,
    only_missing: Path | None = None,
) -> Path:
    """
    Write AlphaFold Server JSON for every manifest entry (native + decoys).

    Upload to https://alphafoldserver.com — place returned zips in
    ``bundle_dir/af3_predictions/``.
    """
    root = Path(bundle_dir)
    manifest = pd.read_csv(root / "manifest.csv")
    pred_dir = only_missing or (root / "af3_predictions")
    jobs: list[dict] = []
    for row in manifest.itertuples():
        job_name = str(row.job_name)
        if pred_dir.is_dir() and match_af3_prediction(pred_dir, job_name) is not None:
            continue
        protein = getattr(row, "protein_sequence", None)
        rna = getattr(row, "rna_sequence", None)
        if not protein or not rna or (isinstance(rna, float) and pd.isna(rna)):
            continue
        jobs.append(
            build_alphafold_server_job(
                str(protein),
                str(rna),
                job_name=job_name,
            )
        )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    import json

    out.write_text(json.dumps(jobs, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(jobs)} AF3 Server job(s) to {out}")
    return out
