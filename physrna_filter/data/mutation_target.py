"""Classify whether a merged training mutation targets protein or RNA."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .fetch_pronab import parse_mutation_string

RNA_BASES = frozenset("AUGC")


@dataclass(frozen=True)
class MutationTarget:
    kind: str  # "protein" or "rna"
    chain_id: str
    position: int
    wt: str
    mut: str


def is_rna_mutation(wt_aa: str, mut_aa: str) -> bool:
    """True when both alleles are standard RNA nucleotides."""
    return wt_aa.upper() in RNA_BASES and mut_aa.upper() in RNA_BASES


def resolve_mutation_target(row: pd.Series) -> MutationTarget:
    """Resolve mutation chain and polymer type from a merged training row."""
    wt, position, mut = parse_mutation_string(row["mutation"])
    chain = row.get("chain")
    chain_id = str(chain).strip() if chain is not None and not pd.isna(chain) else ""

    if chain_id and is_rna_mutation(wt, mut):
        return MutationTarget("rna", chain_id, position, wt, mut)
    if chain_id:
        return MutationTarget("protein", chain_id, position, wt, mut)
    return MutationTarget("protein", "", position, wt, mut)


def validate_mutation_target_for_structure(
    target: MutationTarget,
    *,
    protein_chain_ids: set[str],
    rna_chain_ids: set[str],
) -> None:
    """
    Fail fast when merged metadata disagrees with deposited polymer types.

    Nabe RNA-chain rows sometimes use protein-style allele codes (e.g. R7L on
    chain X).  Treating those as protein mutations on ribosomal-sized complexes
    triggers unbounded interface scans and appears as a hang.
    """
    if not target.chain_id:
        return

    if target.chain_id in rna_chain_ids and target.kind == "protein":
        raise ValueError(
            f"Chain {target.chain_id} is RNA in structure but mutation "
            f"{target.wt}{target.position}{target.mut} uses protein allele codes; "
            "Nabe RNA-chain entries require AUGC alleles"
        )

    known = protein_chain_ids | rna_chain_ids
    if target.chain_id not in known:
        raise ValueError(f"Chain {target.chain_id} not found in structure")

    if target.kind == "protein" and target.chain_id not in protein_chain_ids:
        raise ValueError(
            f"Protein mutation chain {target.chain_id} not in structure protein chains"
        )
