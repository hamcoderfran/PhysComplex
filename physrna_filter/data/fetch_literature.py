"""
Curated protein-RNA mutation ΔΔG from recent literature (2023–2026).

Manual curation of ITC/SPR/MST papers for complexes missing from ProNAB/Nabe.
See literature_mined.csv and docs/DATA_SOURCES.md for provenance.
"""
from __future__ import annotations

import os

import pandas as pd

from .fetch_pronab import _parse_mutations
from .fetch_nabe import normalize_mutation_string

LOCAL_PATH = os.path.join(os.path.dirname(__file__), "literature_mined.csv")


def fetch_literature() -> pd.DataFrame:
    """
    Load manually curated literature entries in ProNAB schema.

    Returns columns: pdb_id, chain, mutation, ddg, method, type, source, ...
    """
    if not os.path.exists(LOCAL_PATH):
        print("Literature: no curated CSV found — returning empty set")
        return pd.DataFrame(
            columns=[
                "pdb_id", "chain", "mutation", "ddg", "method", "type", "source",
            ]
        )

    df = pd.read_csv(LOCAL_PATH)
    df = df.copy()
    df["source"] = "literature"
    df["mutation"] = df["mutation"].map(normalize_mutation_string)
    df = _parse_mutations(df)
    df = df[(df["position"] > 0) & (df["wt_aa"] != "?")].reset_index(drop=True)
    print(f"Literature: loaded {len(df)} curated RNA-protein entries")
    return df
