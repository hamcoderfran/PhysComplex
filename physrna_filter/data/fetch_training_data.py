"""
Merge trusted protein-RNA mutation ΔΔG sources for PhysGT training.

Current sources:
  1. PRA-MutPred / ProNAB scrape (710 entries) — primary
  2. Nabe RNA subset (~318 novel vs ProNAB) — supplemental
  3. Literature curation (2023–2026 ITC/SPR/MST) — high-quality additions

Deduplication key: lowercased pdb_id + normalized mutation string.
When duplicates exist, ProNAB entries are kept (curated for PRA-MutPred).
"""
from __future__ import annotations

import pandas as pd

from .fetch_pronab import fetch_pronab
from .fetch_nabe import fetch_nabe, normalize_mutation_string
from .fetch_literature import fetch_literature


def _entry_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["pdb_id"].astype(str).str.lower()
        + "|"
        + df["mutation"].map(normalize_mutation_string)
    )


def fetch_training_data(
    include_nabe: bool = True,
    include_literature: bool = True,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Return merged, deduplicated training DataFrame for PhysGT.
    """
    pronab = fetch_pronab(force_download=force_download)
    pronab = pronab.copy()
    pronab["source"] = "pronab"
    pronab["mutation"] = pronab["mutation"].map(normalize_mutation_string)

    frames = [pronab]
    if include_nabe:
        nabe = fetch_nabe(force_download=force_download, rna_only=True)
        nabe = nabe.copy()
        nabe["source"] = "nabe"
        nabe["mutation"] = nabe["mutation"].map(normalize_mutation_string)
        frames.append(nabe)

    if include_literature:
        lit = fetch_literature()
        if len(lit):
            lit = lit.copy()
            lit["mutation"] = lit["mutation"].map(normalize_mutation_string)
            frames.append(lit)

    merged = pd.concat(frames, ignore_index=True)
    merged["_key"] = _entry_key(merged)

    # prefer pronab on duplicate keys (first in sort order)
    source_rank = {"pronab": 0, "literature": 1, "nabe": 2}
    merged["_rank"] = merged["source"].map(source_rank)
    merged = merged.sort_values(["_key", "_rank"])
    merged = merged.drop_duplicates("_key", keep="first")
    merged = merged.drop(columns=["_key", "_rank"]).reset_index(drop=True)

    counts = merged["source"].value_counts()
    n_pronab = counts.get("pronab", 0)
    n_nabe = counts.get("nabe", 0)
    n_lit = counts.get("literature", 0)
    print(
        f"Training merge: {len(merged)} entries "
        f"({n_pronab} ProNAB + {n_nabe} Nabe-only + {n_lit} literature), "
        f"{merged['pdb_id'].nunique()} unique PDBs"
    )
    return merged
