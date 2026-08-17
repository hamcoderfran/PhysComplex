"""
Fetch protein-RNA mutation ΔΔG data from the Nabe database.

Nabe (Deng lab, PMC8363842) is a peer-reviewed mutagenesis database with
2,506 mutations across protein-DNA and protein-RNA complexes.  The RNA subset
overlaps PRA-MutPred/ProNAB but adds ~300+ parseable entries with PDB structures.

Downloads (verified 2026-06):
  http://nabe.denglab.org/download/Nabe_database.tar.gz   — full set
  http://nabe.denglab.org/download/PRNA90.txt              — non-redundant RNA benchmark
"""
from __future__ import annotations

import io
import os
import re
import tarfile

import pandas as pd
import requests

from .fetch_pronab import _parse_mutations, parse_mutation_string

NABE_FULL_URL = "http://nabe.denglab.org/download/Nabe_database.tar.gz"
NABE_PRNA90_URL = "http://nabe.denglab.org/download/PRNA90.txt"
LOCAL_PATH = os.path.join(os.path.dirname(__file__), "nabe_raw.csv")
_MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def fetch_nabe(
    force_download: bool = False,
    rna_only: bool = True,
) -> pd.DataFrame:
    """
    Load Nabe mutagenesis data standardized to the ProNAB schema.

    Returns columns: pdb_id, chain, mutation, ddg, method, type, wt_aa, position, mut_aa
    """
    if not os.path.exists(LOCAL_PATH) or force_download:
        _download_nabe_full()

    df = pd.read_csv(LOCAL_PATH)
    if rna_only:
        df = df[df["type"] == "RNA-Protein"].reset_index(drop=True)

    df = _parse_mutations(df)
    df = df[(df["position"] > 0) & (df["wt_aa"] != "?")].reset_index(drop=True)
    print(f"Nabe: loaded {len(df)} RNA-protein entries with measured ΔΔG")
    return df


def _download_nabe_full() -> None:
    print(f"Fetching Nabe database from {NABE_FULL_URL} ...")
    response = requests.get(
        NABE_FULL_URL, timeout=120, headers={"User-Agent": "Mozilla/5.0"}
    )
    response.raise_for_status()

    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tf:
        member = next(m for m in tf.getmembers() if m.name.endswith(".txt"))
        raw = tf.extractfile(member).read().decode("utf-8")

    rows = []
    for line in raw.splitlines()[1:]:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 8:
            continue
        pdb_id, chain, partner, mutant, temp, kd, ddg, pubmed = parts[:8]
        partner_u = partner.upper()
        if partner_u not in ("RNA", "DNA"):
            continue
        mut_clean = mutant.replace(" ", "")
        if "/" in mut_clean or not _MUT_RE.match(mut_clean):
            continue
        rows.append({
            "pdb_id": pdb_id.upper(),
            "chain": chain,
            "mutation": mut_clean,
            "ddg": float(ddg),
            "method": f"Nabe/PMID{pubmed}",
            "type": "RNA-Protein" if partner_u == "RNA" else "DNA-Protein",
            "temperature_k": temp,
            "kd": kd,
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(LOCAL_PATH), exist_ok=True)
    df.to_csv(LOCAL_PATH, index=False)
    print(f"Nabe: saved {len(df)} entries to {LOCAL_PATH}")


def normalize_mutation_string(mutation_str: str) -> str:
    """Normalize 'D 210A' -> 'D210A' for parsers."""
    return re.sub(r"\s+", "", str(mutation_str).strip().upper())
