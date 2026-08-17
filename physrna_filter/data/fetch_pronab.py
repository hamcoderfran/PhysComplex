"""
Fetches and parses the protein-RNA mutation ddG benchmark set.

ProNAB itself contains experimentally measured binding free energy changes
(ΔΔG) for mutations at protein-RNA interfaces, but the new ProNAB site has no
stable bulk-CSV download (search/request form behind bot-detection only).

PRA-MutPred (same lab, IIT Madras) publishes a 710-entry protein-RNA mutation
ddG set curated directly from ProNAB, embedded as plain HTML tables (a 595-row
training split + a 115-row test split). This is scraped and cached locally.
"""

import html as html_lib
import os
import re

PDB_ID_RE = re.compile(r"^[0-9][A-Za-z0-9]{3}$")
_PDB_ID_OCR_FIXES = {"I": "1", "l": "1", "O": "0"}

import pandas as pd
import requests


PRONAB_URL = "https://web.iitm.ac.in/bioinfo2/pronab/"
PRA_MUTPRED_URL = "https://web.iitm.ac.in/bioinfo2/pramutpred/dataset.html"
LOCAL_PATH = os.path.join(os.path.dirname(__file__), "pronab_raw.csv")


def fetch_pronab(force_download: bool = False) -> pd.DataFrame:
    """
    Loads the protein-RNA mutation ddG dataset and returns RNA-protein
    entries with experimental ΔΔG.

    Returns a DataFrame with columns:
        pdb_id, mutation, ddg, method, wt_aa, position, mut_aa
    """
    if not os.path.exists(LOCAL_PATH) or force_download:
        _fetch_pra_mutpred()

    df = pd.read_csv(LOCAL_PATH)
    df = _standardize_columns(df)
    df["pdb_id"] = df["pdb_id"].apply(_normalize_pdb_id)
    df = _filter_rna_protein(df)
    df = _parse_mutations(df)

    print(f"ProNAB: loaded {len(df)} RNA-protein entries with measured ΔΔG")
    return df


def _fetch_pra_mutpred() -> None:
    print(f"Fetching PRA-MutPred (ProNAB-derived) protein-RNA mutation set "
          f"from {PRA_MUTPRED_URL} ...")
    response = requests.get(
        PRA_MUTPRED_URL, timeout=60, headers={"User-Agent": "Mozilla/5.0"}
    )
    response.raise_for_status()

    rows = []
    for table in re.findall(r"<table.*?</table>", response.text, re.S):
        trs = re.findall(r"<tr.*?</tr>", table, re.S)
        if not trs:
            continue
        header = [
            _clean_cell(c)
            for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", trs[0], re.S)
        ]
        if header[:3] != ["S.No", "PDB", "Mutation"]:
            continue
        for tr in trs[1:]:
            cells = [
                _clean_cell(c)
                for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr, re.S)
            ]
            if len(cells) < 5:
                continue
            rows.append({
                "PDB_ID":   cells[1],
                "Mutation": cells[2],
                "DDG":      cells[3],
                "Method":   cells[4],
                "Type":     "RNA-Protein",
            })

    if not rows:
        raise RuntimeError(f"No data tables found at {PRA_MUTPRED_URL}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(LOCAL_PATH), exist_ok=True)
    df.to_csv(LOCAL_PATH, index=False)
    print(f"PRA-MutPred: saved {len(df)} entries to {LOCAL_PATH}")


def _clean_cell(cell: str) -> str:
    text = re.sub(r"<[^>]+>", "", cell)
    return html_lib.unescape(text).strip()


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "PDB_ID":   "pdb_id",
        "Chain":    "chain",
        "Mutation": "mutation",
        "DDG":      "ddg",
        "Method":   "method",
        "Type":     "type",
    }
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    return df.rename(columns=existing)


def _normalize_pdb_id(pdb_id: str) -> str:
    """
    Fixes OCR-style artifacts in PDB IDs scraped from PRA-MutPred's HTML
    tables (e.g. "Izdi" -> "1zdi", where the leading "I" was misread for "1").
    """
    pdb_id = str(pdb_id)
    if PDB_ID_RE.match(pdb_id):
        return pdb_id

    fixed = _PDB_ID_OCR_FIXES.get(pdb_id[:1], pdb_id[:1]) + pdb_id[1:]
    if PDB_ID_RE.match(fixed):
        return fixed

    return pdb_id


def _filter_rna_protein(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        df.get("type", pd.Series(["RNA-Protein"] * len(df))) == "RNA-Protein"
    ) & df["ddg"].notna()
    return df[mask].reset_index(drop=True)


def _parse_mutations(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df["mutation"].apply(_parse_mutation_string)
    df["wt_aa"]    = parsed.apply(lambda x: x[0])
    df["position"] = parsed.apply(lambda x: x[1])
    df["mut_aa"]   = parsed.apply(lambda x: x[2])
    return df


def parse_mutation_string(mutation_str: str) -> tuple[str, int, str]:
    """
    Parses a mutation string like 'Y13A' or 'D 210A'.
    Returns (wild_type_aa, position, mutant_aa).
    """
    clean = re.sub(r"\s+", "", str(mutation_str).strip().upper())
    wt  = clean[0]
    pos = int(clean[1:-1])
    mut = clean[-1]
    return wt, pos, mut


def _parse_mutation_string(mutation_str: str) -> tuple[str, int, str]:
    try:
        return parse_mutation_string(mutation_str)
    except (ValueError, IndexError):
        return ("?", -1, "?")
