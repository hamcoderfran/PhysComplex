"""
Downloads PDB structures from RCSB for a given list of PDB IDs.
"""

import os
import requests


RCSB_BASE = "https://files.rcsb.org/download"
STRUCTURE_DIR = os.path.join(os.path.dirname(__file__), "structures")


def download_pdb(pdb_id: str, output_dir: str = STRUCTURE_DIR) -> str | None:
    """
    Downloads a .pdb file from RCSB.
    Returns local path on success, None on failure.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{pdb_id.lower()}.pdb")

    if os.path.exists(path):
        return path

    url = f"{RCSB_BASE}/{pdb_id.upper()}.pdb"
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"PDB format unavailable for {pdb_id} ({e}); trying mmCIF ...")
        return download_cif(pdb_id, output_dir=output_dir)

    with open(path, "wb") as f:
        f.write(response.content)

    print(f"Downloaded {pdb_id} -> {path}")
    return path


def download_cif(pdb_id: str, output_dir: str = STRUCTURE_DIR) -> str | None:
    """
    Downloads a .cif file from RCSB.
    Preferred for structures with many atoms or unusual residues.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{pdb_id.lower()}.cif")

    if os.path.exists(path):
        return path

    url = f"{RCSB_BASE}/{pdb_id.upper()}.cif"
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to download {pdb_id} as CIF: {e}")
        return None

    with open(path, "wb") as f:
        f.write(response.content)

    print(f"Downloaded {pdb_id} (mmCIF) -> {path}")
    return path


def download_all_pronab_structures(
    pronab_df, output_dir: str = STRUCTURE_DIR
) -> dict:
    """
    Downloads all unique PDB structures referenced in a ProNAB dataframe.
    Returns dict mapping pdb_id -> local file path.
    """
    pdb_ids = pronab_df["pdb_id"].unique()
    paths = {}

    for pdb_id in pdb_ids:
        path = download_pdb(pdb_id, output_dir)
        if path:
            paths[pdb_id] = path

    print(f"Downloaded {len(paths)}/{len(pdb_ids)} structures")
    return paths
