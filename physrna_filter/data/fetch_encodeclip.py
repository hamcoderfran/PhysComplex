"""
Fetches eCLIP binding peak data from ENCODE for biological plausibility
cross-referencing.

eCLIP data tells you where an RBP actually binds on endogenous RNA in living
cells. If AF3 predicts binding at a site with zero eCLIP signal for that RBP,
that is biological evidence the prediction is wrong.
"""

import os
import requests
import pandas as pd


ENCODE_API = "https://www.encodeproject.org"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "eclip_cache")


def search_eclip_experiments(rbp_name: str) -> list:
    """
    Queries ENCODE for eCLIP experiments targeting a specific RBP.
    Returns list of experiment metadata dicts.
    """
    params = {
        "type": "Experiment",
        "assay_title": "eCLIP",
        "target.label": rbp_name,
        "status": "released",
        "format": "json",
        "limit": 50,
    }

    url = f"{ENCODE_API}/search/"
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    experiments = data.get("@graph", [])
    print(f"Found {len(experiments)} eCLIP experiments for {rbp_name}")
    return experiments


def fetch_eclip_peaks(experiment_accession: str) -> pd.DataFrame:
    """
    Fetches peak calls (bed file) for a given ENCODE eCLIP experiment.
    Returns DataFrame: chrom, start, end, name, score, strand.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{experiment_accession}_peaks.bed")

    if os.path.exists(cache_path):
        return _parse_bed(cache_path)

    url = f"{ENCODE_API}/experiments/{experiment_accession}/?format=json"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    experiment = response.json()
    bed_file = _find_bed_file(experiment)

    if not bed_file:
        print(f"No BED file found for {experiment_accession}")
        return pd.DataFrame()

    file_url = f"{ENCODE_API}{bed_file['href']}"
    file_response = requests.get(file_url, timeout=120)
    file_response.raise_for_status()

    with open(cache_path, "wb") as f:
        f.write(file_response.content)

    return _parse_bed(cache_path)


def check_site_has_eclip_signal(
    chrom: str,
    start: int,
    end: int,
    peaks_df: pd.DataFrame,
    min_score: float = 0.0,
) -> bool:
    """
    Returns True if any eCLIP peak overlaps the given genomic coordinates.
    """
    if peaks_df.empty:
        return False

    mask = (
        (peaks_df["chrom"] == chrom)
        & (peaks_df["start"] < end)
        & (peaks_df["end"] > start)
        & (peaks_df["score"] >= min_score)
    )
    return bool(mask.any())


def _find_bed_file(experiment: dict):
    for f in experiment.get("files", []):
        if (
            f.get("file_format") == "bed"
            and f.get("output_type") in ("IDR thresholded peaks", "replicated peaks")
            and f.get("status") == "released"
        ):
            return f
    return None


def _parse_bed(path: str) -> pd.DataFrame:
    cols = ["chrom", "start", "end", "name", "score", "strand"]
    try:
        df = pd.read_csv(path, sep="\t", header=None, comment="#")
        df.columns = cols[: len(df.columns)]
        return df
    except Exception as e:
        print(f"Failed to parse BED {path}: {e}")
        return pd.DataFrame()
