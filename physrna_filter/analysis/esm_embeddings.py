"""
ESM-2 protein language model embeddings for the graph transformer.

Uses esm2_t6_8M_UR50D (320-dim, ~8M params) by default — fast enough to
precompute for all 617 ProNAB chains.  Swap to esm2_t12_35M_UR50D (480-dim)
or esm2_t33_650M_UR50D (1280-dim) by changing MODEL_NAME.

Embeddings are cached per (pdb_id, chain_id) to a .npy + .json pair so
re-running the benchmark does not redo LM inference.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

MODEL_NAME = "esm2_t6_8M_UR50D"
ESM_DIM    = 320   # must match MODEL_NAME

_CACHE_DIR = Path(__file__).parent.parent / "data" / "emb_cache" / "esm2"
_model = None
_alphabet = None
_batch_converter = None


def _cache_key(pdb_id: str, chain_id: str, sequence: str) -> str:
    """
    Cache embeddings by both structure label and sequence content.

    ProNAB contains many mutations from the same PDB/chain.  A key that only
    includes pdb_id and chain_id would silently reuse the first mutant sequence
    for later mutations in that complex.
    """
    seq_hash = hashlib.sha1(sequence.encode("utf-8")).hexdigest()[:12]
    safe_chain = str(chain_id).replace("/", "_")
    return f"{pdb_id.lower()}_{safe_chain}_{seq_hash}"


def _load_model():
    global _model, _alphabet, _batch_converter
    if _model is not None:
        return
    try:
        import esm as esm_lib
    except ImportError as e:
        raise ImportError(
            "ESM-2 not installed. Run: pip install fair-esm"
        ) from e
    print(f"Loading {MODEL_NAME} ...")
    _model, _alphabet = getattr(esm_lib.pretrained, MODEL_NAME)()
    _model.eval()
    if torch.cuda.is_available():
        _model = _model.cuda()
    _batch_converter = _alphabet.get_batch_converter()


def get_esm2_embeddings(
    pdb_id: str,
    chain_id: str,
    sequence: str,
) -> dict[int, torch.Tensor]:
    """
    Returns per-residue ESM-2 embeddings for a protein chain.

    Args:
        pdb_id:   4-character PDB accession (used for cache key only)
        chain_id: chain letter
        sequence: one-letter amino acid sequence (length = n_residues)

    Returns:
        dict mapping 1-based residue index → embedding tensor [ESM_DIM]
    """
    cache_key = _cache_key(pdb_id, chain_id, sequence)
    emb_path  = _CACHE_DIR / f"{cache_key}.npy"
    idx_path  = _CACHE_DIR / f"{cache_key}.json"

    if emb_path.exists() and idx_path.exists():
        emb_matrix = np.load(str(emb_path))
        with open(idx_path) as f:
            residue_indices = json.load(f)
        return {
            int(k): torch.tensor(emb_matrix[i], dtype=torch.float32)
            for i, k in enumerate(residue_indices)
        }

    _load_model()
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    label = f"{pdb_id}_{chain_id}"
    data = [(label, sequence)]
    _, _, tokens = _batch_converter(data)
    if torch.cuda.is_available():
        tokens = tokens.cuda()

    with torch.no_grad():
        results = _model(tokens, repr_layers=[6], return_contacts=False)

    # Shape: [1, seq_len+2, ESM_DIM] — strip BOS/EOS
    representations = results["representations"][6][0, 1:-1].cpu()
    n = min(len(sequence), representations.shape[0])

    residue_indices = list(range(1, n + 1))
    emb_matrix = representations[:n].numpy()
    np.save(str(emb_path), emb_matrix)
    with open(idx_path, "w") as f:
        json.dump(residue_indices, f)

    return {
        i: torch.tensor(emb_matrix[i - 1], dtype=torch.float32)
        for i in residue_indices
    }


def get_esm2_embeddings_from_structure(
    pdb_id: str,
    chain_id: str,
    protein_chains: list,
) -> dict[tuple[str, int], torch.Tensor]:
    """
    Convenience wrapper: extracts sequence from BioPython chain objects and
    returns a dict keyed by (chain_id, residue_number) → embedding tensor.
    """
    from ..structure.mutate import _ONE_TO_THREE
    _THREE_TO_ONE = {v: k for k, v in _ONE_TO_THREE.items()}
    _THREE_TO_ONE.update({
        "HSD": "H", "HSE": "H", "HSP": "H", "HIP": "H", "HIE": "H",
        "MSE": "M", "SEC": "C",
    })

    chain = next((c for c in protein_chains if c.id == chain_id), None)
    if chain is None:
        return {}

    residue_list = [
        r for r in chain
        if r.id[0] == " " and r.resname in _THREE_TO_ONE
    ]
    sequence = "".join(_THREE_TO_ONE.get(r.resname, "X") for r in residue_list)
    resnums  = [r.id[1] for r in residue_list]

    per_idx = get_esm2_embeddings(pdb_id, chain_id, sequence)

    result: dict[tuple[str, int], torch.Tensor] = {}
    for seq_pos, resnum in enumerate(resnums, start=1):
        if seq_pos in per_idx:
            result[(chain_id, resnum)] = per_idx[seq_pos]

    return result
