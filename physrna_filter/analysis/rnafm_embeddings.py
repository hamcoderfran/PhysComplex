"""
RNA-FM pre-trained RNA language model embeddings.

IMPORTANT: Training never downloads weights automatically.  The CUHK CDN
returns HTTP 403 for most users and must not be contacted during ProNAB
training.  Download weights once, separately:

    python -m physrna_filter.data.download_rnafm_weights

Then either leave them in physrna_filter/data/rnafm_weights/ or set:

    set RNAFM_CHECKPOINT=C:\\path\\to\\RNA-FM_pretrained.pth

Verify a manual download:

    python -m physrna_filter.data.verify_rnafm_weights

If no local weights are found, a deterministic one-hot + positional
encoding (dim=20) is used and training continues without skipping entries.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

RNAFM_DIM      = 640
RNAFM_FALLBACK = 20
RNAFM_FILENAME = "RNA-FM_pretrained.pth"
# Real checkpoint is ~1.19 GB; reject 403 HTML pages saved as .pth
MIN_WEIGHT_BYTES = 500_000_000

_WEIGHTS_DIR = Path(__file__).parent.parent / "data" / "rnafm_weights"
_CACHE_DIR   = Path(__file__).parent.parent / "data" / "emb_cache" / "rnafm"
_PROJECT_ROOT = Path(__file__).parent.parent.parent

_model = None
_alphabet = None
_batch_converter = None
_package_installed: bool | None = None
_mode_announced = False
_load_warning_emitted = False
_hub_blocked = False
_load_failed = False


def _cache_key(pdb_id: str, chain_id: str, sequence: str) -> str:
    seq_hash = hashlib.sha1(sequence.encode("utf-8")).hexdigest()[:12]
    safe_chain = str(chain_id).replace("/", "_")
    return f"{pdb_id.lower()}_{safe_chain}_{seq_hash}"


def _clean_path_string(raw: str) -> str:
    """Strip quotes/whitespace from env vars (common on Windows)."""
    return raw.strip().strip('"').strip("'").strip()


def _looks_like_html_error(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(512)
    except OSError:
        return True
    lowered = head.lower()
    return (
        head.lstrip().startswith(b"<")
        or b"<!doctype" in lowered
        or b"<html" in lowered
        or b"forbidden" in lowered
        or b"403" in head[:80]
    )


def is_valid_weights_file(path: Path | str) -> bool:
    """True when path looks like a real RNA-FM checkpoint (not a 403 HTML page)."""
    p = Path(path)
    if not p.is_file():
        return False
    size = p.stat().st_size
    if size < MIN_WEIGHT_BYTES:
        return False
    if _looks_like_html_error(p):
        return False
    return True


def candidate_weight_paths() -> list[Path]:
    """All locations checked for RNA-FM_pretrained.pth (in priority order)."""
    seen: set[str] = set()
    candidates: list[Path] = []

    def add(path: Path) -> None:
        try:
            resolved = str(path.expanduser().resolve())
        except OSError:
            resolved = str(path.expanduser())
        if resolved not in seen:
            seen.add(resolved)
            candidates.append(path.expanduser())

    env_path = _clean_path_string(os.environ.get("RNAFM_CHECKPOINT", ""))
    if env_path:
        add(Path(env_path))

    weights_dir = _clean_path_string(os.environ.get("RNAFM_WEIGHTS_DIR", ""))
    if weights_dir:
        add(Path(weights_dir) / RNAFM_FILENAME)

    add(_WEIGHTS_DIR / RNAFM_FILENAME)

    hub_path = Path(torch.hub.get_dir()) / "checkpoints" / RNAFM_FILENAME
    add(hub_path)

    for base in (Path.cwd(), _PROJECT_ROOT, _WEIGHTS_DIR):
        add(base / RNAFM_FILENAME)
        add(base / "rnafm_weights" / RNAFM_FILENAME)

    for alt_name in (
        RNAFM_FILENAME,
        "rnafm_pretrained.pth",
        "RNA_FM_pretrained.pth",
        "file_RNA-FM_pretrained.pth",
    ):
        add(_WEIGHTS_DIR / alt_name)
        if env_path:
            parent = Path(env_path).expanduser().parent
            add(parent / alt_name)

    return candidates


def resolve_weights_path() -> Path | None:
    """Return path to locally cached RNA-FM weights.  Never downloads."""
    if os.environ.get("RNAFM_DISABLE", "").lower() in ("1", "true", "yes"):
        return None

    for candidate in candidate_weight_paths():
        if is_valid_weights_file(candidate):
            return candidate.resolve()

    return None


def describe_weight_search() -> list[dict]:
    """Diagnostic info for each candidate path (used by verify script)."""
    rows = []
    for candidate in candidate_weight_paths():
        p = candidate.expanduser()
        row = {
            "path": str(p),
            "exists": p.is_file(),
            "size_bytes": p.stat().st_size if p.is_file() else 0,
            "valid": is_valid_weights_file(p),
        }
        if row["exists"] and not row["valid"]:
            if row["size_bytes"] < MIN_WEIGHT_BYTES:
                row["reason"] = (
                    f"too small ({row['size_bytes']:,} B; need >={MIN_WEIGHT_BYTES:,} B) "
                    "— likely a failed download or 403 HTML page"
                )
            elif _looks_like_html_error(p):
                row["reason"] = "file looks like HTML (403 error page saved as .pth)"
            else:
                row["reason"] = "failed validation"
        rows.append(row)
    return rows


def _load_checkpoint(path: Path | str) -> dict:
    """Load RNA-FM checkpoint.  PyTorch 2.6+ defaults weights_only=True which breaks rna-fm."""
    return torch.load(str(path), map_location="cpu", weights_only=False)


def rnafm_package_installed() -> bool:
    global _package_installed
    if _package_installed is None:
        try:
            import fm  # noqa: F401
            _package_installed = True
        except ImportError:
            _package_installed = False
    return _package_installed


def embedding_dim() -> int:
    """Feature width that graph nodes will use.  Matches actually loaded model state."""
    return RNAFM_DIM if _model is not None else RNAFM_FALLBACK


def prepare_rnafm_for_training(use_rnafm: bool) -> None:
    """Eagerly load RNA-FM before graph building so embedding dims stay consistent."""
    if use_rnafm:
        announce_rnafm_mode()


def effective_rnafm_feature_dim(use_rnafm: bool) -> int:
    if not use_rnafm:
        return 0
    return embedding_dim()


def announce_rnafm_mode() -> str:
    """Print once how RNA-FM will be used.  Eagerly loads weights when present."""
    global _mode_announced
    if _mode_announced:
        return "real" if _model is not None else "fallback"
    _mode_announced = True

    path = resolve_weights_path()
    if path and rnafm_package_installed():
        try:
            _load_model()
            print(f"RNA-FM: loaded from {path}")
            return "real"
        except Exception as e:
            _emit_fallback_warning(str(e).splitlines()[0][:200])
            return "fallback"

    reason = "rna-fm not installed" if not rnafm_package_installed() else "no valid local weights"
    print(
        f"RNA-FM: {reason} — using fallback encoding (dim={RNAFM_FALLBACK}).\n"
        f"  If you downloaded manually, verify the file:\n"
        f"    python -m physrna_filter.data.verify_rnafm_weights\n"
        f"  Or download via Hugging Face:\n"
        f"    python -m physrna_filter.data.download_rnafm_weights"
    )
    return "fallback"


def _emit_fallback_warning(reason: str) -> None:
    global _load_warning_emitted
    if _load_warning_emitted:
        return
    _load_warning_emitted = True
    print(
        f"\nRNA-FM fallback active ({reason}). Training continues.\n"
        f"Verify weights: python -m physrna_filter.data.verify_rnafm_weights\n",
        file=sys.stderr,
    )


def _block_rnafm_cdn() -> None:
    """Prevent rna-fm from ever contacting the CUHK CDN during training."""
    global _hub_blocked
    if _hub_blocked:
        return
    _hub_blocked = True

    import fm.pretrained as pretrained

    def _blocked_hub(*_args, **_kwargs):
        raise RuntimeError(
            "RNA-FM CUHK CDN download blocked. "
            "Place RNA-FM_pretrained.pth locally and set RNAFM_CHECKPOINT, "
            "or run: python -m physrna_filter.data.verify_rnafm_weights"
        )

    pretrained.load_fm_model_and_alphabet_hub = _blocked_hub

    _orig_rna_fm_t12 = pretrained.rna_fm_t12

    def _safe_rna_fm_t12(model_location=None):
        if model_location is None:
            raise RuntimeError(
                "rna_fm_t12() called without model_location — CUHK CDN blocked. "
                "Set RNAFM_CHECKPOINT to your local .pth file."
            )
        loc = Path(_clean_path_string(str(model_location))).expanduser().resolve()
        if not loc.is_file():
            raise FileNotFoundError(f"RNA-FM weights not found at {loc}")
        if not is_valid_weights_file(loc):
            raise ValueError(
                f"RNA-FM weights at {loc} failed validation "
                f"(size={loc.stat().st_size:,} B). "
                "Run: python -m physrna_filter.data.verify_rnafm_weights"
            )
        return pretrained.load_model_and_alphabet_local(str(loc), theme="rna")

    pretrained.rna_fm_t12 = _safe_rna_fm_t12
    pretrained._physrna_safe_rna_fm_t12 = _safe_rna_fm_t12
    pretrained._physrna_orig_rna_fm_t12 = _orig_rna_fm_t12


def _load_model() -> None:
    """Load RNA-FM from a LOCAL path only.  Never contacts CUHK CDN."""
    global _model, _alphabet, _batch_converter, _load_failed

    if _model is not None:
        return
    if _load_failed:
        raise RuntimeError("RNA-FM load previously failed — using fallback encoding")

    weights_path = resolve_weights_path()
    if weights_path is None:
        raise RuntimeError(
            "RNA-FM weights not found locally. "
            "Run: python -m physrna_filter.data.verify_rnafm_weights"
        )

    if not rnafm_package_installed():
        raise ImportError("pip install rna-fm")

    _block_rnafm_cdn()
    import fm.pretrained as pretrained

    abs_path = weights_path.resolve()
    print(f"Loading RNA-FM from {abs_path} ({abs_path.stat().st_size / 1e9:.2f} GB) ...")
    try:
        model_data = _load_checkpoint(abs_path)
        _model, _alphabet = pretrained.load_model_and_alphabet_core(
            abs_path.stem, model_data, regression_data=None, theme="rna"
        )
    except Exception:
        _load_failed = True
        raise

    _model.eval()
    if torch.cuda.is_available():
        _model = _model.cuda()
    _batch_converter = _alphabet.get_batch_converter()


def _fallback_embedding(sequence: str, resnums: list[int]) -> dict[int, torch.Tensor]:
    vocab = {"A": 0, "G": 1, "C": 2, "U": 3, "T": 3}
    result = {}
    for i, (nt, rn) in enumerate(zip(sequence, resnums)):
        oh = torch.zeros(4)
        oh[vocab.get(nt.upper(), 3)] = 1.0
        pos = torch.zeros(16)
        for k in range(8):
            freq = 1.0 / (10000 ** (2 * k / 16))
            pos[2 * k]     = float(np.sin(i * freq))
            pos[2 * k + 1] = float(np.cos(i * freq))
        result[rn] = torch.cat([oh, pos])
    return result


def get_rnafm_embeddings(
    pdb_id: str,
    chain_id: str,
    sequence: str,
    resnums: list[int],
) -> dict[tuple[str, int], torch.Tensor]:
    """
    Per-nucleotide embeddings.  Never raises.  Never downloads.
    """
    cache_key = _cache_key(pdb_id, chain_id, sequence)
    emb_path  = _CACHE_DIR / f"{cache_key}.npy"
    idx_path  = _CACHE_DIR / f"{cache_key}.json"

    if emb_path.exists() and idx_path.exists():
        emb_matrix = np.load(str(emb_path))
        expected_width = embedding_dim()
        if emb_matrix.ndim == 2 and emb_matrix.shape[1] == expected_width:
            with open(idx_path) as f:
                cached_resnums = json.load(f)
            return {
                (chain_id, rn): torch.tensor(emb_matrix[i], dtype=torch.float32)
                for i, rn in enumerate(cached_resnums)
            }
        emb_path.unlink(missing_ok=True)
        idx_path.unlink(missing_ok=True)

    if resolve_weights_path() is None or not rnafm_package_installed() or _load_failed:
        keyed = _fallback_embedding(sequence, resnums)
        return {(chain_id, rn): v for rn, v in keyed.items()}

    try:
        _load_model()
    except Exception as e:
        _emit_fallback_warning(str(e).splitlines()[0][:200])
        keyed = _fallback_embedding(sequence, resnums)
        return {(chain_id, rn): v for rn, v in keyed.items()}

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    rna_seq = sequence.upper().replace("T", "U")
    label = f"{pdb_id}_{chain_id}"
    _, _, tokens = _batch_converter([(label, rna_seq)])
    if torch.cuda.is_available():
        tokens = tokens.cuda()

    with torch.no_grad():
        results = _model(tokens, repr_layers=[12], return_contacts=False)

    representations = results["representations"][12][0, 1:-1].cpu()
    n = min(len(sequence), representations.shape[0], len(resnums))

    emb_matrix = representations[:n].numpy()
    np.save(str(emb_path), emb_matrix)
    with open(idx_path, "w") as f:
        json.dump(resnums[:n], f)

    return {
        (chain_id, resnums[i]): torch.tensor(emb_matrix[i], dtype=torch.float32)
        for i in range(n)
    }


def get_rnafm_embeddings_from_structure(
    pdb_id: str,
    chain_id: str,
    rna_chains: list,
) -> dict[tuple[str, int], torch.Tensor]:
    from ..structure.parse_complex import RNA_RESIDUE_NAMES

    _NUC_MAP = {
        "A": "A", "G": "G", "C": "C", "U": "U", "T": "U",
        "DA": "A", "DG": "G", "DC": "C", "DT": "U",
    }

    chain = next((c for c in rna_chains if c.id == chain_id), None)
    if chain is None:
        chain = rna_chains[0] if rna_chains else None
    if chain is None:
        return {}

    residue_list = [
        r for r in chain
        if r.id[0] == " " and r.resname.strip().upper() in RNA_RESIDUE_NAMES
    ]
    sequence = "".join(_NUC_MAP.get(r.resname.strip(), "N") for r in residue_list)
    resnums  = [r.id[1] for r in residue_list]

    return get_rnafm_embeddings(pdb_id, chain_id, sequence, resnums)
