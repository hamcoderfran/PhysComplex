"""
PhysRNA paths and checkpoint resolution.

Shipped checkpoint lives in the package (read-only after ``pip install``).
Fine-tuned weights are written to the user data directory so ``git pull``
never overwrites your trained model.

Override with ``PHYSRNA_HOME`` or ``PHYSRNA_CHECKPOINT``.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import torch

_SHIPPED_CHECKPOINT = (
    Path(__file__).resolve().parent / "validation" / "gt_checkpoint.pt"
)
_INTERFACE_PRETRAIN = (
    Path(__file__).resolve().parent / "validation" / "gt_interface_pretrain.pt"
)
DEFAULT_INTERFACE_CUTOFF = 5.0


def user_data_dir() -> Path:
    raw = os.environ.get("PHYSRNA_HOME", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".physrna"


def shipped_checkpoint() -> Path:
    return _SHIPPED_CHECKPOINT


def default_interface_pretrain_path() -> Path:
    return _INTERFACE_PRETRAIN


def default_interface_cutoff() -> float:
    raw = os.environ.get("PHYSRNA_INTERFACE_CUTOFF", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return DEFAULT_INTERFACE_CUTOFF


def user_checkpoint() -> Path:
    return user_data_dir() / "gt_checkpoint.pt"


def af3_api_key_path() -> Path:
    return user_data_dir() / "af3_api_key"


def load_af3_api_key() -> str | None:
    """
    Resolve AlphaFold API key from env or ``~/.physrna/af3_api_key``.

    Checks ``AF3_API_KEY`` and ``ALPHAFOLD_API_KEY`` first (never logged).
    """
    for var in ("AF3_API_KEY", "ALPHAFOLD_API_KEY"):
        raw = os.environ.get(var, "").strip()
        if raw:
            return raw
    path = af3_api_key_path()
    if path.is_file():
        key = path.read_text(encoding="utf-8").strip()
        return key or None
    return None


def save_af3_api_key(api_key: str) -> Path:
    """Persist API key under ``~/.physrna/`` with restrictive permissions."""
    key = api_key.strip()
    if not key:
        raise ValueError("API key is empty")
    user_data_dir().mkdir(parents=True, exist_ok=True)
    path = af3_api_key_path()
    path.write_text(key + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return path


def clear_af3_api_key() -> bool:
    path = af3_api_key_path()
    if path.is_file():
        path.unlink()
        return True
    return False


def mask_secret(value: str, *, visible: int = 4) -> str:
    if len(value) <= visible:
        return "*" * len(value)
    return value[:visible] + "*" * (len(value) - visible)


def resolve_gt_checkpoint(explicit: str | Path | None = None) -> Path:
    """
    Best checkpoint for inference (user fine-tune > env > shipped).
    """
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file() and path.stat().st_size > 10_000:
            return path.resolve()

    env = os.environ.get("PHYSRNA_CHECKPOINT", "").strip()
    if env:
        path = Path(env).expanduser()
        if path.is_file() and path.stat().st_size > 10_000:
            return path.resolve()

    user = user_checkpoint()
    if user.is_file() and user.stat().st_size > 10_000:
        return user.resolve()

    shipped = shipped_checkpoint()
    if shipped.is_file() and shipped.stat().st_size > 10_000:
        return shipped.resolve()

    from .validation.download_gt_checkpoint import ensure_public_checkpoint

    return ensure_public_checkpoint(shipped)


def writable_gt_checkpoint(explicit: str | Path | None = None) -> Path:
    """
    Checkpoint path for fine-tune / deploy writes (never the installed package).
    """
    if explicit:
        path = Path(explicit).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.resolve()

    user = user_checkpoint()
    if user.is_file() and user.stat().st_size > 10_000:
        return user.resolve()

    shipped = resolve_gt_checkpoint()
    dest = user_data_dir()
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(shipped, user)
    print(f"Copied base checkpoint to {user} (fine-tunes are saved here)")
    return user.resolve()


def backup_checkpoint(path: str | Path) -> Path | None:
    """Copy checkpoint to ``.bak`` before destructive writes."""
    src = Path(path)
    if not src.is_file():
        return None
    backup = src.with_suffix(src.suffix + ".bak")
    shutil.copy2(src, backup)
    return backup


def checkpoint_is_finetuned(path: str | Path | None = None) -> bool:
    ckpt = Path(path) if path else resolve_gt_checkpoint()
    if not ckpt.is_file():
        return False
    try:
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        return bool(payload.get("af3_panel_finetuned"))
    except Exception:
        return False


def init_user_environment(*, copy_checkpoint: bool = True) -> dict[str, str]:
    """
    One-shot setup: ensure user data dir and checkpoint exist; detect oxRNA.
  Returns a short status dict for CLI messaging.
    """
    status: dict[str, str] = {}
    user_data_dir().mkdir(parents=True, exist_ok=True)
    status["data_dir"] = str(user_data_dir())

    if copy_checkpoint:
        ckpt = writable_gt_checkpoint()
        status["checkpoint"] = str(ckpt)
        finetuned = checkpoint_is_finetuned(ckpt)
        status["finetuned"] = "yes" if finetuned else "no"
    else:
        ckpt = resolve_gt_checkpoint()
        status["checkpoint"] = str(ckpt)

    oxdna = os.environ.get("OXDNA_BIN", "").strip()
    if oxdna:
        status["oxdna"] = oxdna
    else:
        import shutil as _sh

        found = _sh.which("oxDNA")
        status["oxdna"] = found or "not found (internal C4' fallback will be used)"

    return status


def doctor_report() -> dict[str, str]:
    """Health check for checkpoint, oxDNA, and optional RNA-FM weights."""
    status = init_user_environment(copy_checkpoint=False)
    try:
        ckpt = resolve_gt_checkpoint()
        status["checkpoint_ok"] = "yes"
        status["checkpoint_path"] = str(ckpt)
        status["finetuned"] = "yes" if checkpoint_is_finetuned(ckpt) else "no"
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        status["interface_head_trained"] = str(
            bool(payload.get("interface_head_trained"))
        )
    except Exception as exc:
        status["checkpoint_ok"] = "no"
        status["checkpoint_error"] = str(exc).splitlines()[0]

    try:
        from .data.verify_rnafm_weights import rnafm_weights_available

        status["rnafm_weights"] = "ok" if rnafm_weights_available() else "missing (optional for AF3 screening)"
    except Exception:
        status["rnafm_weights"] = "not checked"

    key = load_af3_api_key()
    if key:
        status["af3_api_key"] = f"configured ({mask_secret(key)})"
    else:
        status["af3_api_key"] = "not set (use: physrna configure af3 --api-key KEY)"

    return status
