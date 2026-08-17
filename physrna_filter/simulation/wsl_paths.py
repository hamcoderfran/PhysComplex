"""
Windows ↔ WSL path helpers for oxDNA / oat subprocess calls.

When PhysRNA runs on Windows but ``OXDNA_BIN=wsl:/home/.../oxDNA``, simulation
files under ``C:\\Users\\...`` must be passed to WSL as ``/mnt/c/Users/...``.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys


def oxdna_uses_wsl() -> bool:
    raw = os.environ.get("OXDNA_BIN", "").strip().lower()
    return raw.startswith("wsl:")


def windows_to_wsl_path(path: str) -> str:
    """Convert a Windows absolute path to a WSL /mnt/<drive>/... path."""
    normalized = path.replace("\\", "/")
    if normalized.startswith("/mnt/"):
        return normalized

    m = re.match(r"^([A-Za-z]):/(.*)$", normalized)
    if m:
        return f"/mnt/{m.group(1).lower()}/{m.group(2)}"

    if sys.platform != "win32":
        return normalized

    abs_path = os.path.normpath(path)
    drive, rest = os.path.splitdrive(abs_path)
    if not drive:
        return abs_path.replace("\\", "/")

    letter = drive.rstrip(":").lower()
    rest = rest.replace("\\", "/")
    if not rest.startswith("/"):
        rest = "/" + rest
    return f"/mnt/{letter}{rest}"


def host_path_for_subprocess(path: str) -> str:
    """Return the path form expected by the active oxDNA/oat backend."""
    normalized = path.replace("\\", "/")
    if normalized.startswith("/home/") or normalized.startswith("/tmp/") or normalized.startswith("/mnt/"):
        return normalized
    if oxdna_uses_wsl() and sys.platform == "win32":
        return windows_to_wsl_path(path)
    return path


def normalize_wsl_posix_path(path: str) -> str:
    """
    Normalize a WSL/Linux path without using pathlib (which corrupts
    ``/home/...`` into ``\\home\\...`` on Windows).
    """
    return path.strip().replace("\\", "/")


def wsl_python_for_oat(oat_bin: str) -> str:
    """Return the miniconda ``python`` sibling of a WSL ``oat`` executable."""
    path = normalize_wsl_posix_path(oat_bin)
    if path.endswith("/python"):
        return path
    if "/bin/" in path or path.endswith("/oat"):
        return path.rsplit("/", 1)[0] + "/python"
    return path


def wsl_login_command(inner: str) -> list[str]:
    """Run a shell command inside WSL (loads ~/.bashrc for conda PATH)."""
    return ["wsl", "bash", "-lc", inner]


_WSL_CONDA_INIT = (
    'source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null; '
    "conda activate base 2>/dev/null; "
)


def wsl_find_oat_path() -> str | None:
    """
    Locate ``oat`` inside WSL when conda is not on non-interactive PATH.

    Honors ``OAT_BIN=wsl:/home/you/miniconda3/bin/oat``.
    """
    if sys.platform != "win32":
        return None

    raw = os.environ.get("OAT_BIN", "").strip()
    if raw.lower().startswith("wsl:"):
        path = raw[4:].strip()
        return path or None

    probe = (
        f"{_WSL_CONDA_INIT}"
        'command -v oat 2>/dev/null || '
        'test -x "$HOME/miniconda3/bin/oat" && echo "$HOME/miniconda3/bin/oat" || '
        'test -x "$HOME/.local/bin/oat" && echo "$HOME/.local/bin/oat"'
    )
    try:
        result = subprocess.run(
            wsl_login_command(probe),
            capture_output=True,
            text=True,
            timeout=20,
        )
        for line in reversed((result.stdout or "").strip().splitlines()):
            line = line.strip()
            if line.endswith("/oat") or line == "oat":
                return line
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    return None


def wsl_mktemp_dir(prefix: str = "physrna_sim_") -> str:
    """Create a directory under ``/tmp`` inside WSL and return its POSIX path."""
    result = subprocess.run(
        ["wsl", "mktemp", "-d", "-p", "/tmp", f"{prefix}XXXXXX"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    if result.returncode != 0:
        raise RuntimeError(f"WSL mktemp failed: {(result.stderr or result.stdout or '').strip()}")
    path = (result.stdout or "").strip()
    if not path.startswith("/"):
        raise RuntimeError(f"WSL mktemp returned unexpected path: {path!r}")
    return path


def wsl_copy_dir_contents(src_posix: str, dst_posix: str) -> None:
    """Recursively copy contents of *src_posix* into *dst_posix* (WSL paths)."""
    src = src_posix.replace("\\", "/")
    dst = dst_posix.replace("\\", "/")
    inner = f"mkdir -p '{dst}' && cp -a '{src}/.' '{dst}/'"
    result = subprocess.run(wsl_login_command(inner), capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(
            f"WSL copy failed ({src} -> {dst}): "
            f"{(result.stderr or result.stdout or '').strip()[:500]}"
        )


def wsl_copy_file(src_posix: str, dst_posix: str) -> None:
    """Copy a single file between WSL paths."""
    result = subprocess.run(
        ["wsl", "cp", src_posix.replace("\\", "/"), dst_posix.replace("\\", "/")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"WSL cp failed: {(result.stderr or result.stdout or '').strip()[:300]}"
        )


def wsl_copy_file_if_exists(src_posix: str, dst_posix: str) -> bool:
    """Copy a WSL file when present; return whether the copy ran."""
    src = src_posix.replace("\\", "/")
    check = subprocess.run(["wsl", "test", "-f", src], capture_output=True, timeout=15)
    if check.returncode != 0:
        return False
    wsl_copy_file(src, dst_posix.replace("\\", "/"))
    return True


def wsl_rm_rf(path_posix: str) -> None:
    """Remove a path inside WSL (best-effort)."""
    try:
        subprocess.run(
            ["wsl", "rm", "-rf", path_posix.replace("\\", "/")],
            capture_output=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass


def wsl_oat_available() -> bool:
    if sys.platform != "win32" or not oxdna_uses_wsl():
        return False
    return wsl_find_oat_path() is not None


def wsl_run_shell(inner: str, *, timeout: int | None = None) -> subprocess.CompletedProcess:
    """Execute a command string in WSL with login shell (conda PATH)."""
    return subprocess.run(
        wsl_login_command(inner),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def wsl_run(
    argv: list[str],
    *,
    cwd: str | None = None,
    timeout: int | None = None,
    capture_output: bool = True,
) -> subprocess.CompletedProcess:
    """
    Run a command inside WSL.

    ``argv`` is the Linux-side command, e.g. ``["oat", "PDB_oxDNA", ...]``.
  Paths in argv are converted from Windows when needed.
    """
    wsl_argv = [host_path_for_subprocess(a) if _looks_like_path(a) else a for a in argv]
    cmd: list[str] = ["wsl"]
    if cwd:
        cmd.extend(["--cd", host_path_for_subprocess(cwd)])
    cmd.extend(wsl_argv)

    return subprocess.run(
        cmd,
        capture_output=capture_output,
        text=capture_output,
        timeout=timeout,
    )


def _looks_like_path(arg: str) -> bool:
    if not arg or arg.startswith("-"):
        return False
    if arg.startswith("/"):
        return True
    if sys.platform == "win32" and re.match(r"^[A-Za-z]:\\", arg):
        return True
    return os.path.isabs(arg)
