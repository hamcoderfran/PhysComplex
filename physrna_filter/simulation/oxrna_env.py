"""
oxDNA / oxRNA discovery helpers.

PhysRNA uses the oxDNA binary (RNA2 force field) for free-RNA MD.  On Windows
the supported path is WSL (Linux); native Windows binaries are not provided.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .wsl_paths import oxdna_uses_wsl, wsl_find_oat_path, wsl_oat_available


def _importable(name: str) -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _wsl_oat_python_path() -> str | None:
    raw = os.environ.get("OAT_BIN", "").strip()
    if raw.lower().startswith("wsl:"):
        return raw[4:].strip() or None
    return wsl_find_oat_path()


def _pdb_converter_available() -> bool:
    """True when a working PDB → oxDNA converter is importable on this host."""
    if _importable("oxDNA.utils.pdb_to_oxdna"):
        return True
    if _importable("oxDNA_analysis_tools.PDB_oxDNA"):
        return True
    if oxdna_uses_wsl() and sys.platform == "win32":
        from .oat_convert import wsl_oat_pdb_module_ok, wsl_oxdna_utils_ok

        wsl_path = _wsl_oat_python_path()
        if wsl_path:
            if wsl_oxdna_utils_ok(wsl_path):
                return True
            if wsl_oat_pdb_module_ok(wsl_path):
                return True
    return False


def resolve_oxdna_command() -> list[str] | None:
    """
    Return argv prefix to launch oxDNA, or None if not found.

    Environment:
      OXDNA_BIN — path to oxDNA executable, or a WSL command prefixed with
                  ``wsl:``, e.g. ``wsl:/home/you/oxDNA/build/bin/oxDNA``
    """
    raw = os.environ.get("OXDNA_BIN", "oxDNA").strip()
    if not raw:
        raw = "oxDNA"

    if raw.lower().startswith("wsl:"):
        wsl_path = raw[4:].strip()
        if not wsl_path:
            return None
        return ["wsl", wsl_path]

    if " " in raw and not Path(raw).exists():
        return raw.split()

    path = shutil.which(raw) if not Path(raw).exists() else raw
    if not path:
        return None
    return [path]


def resolve_oat_command() -> list[str] | None:
    """
    Return argv prefix for ``oat`` (oxDNA-analysis-tools), or None.

    Environment:
      OAT_BIN — optional explicit path, e.g.
                ``wsl:/home/you/miniconda3/bin/oat``

    On Windows with ``OXDNA_BIN=wsl:...``, probes WSL conda paths.
    """
    raw = os.environ.get("OAT_BIN", "").strip()
    if raw:
        if raw.lower().startswith("wsl:"):
            path = raw[4:].strip()
            return ["wsl", path] if path else None
        path = shutil.which(raw) if not Path(raw).exists() else raw
        return [path] if path else None

    oat = shutil.which("oat")
    if oat:
        return [oat]

    if oxdna_uses_wsl():
        wsl_path = wsl_find_oat_path()
        if wsl_path:
            return ["wsl", wsl_path]

    return None


def oat_available() -> bool:
    return _pdb_converter_available()


def oxrna_ready() -> bool:
    """True when oxDNA binary and a working PDB converter are available."""
    return oxdna_available() and _pdb_converter_available()


def oxdna_available() -> bool:
    cmd = resolve_oxdna_command()
    if not cmd:
        return False
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        # oxDNA v3.x prints usage when invoked without an input file
        combined = (result.stdout or "") + (result.stderr or "")
        if "Usage is" in combined and "oxDNA" in combined:
            return True
        # older builds may support --version
        ver = subprocess.run(
            [*cmd, "--version"],
            capture_output=True,
            timeout=10,
        )
        return ver.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def describe_oxdna_setup() -> list[dict]:
    """Rows for verify_oxrna / diagnostics."""
    rows: list[dict] = []
    env = os.environ.get("OXDNA_BIN", "")
    rows.append({
        "label": "OXDNA_BIN",
        "value": env or "(not set — using PATH lookup for 'oxDNA')",
    })
    oat_env = os.environ.get("OAT_BIN", "")
    rows.append({
        "label": "OAT_BIN",
        "value": oat_env or "(not set — auto-detect in WSL)",
    })

    cmd = resolve_oxdna_command()
    rows.append({
        "label": "resolved_command",
        "value": " ".join(cmd) if cmd else "(not found)",
    })

    rows.append({
        "label": "binary_ok",
        "value": str(oxdna_available()),
    })

    rows.append({
        "label": "wsl_backend",
        "value": str(oxdna_uses_wsl()),
    })

    rows.append({
        "label": "oat_ok",
        "value": str(oat_available()),
    })

    rows.append({
        "label": "oat_cli_resolved",
        "value": " ".join(resolve_oat_command()) if resolve_oat_command() else "(not found)",
    })

    rows.append({
        "label": "oxrna_ready",
        "value": str(oxrna_ready()),
    })

    oat_cmd = resolve_oat_command()
    rows.append({
        "label": "resolved_oat",
        "value": " ".join(oat_cmd) if oat_cmd else "(not found)",
    })

    for name in ("oat", "oxDNA"):
        path = shutil.which(name)
        rows.append({
            "label": f"which({name})",
            "value": path or "(not on PATH)",
        })

    # legacy Python module shipped with a source oxDNA build
    try:
        import importlib.util
        has_mod = importlib.util.find_spec("oxDNA") is not None
    except Exception:
        has_mod = False
    rows.append({
        "label": "python_module_oxDNA",
        "value": str(has_mod),
    })

    try:
        import importlib.util
        has_oat = importlib.util.find_spec("oxDNA_analysis_tools.PDB_oxDNA") is not None
    except Exception:
        has_oat = False
    rows.append({
        "label": "python_module_PDB_oxDNA",
        "value": str(has_oat),
    })

    if oxdna_uses_wsl() and sys.platform == "win32":
        from .oat_convert import wsl_oat_pdb_module_ok, wsl_oxdna_utils_ok

        wsl_path = _wsl_oat_python_path()
        if wsl_path:
            rows.append({
                "label": "wsl_oxDNA_utils_ok",
                "value": str(wsl_oxdna_utils_ok(wsl_path)),
            })
            rows.append({
                "label": "wsl_PDB_oxDNA_import_ok",
                "value": str(wsl_oat_pdb_module_ok(wsl_path)),
            })

    rows.append({
        "label": "platform",
        "value": sys.platform,
    })
    return rows
