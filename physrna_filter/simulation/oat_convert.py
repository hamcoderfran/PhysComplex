"""
WSL-side PDB → oxDNA conversion helpers.

Uses a small script file written into the simulation work directory to avoid
PowerShell / bash quoting issues when invoking miniconda Python in WSL.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .wsl_paths import (
    _WSL_CONDA_INIT,
    host_path_for_subprocess,
    normalize_wsl_posix_path,
    oxdna_uses_wsl,
    wsl_python_for_oat,
    wsl_run,
    wsl_run_shell,
)

_WSL_PDB_OXDNA_SCRIPT = """\
#!/usr/bin/env python3
\"\"\"PhysRNA helper: run PDB_oxDNA API and write .top/.dat files.\"\"\"
import sys
from pathlib import Path


def main() -> int:
    pdb_path, out_base = sys.argv[1], sys.argv[2]
    try:
        from oxDNA_analysis_tools.PDB_oxDNA import PDB_oxDNA
        from oxDNA_analysis_tools.UTILS.RyeReader import write_conf, write_top
    except ImportError as exc:
        print(f"PDB_oxDNA import failed: {exc}", file=sys.stderr)
        print(
            "Reinstall in WSL: cd ~/oxDNA/analysis && pip install . "
            "(do not use the old PyPI oxDNA-analysis-tools 2.0.5 wheel)",
            file=sys.stderr,
        )
        return 2

    pdb_str = Path(pdb_path).read_text()
    configs, systems = PDB_oxDNA(pdb_str)
    if not configs or not systems:
        print("PDB_oxDNA: no nucleotides parsed from PDB", file=sys.stderr)
        return 3

    for i, (conf, system) in enumerate(zip(configs, systems)):
        suffix = "" if len(configs) == 1 else f"_{i}"
        top = out_base + suffix + ".top"
        dat = out_base + suffix + ".dat"
        write_top(top, system)
        write_conf(dat, conf)
        print(f"wrote {top}, {dat}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""

WSL_OAT_REINSTALL_HINT = (
    "WSL PDB_oxDNA module is missing. Do NOT use the old PyPI wheel "
    "(oxDNA-analysis-tools 2.0.5 lacks PDB_oxDNA).\n"
    "Install from your oxDNA git clone instead:\n"
    "  source ~/miniconda3/etc/profile.d/conda.sh && conda activate base\n"
    "  pip uninstall -y oxDNA-analysis-tools oxdna-analysis-tools\n"
    "  cd ~/oxDNA/analysis && pip install .\n"
    "If ~/oxDNA is missing: git clone https://github.com/lorenzo-rovigatti/oxDNA.git ~/oxDNA\n"
    "Or rebuild oxDNA with Python bindings: cd ~/oxDNA/build && cmake .. -DPython=ON && make -j4 && make install"
)

# PowerShell: install from oxDNA/analysis (not stale PyPI 2.0.5 wheel).
WSL_OAT_REINSTALL_POWERSHELL = (
    'wsl bash -lc "source ~/miniconda3/etc/profile.d/conda.sh && '
    "conda activate base && pip uninstall -y oxDNA-analysis-tools oxdna-analysis-tools "
    '&& cd ~/oxDNA/analysis && pip install ."'
)

WSL_OAT_IMPORT_TEST_POWERSHELL = (
    'wsl /home/frann/miniconda3/bin/python -c '
    '"import oxDNA_analysis_tools.PDB_oxDNA; print(' + "'PDB_oxDNA OK')" + '"'
)

WSL_OXDNA_UTILS_TEST_POWERSHELL = (
    'wsl /home/frann/miniconda3/bin/python -c '
    '"import oxDNA.utils.pdb_to_oxdna; print(' + "'oxDNA.utils OK')" + '"'
)


def wsl_python_module_ok(python_bin: str, import_stmt: str) -> bool:
    """Return True when WSL *python_bin* can execute ``import_stmt``."""
    py = normalize_wsl_posix_path(python_bin)
    try:
        if sys.platform == "win32":
            result = subprocess.run(
                ["wsl", py, "-c", import_stmt],
                capture_output=True,
                text=True,
                timeout=30,
            )
        else:
            result = subprocess.run(
                [py, "-c", import_stmt],
                capture_output=True,
                timeout=30,
            )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def wsl_oat_pdb_module_ok(oat_bin: str) -> bool:
    """Return True when WSL miniconda can import ``oxDNA_analysis_tools.PDB_oxDNA``."""
    return wsl_python_module_ok(
        wsl_python_for_oat(oat_bin),
        "import oxDNA_analysis_tools.PDB_oxDNA",
    )


def wsl_oxdna_utils_ok(oat_bin: str) -> bool:
    """Return True when WSL Python can import ``oxDNA.utils.pdb_to_oxdna``."""
    return wsl_python_module_ok(
        wsl_python_for_oat(oat_bin),
        "import oxDNA.utils.pdb_to_oxdna",
    )


def write_wsl_pdb_oxdna_script(work_dir: str) -> str:
    """Write the bundled conversion script into *work_dir* and return its path."""
    path = os.path.join(work_dir, "_physrna_pdb_oxdna.py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(_WSL_PDB_OXDNA_SCRIPT)
    return path


def run_wsl_pdb_oxdna_script(
    *,
    oat_bin: str,
    work_dir: str,
    pdb_path: str,
    out_base: str,
    timeout: int = 120,
) -> subprocess.CompletedProcess:
    """
    Run the bundled PDB_oxDNA script via WSL miniconda Python.

    Paths may be Windows or Linux; they are converted when needed.
    """
    script_path = write_wsl_pdb_oxdna_script(work_dir)
    py = wsl_python_for_oat(oat_bin)
    return wsl_run(
        [
            py,
            host_path_for_subprocess(script_path),
            host_path_for_subprocess(pdb_path),
            host_path_for_subprocess(out_base),
        ],
        cwd=work_dir,
        timeout=timeout,
    )


def run_native_pdb_oxdna_script(
    *,
    work_dir: str,
    pdb_path: str,
    out_base: str,
    python_exe: str | None = None,
    timeout: int = 120,
) -> subprocess.CompletedProcess:
    """Run the bundled PDB_oxDNA script with the active Python interpreter."""
    script_path = write_wsl_pdb_oxdna_script(work_dir)
    py = python_exe or sys.executable
    return subprocess.run(
        [py, script_path, pdb_path, out_base],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=work_dir,
    )


def smoke_test_conversion(pdb_path: str | None = None) -> tuple[bool, str]:
    """
    Build an extended RNA PDB and attempt PDB → oxDNA conversion.

    Returns (success, message). Safe to call from Windows PowerShell via
    ``python -m physrna_filter.data.verify_oxrna --test-conversion``.
    """
    from .fold_rna import _build_extended_strand
    from .run_simulation import _pdb_to_oxrna

    work = tempfile.mkdtemp(prefix="physrna_oat_test_")
    pdb = pdb_path or os.path.join(work, "test_rna.pdb")
    if pdb_path is None:
        _build_extended_strand("AUGCAUGC", pdb)

    top_path = os.path.join(work, "rna.top")
    conf_path = os.path.join(work, "rna.conf")
    try:
        _pdb_to_oxrna(pdb, top_path, conf_path)
    except RuntimeError as exc:
        return False, str(exc)

    if not (os.path.isfile(top_path) and os.path.isfile(conf_path)):
        return False, f"conversion reported success but outputs missing in {work}"

    return True, f"OK — wrote {top_path} and {conf_path}"
