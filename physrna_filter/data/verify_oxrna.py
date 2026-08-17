"""
Verify oxDNA / oxRNA is installed for the free-RNA simulation branch.

On Windows, install oxDNA inside WSL (Ubuntu) and either:
  - run PhysRNA from WSL, or
  - set OXDNA_BIN=wsl:/home/<you>/oxDNA/build/bin/oxDNA

Examples
--------
    python -m physrna_filter.data.verify_oxrna
    python -m physrna_filter.data.verify_oxrna --quick
"""
from __future__ import annotations

import argparse
import sys

from physrna_filter.simulation.oat_convert import (
    WSL_OAT_IMPORT_TEST_POWERSHELL,
    WSL_OAT_REINSTALL_HINT,
    WSL_OAT_REINSTALL_POWERSHELL,
    WSL_OXDNA_UTILS_TEST_POWERSHELL,
    smoke_test_conversion,
)
from physrna_filter.simulation.oxrna_env import describe_oxdna_setup, oxrna_ready, oxdna_available
from physrna_filter.simulation.run_simulation import smoke_test_simulation
from physrna_filter.simulation.wsl_paths import oxdna_uses_wsl


def _default_run_conversion_test() -> bool:
    """On Windows+WSL, always exercise the real PDB converter."""
    return sys.platform == "win32" and oxdna_uses_wsl()


def verify_oxrna(
    *,
    test_conversion: bool | None = None,
    test_simulation: bool = False,
) -> bool:
    print("oxDNA / oxRNA verification")
    print("=" * 60)
    rows = describe_oxdna_setup()
    for row in rows:
        print(f"  {row['label']}: {row['value']}")
    print()

    # Detect stale PhysRNA install (pre-PR-#16 labels).
    labels = {row["label"] for row in rows}
    if "wsl_PDB_oxDNA_import_ok" not in labels:
        print(
            "WARNING: installed PhysRNA is outdated (missing WSL PDB_oxDNA probe).\n"
            "  git fetch origin && git checkout cursor/oat-wsl-conversion-v2-8ae1\n"
            "  pip install -e . --no-deps\n"
        )

    if test_conversion is None:
        test_conversion = _default_run_conversion_test()

    conversion_ok = True
    if test_conversion:
        print("Running PDB → oxDNA conversion smoke test ...")
        conversion_ok, message = smoke_test_conversion()
        print(message)
        print()
        if not conversion_ok:
            print("FAILED: PDB conversion did not succeed.")
            if oxdna_available():
                print()
                print(WSL_OAT_REINSTALL_HINT)
                print()
                print("PowerShell reinstall:")
                print(f"  {WSL_OAT_REINSTALL_POWERSHELL}")
                print()
                print("PowerShell import checks:")
                print(f"  {WSL_OXDNA_UTILS_TEST_POWERSHELL}")
                print(f"  {WSL_OAT_IMPORT_TEST_POWERSHELL}")
            return False

    simulation_ok = True
    if test_simulation:
        print("Running short oxRNA MD smoke test ...")
        simulation_ok, message = smoke_test_simulation()
        print(message)
        print()
        if not simulation_ok:
            print("FAILED: oxRNA MD smoke test did not succeed.")
            log_dir = ""
            if "work_dir=" in message:
                log_dir = message.rsplit("work_dir=", 1)[-1].rstrip(")")
            if log_dir:
                log_path = f"{log_dir}/oxdna_run.log"
                print(f"Check {log_path} for the full oxDNA output.")
            return False

    if conversion_ok and oxdna_available():
        print("OK: oxDNA + PDB converter ready")
        if test_conversion:
            print("Conversion smoke test passed.")
        if test_simulation:
            print("oxRNA MD smoke test passed.")
        if not oxrna_ready():
            print(
                "Note: PDB_oxDNA module missing, but conversion works via "
                "oxDNA.utils in WSL (this is fine)."
            )
        print("The pipeline will use oxRNA (RNA2) instead of CG Langevin fallback.")
        return True

    if oxdna_available():
        print("PARTIAL: oxDNA binary found but PDB converter is missing or broken.")
        print()
        print(WSL_OAT_REINSTALL_HINT)
        print()
        print("PowerShell reinstall:")
        print(f"  {WSL_OAT_REINSTALL_POWERSHELL}")
        print()
        print("PowerShell import checks:")
        print(f"  {WSL_OXDNA_UTILS_TEST_POWERSHELL}")
        print(f"  {WSL_OAT_IMPORT_TEST_POWERSHELL}")
        return False

    print("NOT FOUND: oxDNA binary is missing or not on PATH.")
    print()
    print("Windows (recommended): use WSL Ubuntu")
    print("  wsl --install")
    print("  # inside WSL:")
    print("  sudo apt update && sudo apt install -y build-essential cmake git")
    print("  git clone https://github.com/lorenzo-rovigatti/oxDNA.git")
    print("  cd oxDNA && mkdir build && cd build")
    print("  cmake .. -DPython=OFF")
    print("  make -j4")
    print("  export PATH=$PWD/bin:$PATH")
    print("  oxDNA --version")
    print()
    print("Optional PDB converter (pip, inside WSL or native Linux):")
    print("  pip install oxDNA-analysis-tools")
    print()
    print("From Windows PowerShell (PhysRNA on Windows, oxDNA in WSL):")
    print(f"  {WSL_OAT_REINSTALL_POWERSHELL}")
    print("  $env:OXDNA_BIN = \"wsl:/home/<you>/oxDNA/build/bin/oxDNA\"")
    print("  $env:OAT_BIN   = \"wsl:/home/<you>/miniconda3/bin/oat\"")
    print("  python -m physrna_filter.data.verify_oxrna")
    print()
    print("Or run the whole pipeline inside WSL where oxDNA is on PATH.")
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify oxDNA / oxRNA setup")
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Skip PDB → oxDNA conversion smoke test (not recommended on Windows+WSL)",
    )
    ap.add_argument(
        "--test-conversion",
        action="store_true",
        help="(default on Windows+WSL) Force conversion smoke test",
    )
    ap.add_argument(
        "--test-simulation",
        action="store_true",
        help="Run a short oxRNA MD after conversion (Windows+WSL debugging)",
    )
    args = ap.parse_args()
    if args.quick:
        test_conversion = False
    elif args.test_conversion:
        test_conversion = True
    else:
        test_conversion = None
    ok = verify_oxrna(
        test_conversion=test_conversion,
        test_simulation=args.test_simulation,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
