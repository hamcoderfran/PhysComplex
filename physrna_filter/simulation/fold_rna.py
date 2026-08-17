"""
Generates an initial 3D RNA structure from sequence for simulation.

The free-RNA conformational ensemble requires a 3D starting structure built
from the RNA sequence alone — explicitly without the protein — so that the
simulation explores unbound conformational space.

Preference order:
  1. RNAfold (ViennaRNA) for 2D secondary structure -> FARFAR2 for 3D
  2. SimRNA for combined 2D+3D
  3. Fallback: extended (unfolded) strand via BioPython/internal builder
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile


def fold_rna_sequence(
    sequence: str,
    output_pdb: str,
    method: str = "auto",
) -> str:
    """
    Folds an RNA sequence into a 3D starting structure.

    Args:
        sequence:   RNA sequence string (e.g. "AUGCAUGC")
        output_pdb: path to write the output PDB
        method:     "rnafold+farfar2" | "simrna" | "extended" | "auto"

    Returns path to the generated PDB file.
    """
    os.makedirs(os.path.dirname(output_pdb) or ".", exist_ok=True)

    if method == "auto":
        method = _detect_available_method()

    print(f"Folding {len(sequence)}-nt RNA using method: {method}")

    if method == "rnafold+farfar2":
        return _fold_rnafold_farfar2(sequence, output_pdb)
    elif method == "simrna":
        return _fold_simrna(sequence, output_pdb)
    else:
        return _build_extended_strand(sequence, output_pdb)


def _detect_available_method() -> str:
    if _command_exists("RNAfold") and _command_exists("rna_denovo"):
        return "rnafold+farfar2"
    if _command_exists("SimRNA"):
        return "simrna"
    return "extended"


def _fold_rnafold_farfar2(sequence: str, output_pdb: str) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        # step 1: predict secondary structure
        ss_path = os.path.join(tmpdir, "secondary.txt")
        result = subprocess.run(
            ["RNAfold", "--noPS"],
            input=sequence,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"RNAfold failed: {result.stderr}")

        lines = result.stdout.strip().splitlines()
        dot_bracket = lines[1].split()[0] if len(lines) > 1 else "." * len(sequence)

        with open(ss_path, "w") as f:
            f.write(f"{sequence}\n{dot_bracket}\n")

        # step 2: build 3D with FARFAR2 (Rosetta rna_denovo)
        fasta_path = os.path.join(tmpdir, "rna.fasta")
        with open(fasta_path, "w") as f:
            f.write(f">rna\n{sequence.lower()}\n")

        result2 = subprocess.run(
            [
                "rna_denovo",
                f"-fasta {fasta_path}",
                f"-secstruct_file {ss_path}",
                f"-out:file:silent {tmpdir}/out.silent",
                "-nstruct 1",
                "-minimize_rna true",
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result2.returncode != 0:
            print(f"FARFAR2 warning: {result2.stderr[:200]}")

        # extract PDB from silent file
        silent = os.path.join(tmpdir, "out.silent")
        if os.path.exists(silent):
            subprocess.run(
                ["extract_pdbs", f"-silent {silent}", f"-out:path:pdb {tmpdir}"],
                timeout=60,
            )
            pdb_files = [f for f in os.listdir(tmpdir) if f.endswith(".pdb")]
            if pdb_files:
                import shutil
                shutil.copy(os.path.join(tmpdir, pdb_files[0]), output_pdb)
                return output_pdb

    # fallback if extraction failed
    return _build_extended_strand(sequence, output_pdb)


def _fold_simrna(sequence: str, output_pdb: str) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        seq_path = os.path.join(tmpdir, "sequence.txt")
        with open(seq_path, "w") as f:
            f.write(sequence)

        result = subprocess.run(
            ["SimRNA", "-s", seq_path, "-o", os.path.join(tmpdir, "simrna_out")],
            capture_output=True,
            text=True,
            timeout=600,
        )

        out_pdb = os.path.join(tmpdir, "simrna_out_01.pdb")
        if os.path.exists(out_pdb):
            import shutil
            shutil.copy(out_pdb, output_pdb)
            return output_pdb

    return _build_extended_strand(sequence, output_pdb)


def _build_extended_strand(sequence: str, output_pdb: str) -> str:
    """
    Builds an extended A-form RNA strand with enough backbone and base atoms for
    ``oat PDB_oxDNA`` (requires C3'/C5' and ring atoms C2,C4,C5,C6,N1,N3).
    """
    import math

    RISE = 2.81
    TWIST = 32.7
    RADIUS = 9.0
    one_to_three = {"A": "A  ", "U": "U  ", "G": "G  ", "C": "C  "}

    # Local ring coordinates (Angstrom) relative to C1' for oat's ring normal calc.
    _PYRIMIDINE_RING = {
        "N1": (0.0, 0.5, 1.0),
        "C2": (1.1, 0.3, 1.3),
        "N3": (1.8, 0.0, 0.3),
        "C4": (1.3, -0.2, -0.9),
        "C5": (0.0, -0.3, -1.2),
        "C6": (-0.9, 0.0, 0.0),
    }
    _PURINE_RING = {
        "N9": (0.0, 0.5, 1.0),
        "N1": (-1.0, 0.0, 0.8),
        "C2": (-1.3, 0.5, -0.2),
        "N3": (-0.6, 0.3, -1.2),
        "C4": (0.5, -0.2, -1.3),
        "C5": (1.2, 0.0, -0.5),
        "C6": (1.0, 0.2, 0.6),
    }

    lines = ["REMARK  Extended A-form RNA strand (oxRNA / oat starting point)"]
    atom_num = 1

    def _atom(name: str, element: str, resname: str, resnum: int, xyz) -> None:
        nonlocal atom_num
        x, y, z = xyz
        lines.append(
            f"ATOM  {atom_num:5d}  {name:<4}{resname} A{resnum:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2}"
        )
        atom_num += 1

    for i, base in enumerate(sequence.upper()):
        angle = math.radians(i * TWIST)
        z = i * RISE
        resname = one_to_three.get(base, "A  ")

        def pt(r_offset: float, ang_offset: float = 0.0, z_offset: float = 0.0):
            a = angle + math.radians(ang_offset)
            r = RADIUS + r_offset
            return (r * math.cos(a), r * math.sin(a), z + z_offset)

        def local_from_c1(c1_xyz, lx: float, ly: float, lz: float):
            ca, sa = math.cos(angle), math.sin(angle)
            return (
                c1_xyz[0] + lx * ca - ly * sa,
                c1_xyz[1] + lx * sa + ly * ca,
                c1_xyz[2] + lz,
            )

        _atom("P  ", "P", resname, i + 1, pt(1.6, -8, -1.2))
        _atom("OP1", "O", resname, i + 1, pt(2.0, -12, -1.0))
        _atom("OP2", "O", resname, i + 1, pt(2.0, -4, -1.4))
        _atom("O5'", "O", resname, i + 1, pt(0.9, -5, -0.6))
        _atom("C5'", "C", resname, i + 1, pt(0.5, -3, -0.3))
        c4 = pt(0.0, 0, 0.0)
        _atom("C4'", "C", resname, i + 1, c4)
        _atom("O4'", "O", resname, i + 1, pt(-0.6, 2, 0.2))
        _atom("C3'", "C", resname, i + 1, pt(-0.4, -2, 0.4))
        _atom("O3'", "O", resname, i + 1, pt(-0.8, -4, 0.6))
        _atom("C2'", "C", resname, i + 1, pt(-0.2, -1, 0.2))
        _atom("O2'", "O", resname, i + 1, pt(0.3, -1.5, 1.0))
        c1 = pt(-1.1, 5, 0.5)
        _atom("C1'", "C", resname, i + 1, c1)

        ring = _PURINE_RING if base in ("A", "G") else _PYRIMIDINE_RING
        ring_elements = {
            "N1": "N", "N3": "N", "N9": "N",
            "C2": "C", "C4": "C", "C5": "C", "C6": "C",
        }
        for atom_name, offset in ring.items():
            _atom(
                f"{atom_name} ",
                ring_elements.get(atom_name, "C"),
                resname,
                i + 1,
                local_from_c1(c1, *offset),
            )

    lines.append("END")
    with open(output_pdb, "w") as f:
        f.write("\n".join(lines) + "\n")
    return output_pdb


def _command_exists(name: str) -> bool:
    return shutil.which(name) is not None
