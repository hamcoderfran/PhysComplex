"""
Runs a coarse-grained oxRNA simulation of the free (unbound) RNA and extracts
per-frame coordinate snapshots for the interface nucleotides.

Simulation priority:
  1. oxRNA (if oxDNA binary installed) — gold standard coarse-grained MD
  2. Internal C4' Langevin MD — independent of AF3 coords (non-circular)
  3. Never uses AF3 bound coordinates for the free ensemble

oxRNA is chosen because:
  - It is specifically parameterized for RNA (not just DNA)
  - It runs on consumer GPUs in minutes for sequences under ~50 nt
  - It accurately captures the A-form helix preference and loop flexibility
    relevant to RBP binding motifs from CLIP-seq data
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from Bio.PDB import PDBParser

from .fold_rna import fold_rna_sequence
from .cg_rna_md import run_cg_langevin_md
from .oat_convert import (
    WSL_OAT_REINSTALL_HINT,
    run_native_pdb_oxdna_script,
    run_wsl_pdb_oxdna_script,
    wsl_oat_pdb_module_ok,
    wsl_python_for_oat,
)
from .oxrna_env import oxdna_available, oxrna_ready, resolve_oat_command, resolve_oxdna_command
from .wsl_paths import (
    _WSL_CONDA_INIT,
    host_path_for_subprocess,
    oxdna_uses_wsl,
    wsl_copy_dir_contents,
    wsl_copy_file_if_exists,
    wsl_mktemp_dir,
    wsl_rm_rf,
    wsl_run,
    wsl_run_shell,
)
from ..structure.extract_interface import extract_interface_coords, rna_residue_to_trajectory_indices
from ..structure.local_geometry import extract_geometry, geometry_to_matrix


def _default_oxrna_relax_steps(production_steps: int) -> int:
    """Backbone relaxation before production MD; override with PHYSRNA_OXRNA_RELAX_STEPS."""
    raw = os.environ.get("PHYSRNA_OXRNA_RELAX_STEPS", "").strip()
    if raw:
        return max(1000, int(raw))
    if production_steps <= 10_000:
        return max(3_000, production_steps)
    return 50_000


def _default_oxrna_steps() -> int:
    """Panel-friendly default; override with PHYSRNA_OXRNA_STEPS or PHYSRNA_OXRNA_PROFILE."""
    raw = os.environ.get("PHYSRNA_OXRNA_STEPS", "").strip()
    if raw:
        try:
            return max(1000, int(raw))
        except ValueError:
            pass
    profile = os.environ.get("PHYSRNA_OXRNA_PROFILE", "panel").strip().lower()
    if profile == "production":
        return 1_000_000
    if profile == "fast":
        return 10_000
    return 50_000


def simulate_free_rna(
    rna_chains: list,
    interface_residues: list[tuple[str, int]],
    n_steps: int | None = None,
    output_freq: int = 1_000,
    temperature: float = 300.0,
    work_dir: str | None = None,
    require_oxrna: bool = False,
    n_frames: int = 500,
    partner_rna_chains: list | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Simulates the free RNA and extracts interface snapshots.

    The free ensemble is built from sequence-folded starting structures only —
    AF3 bound coordinates are never used, eliminating circular entropic scoring.

    Args:
        require_oxrna: if True, raise when oxRNA is not installed
        n_frames:      frames to collect (for CG MD fallback)

    Returns:
        coord_snapshots   — (n_frames, n_interface_atoms, 3) C4' coordinates
        geom_snapshots    — (n_frames, n_nuc * 9) geometry feature vectors
    """
    from ..structure.parse_complex import get_rna_sequence

    sim_chains = partner_rna_chains or rna_chains
    if not sim_chains:
        raise ValueError("No RNA chains available for simulation")

    sequence = get_rna_sequence(sim_chains[0])
    if n_steps is None:
        n_steps = _default_oxrna_steps()

    work_dir = work_dir or tempfile.mkdtemp(prefix="physrna_sim_")
    os.makedirs(work_dir, exist_ok=True)

    start_pdb = os.path.join(work_dir, "free_rna_start.pdb")
    fold_rna_sequence(sequence, start_pdb)

    simulation_method = "unknown"

    if oxrna_ready():
        try:
            simulation_method = "oxrna"
            traj_path = _run_oxrna(start_pdb, work_dir, n_steps, output_freq, temperature)
            coord_snaps, geom_snaps = _extract_snapshots_oxrna(
                traj_path, sim_chains, interface_residues, work_dir
            )
            geom_snaps = _enrich_geometry_from_folded(
                start_pdb, sim_chains, interface_residues, coord_snaps, geom_snaps
            )
        except RuntimeError as exc:
            if require_oxrna:
                raise
            simulation_method = "cg_langevin"
            print(
                f"oxRNA failed ({str(exc).replace(chr(10), ' ')[:240]}) — "
                "running internal C4' Langevin MD"
            )
            coord_snaps, geom_snaps = run_cg_langevin_md(
                start_pdb, interface_residues, n_frames=n_frames, rna_chains=sim_chains
            )
            geom_snaps = _enrich_geometry_from_folded(
                start_pdb, rna_chains, interface_residues, coord_snaps, geom_snaps
            )
    else:
        if require_oxrna:
            raise RuntimeError(
                "oxRNA/oxDNA not installed. Install from "
                "https://github.com/lorenzo-rovigatti/oxDNA "
                "or set require_oxrna=False."
            )
        simulation_method = "cg_langevin"
        print(
            "oxRNA not found — running internal C4' Langevin MD "
            "(independent of AF3 coordinates)"
        )
        coord_snaps, geom_snaps = run_cg_langevin_md(
            start_pdb, interface_residues, n_frames=n_frames, rna_chains=sim_chains
        )

        # Extract geometry from folded structure for interface nucleotides
        geom_snaps = _enrich_geometry_from_folded(
            start_pdb, sim_chains, interface_residues, coord_snaps, geom_snaps
        )

    print(
        f"Simulation ({simulation_method}): {len(coord_snaps)} snapshots, "
        f"{coord_snaps.shape[1]} interface atoms"
    )
    return coord_snaps, geom_snaps, simulation_method


def _enrich_geometry_from_folded(
    start_pdb: str,
    rna_chains: list,
    interface_residues: list[tuple[str, int]],
    coord_snaps: np.ndarray,
    geom_snaps: np.ndarray,
) -> np.ndarray:
    """
    Replace CG-proxy geometry with real torsion angles from the folded PDB
    when available, then add Langevin noise per frame.
    """
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("rna", start_pdb)
        chains = list(structure[0])
        geom_dict = extract_geometry(chains, interface_residues)
        base_geom = geometry_to_matrix(geom_dict, interface_residues)
        rng = np.random.default_rng(42)
        enriched = []
        for i in range(len(geom_snaps)):
            noise = rng.normal(scale=12.0, size=base_geom.shape)
            enriched.append(base_geom + noise)
        return np.array(enriched)
    except Exception:
        return geom_snaps


def _oxrna_backend() -> str:
    """Choose oxDNA backend. WSL runs must use CPU (Windows torch.cuda ≠ WSL oxDNA CUDA)."""
    if oxdna_uses_wsl():
        return "CPU"
    return "CUDA" if _cuda_available() else "CPU"


def _oxdna_repo_root_from_bin(oxdna_bin: str) -> str | None:
    """Guess oxDNA source root from the ``oxDNA`` executable path."""
    from .wsl_paths import normalize_wsl_posix_path

    path = normalize_wsl_posix_path(oxdna_bin)
    parts = [p for p in path.split("/") if p]
    if "build" in parts:
        idx = parts.index("build")
        return "/" + "/".join(parts[:idx])
    if len(parts) >= 2 and parts[-2] == "bin":
        return "/" + "/".join(parts[:-2])
    return None


def _stage_oxdna_data_files(work_dir: str) -> bool:
    """
    Copy ``rna_sequence_dependent_parameters.txt`` into *work_dir* when possible.

    Returns True when the file is present in *work_dir* after staging.
    """
    dest = os.path.join(work_dir, "rna_sequence_dependent_parameters.txt")
    if os.path.isfile(dest):
        return True

    candidates: list[str] = []
    data_env = os.environ.get("OXDNA_DATA", "").strip()
    if data_env:
        root = data_env[4:].strip() if data_env.lower().startswith("wsl:") else data_env
        candidates.append(os.path.join(root, "rna_sequence_dependent_parameters.txt"))

    oxdna_cmd = resolve_oxdna_command()
    if oxdna_cmd:
        root = _oxdna_repo_root_from_bin(oxdna_cmd[-1])
        if root:
            candidates.append(f"{root}/rna_sequence_dependent_parameters.txt")

    if oxdna_uses_wsl() and os.name == "nt":
        wsl_dest = host_path_for_subprocess(dest)
        for src in candidates:
            src = src.replace("\\", "/")
            check = subprocess.run(["wsl", "test", "-f", src], capture_output=True, timeout=15)
            if check.returncode != 0:
                continue
            copy = subprocess.run(["wsl", "cp", src, wsl_dest], capture_output=True, timeout=30)
            if copy.returncode == 0 and os.path.isfile(dest):
                return True
        return False

    for src in candidates:
        if os.path.isfile(src):
            shutil.copy2(src, dest)
            return True
    return False


_OXDNA_NATIVE_OUTPUTS = ("log.dat", "energy.dat", "last_conf.dat")


def _read_text_tail(path: str, *, max_chars: int = 3500) -> str | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read().strip()
    except OSError:
        return None
    if not text:
        return None
    if len(text) > max_chars:
        return text[-max_chars:]
    return text


def _pull_wsl_native_outputs(
    native_work: str,
    wsl_host_work: str,
    *,
    traj_name: str | None = None,
    extra_names: tuple[str, ...] = (),
) -> None:
    """Copy oxDNA outputs from WSL-native work dir back to the host work dir."""
    if traj_name:
        wsl_copy_file_if_exists(
            f"{native_work}/{traj_name}",
            f"{wsl_host_work}/{traj_name}",
        )
    for name in (*_OXDNA_NATIVE_OUTPUTS, *extra_names):
        wsl_copy_file_if_exists(f"{native_work}/{name}", f"{wsl_host_work}/{name}")


def _format_oxdna_failure(
    result: subprocess.CompletedProcess,
    work_dir: str | None = None,
) -> str:
    chunks: list[str] = []
    if result.stdout and result.stdout.strip():
        chunks.append(result.stdout.strip())
    if result.stderr and result.stderr.strip():
        chunks.append(result.stderr.strip())
    if work_dir:
        for log_name in ("relax_log.dat", "log.dat"):
            log_text = _read_text_tail(os.path.join(work_dir, log_name))
            if log_text:
                chunks.append(f"--- {log_name} ---\n{log_text}")
        for energy_name in ("relax_energy.dat", "energy.dat"):
            energy_text = _read_text_tail(os.path.join(work_dir, energy_name), max_chars=1200)
            if energy_text:
                energy_lines = energy_text.splitlines()[-8:]
                chunks.append(f"--- {energy_name} (last lines) ---\n" + "\n".join(energy_lines))
                break
    if not chunks:
        return f"(exit code {result.returncode}, no output)"
    return "\n".join(chunks)[:8000]


def _save_oxdna_log(work_dir: str, result: subprocess.CompletedProcess) -> None:
    log_path = os.path.join(work_dir, "oxdna_run.log")
    try:
        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write(_format_oxdna_failure(result, work_dir))
    except OSError:
        pass


def _execute_oxdna(work_dir: str, input_path: str) -> subprocess.CompletedProcess:
    """Run oxDNA once; on WSL+Windows stages work dir under ``/tmp``."""
    oxdna_cmd = resolve_oxdna_command()
    if not oxdna_cmd:
        raise RuntimeError("oxDNA binary not found (set OXDNA_BIN)")

    input_name = os.path.basename(input_path)
    extra_outputs = (
        "relaxed.conf",
        "last_conf.dat",
        "relax_traj.dat",
        "relax_log.dat",
        "relax_energy.dat",
    )

    if oxdna_uses_wsl() and os.name == "nt":
        oxdna_bin = oxdna_cmd[-1]
        wsl_host_work = host_path_for_subprocess(work_dir)
        native_work = wsl_mktemp_dir()
        try:
            wsl_copy_dir_contents(wsl_host_work, native_work)
            result = wsl_run([oxdna_bin, input_name], cwd=native_work, timeout=3600)
            _pull_wsl_native_outputs(
                native_work,
                wsl_host_work,
                traj_name=os.path.basename(
                    _oxdna_input_value(input_path, "trajectory_file") or ""
                ) or None,
                extra_names=extra_outputs,
            )
        finally:
            wsl_rm_rf(native_work)
        return result

    return subprocess.run(
        [*oxdna_cmd, input_path],
        cwd=work_dir,
        capture_output=True,
        text=True,
        timeout=3600,
    )


def _oxdna_input_value(input_path: str, key: str) -> str | None:
    try:
        with open(input_path, encoding="utf-8") as handle:
            prefix = f"{key} ="
            for line in handle:
                if line.strip().startswith(prefix):
                    return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return None


def _run_oxrna(
    start_pdb: str,
    work_dir: str,
    n_steps: int,
    output_freq: int,
    temperature: float,
) -> str:
    """Writes oxRNA input files and runs the simulation."""
    top_path  = os.path.join(work_dir, "rna.top")
    conf_path = os.path.join(work_dir, "rna.conf")
    traj_path = os.path.join(work_dir, "trajectory.dat")
    relaxed_conf = os.path.join(work_dir, "relaxed.conf")

    _pdb_to_oxrna(start_pdb, top_path, conf_path)

    has_seq_dep = _stage_oxdna_data_files(work_dir)
    backend = _oxrna_backend()
    relax_steps = _default_oxrna_relax_steps(n_steps)

    relax_input = os.path.join(work_dir, "input_relax.dat")
    _write_oxrna_input(
        relax_input,
        top_path,
        conf_path,
        os.path.join(work_dir, "relax_traj.dat"),
        relax_steps,
        output_freq,
        temperature,
        use_sequence_dependent=has_seq_dep,
        backend=backend,
        phase="relax",
        lastconf_path=relaxed_conf,
    )
    _patch_oxrna_backend(relax_input, backend)

    relax_result = _execute_oxdna(work_dir, relax_input)
    if relax_result.returncode != 0:
        _save_oxdna_log(work_dir, relax_result)
        raise RuntimeError(
            "oxRNA relaxation failed:\n"
            f"{_format_oxdna_failure(relax_result, work_dir)}"
        )

    prod_conf = relaxed_conf if os.path.isfile(relaxed_conf) else os.path.join(work_dir, "last_conf.dat")
    if not os.path.isfile(prod_conf):
        raise RuntimeError(
            "oxRNA relaxation finished but no relaxed configuration was written "
            f"(expected {relaxed_conf} or last_conf.dat)"
        )

    input_path = os.path.join(work_dir, "input.dat")
    _write_oxrna_input(
        input_path,
        top_path,
        prod_conf,
        traj_path,
        n_steps,
        output_freq,
        temperature,
        use_sequence_dependent=has_seq_dep,
        backend=backend,
        phase="production",
    )
    _patch_oxrna_backend(input_path, backend)

    result = _execute_oxdna(work_dir, input_path)
    if result.returncode != 0:
        _save_oxdna_log(work_dir, result)
        raise RuntimeError(
            f"oxRNA simulation failed:\n{_format_oxdna_failure(result, work_dir)}"
        )

    if not os.path.isfile(traj_path):
        raise RuntimeError(f"oxRNA simulation produced no trajectory at {traj_path}")

    return traj_path


def _patch_oxrna_backend(input_path: str, backend: str) -> None:
    with open(input_path) as f:
        content = f.read()
    content = content.replace("backend = CPU", f"backend = {backend}")
    with open(input_path, "w") as f:
        f.write(content)


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _write_oxrna_input(
    input_path: str,
    top_path: str,
    conf_path: str,
    traj_path: str,
    n_steps: int,
    output_freq: int,
    temperature: float,
    *,
    use_sequence_dependent: bool = True,
    backend: str = "CPU",
    phase: str = "production",
    lastconf_path: str | None = None,
) -> None:
    # Use basenames so oxDNA can run with cwd=work_dir (required for WSL bridge).
    seq_lines = (
        "use_average_seq = no\n"
        "seq_dep_file = rna_sequence_dependent_parameters.txt\n"
        if use_sequence_dependent
        else "use_average_seq = yes\n"
    )
    precision_line = "backend_precision = mixed\n" if backend == "CUDA" else ""
    lastconf = os.path.basename(lastconf_path or os.path.join(os.path.dirname(conf_path), "last_conf.dat"))
    if phase == "relax":
        energy_name = "relax_energy.dat"
        log_name = "relax_log.dat"
        dt = 0.001
        diff_coeff = 1.0
        refresh_vel = 1
        relax_lines = "max_backbone_force = 10\nmax_backbone_force_far = 10\n"
        traj_interval = max(output_freq, n_steps)
    else:
        energy_name = "energy.dat"
        log_name = "log.dat"
        dt = 0.003
        diff_coeff = 2.5
        refresh_vel = 0
        relax_lines = ""
        traj_interval = output_freq

    content = f"""backend = CPU
{precision_line}interaction_type = RNA2
sim_type = MD

topology = {os.path.basename(top_path)}
conf_file = {os.path.basename(conf_path)}
trajectory_file = {os.path.basename(traj_path)}
lastconf_file = {lastconf}
energy_file = {energy_name}
log_file = {log_name}

steps = {n_steps}
dt = {dt}
T = {temperature}K
salt_concentration = 0.15

newtonian_steps = 103
diff_coeff = {diff_coeff}
refresh_vel = {refresh_vel}
restart_step_counter = 1
thermostat = john
time_scale = linear
external_forces = 0
no_stdout_energy = 0
{relax_lines}
print_conf_interval = {traj_interval}
print_energy_every = {traj_interval * 10}

{seq_lines}
verlet_skin = 0.5
max_density_multiplier = 10
"""
    with open(input_path, "w") as f:
        f.write(content)


def _wsl_oat_shell_command(
    oat_bin: str,
    wsl_work: str,
    wsl_pdb: str,
    wsl_out: str,
) -> str:
    """Run ``oat PDB_oxDNA`` inside WSL (conda init for miniconda shebang)."""
    return (
        f"{_WSL_CONDA_INIT}"
        f"cd '{wsl_work}' && "
        f"'{oat_bin}' PDB_oxDNA '{wsl_pdb}' -o '{wsl_out}'"
    )


def _locate_oxdna_outputs(work: str, out_base: str) -> tuple[str, str] | None:
    """Return (topology, configuration) paths produced by oat / oxDNA utils."""
    stem = os.path.basename(out_base)
    pairs = [
        ("topology.top", "generated_topology.conf"),
        ("topology.top", "generated.conf"),
        (f"{stem}.top", f"{stem}.conf"),
        (f"{stem}.top", f"{stem}.dat"),
        ("generated.top", "generated.conf"),
        ("generated.top", "generated.dat"),
        (f"{stem}.top", f"{stem}.oxdna"),
    ]
    for top_name, conf_name in pairs:
        top = os.path.join(work, top_name)
        conf = os.path.join(work, conf_name)
        if os.path.isfile(top) and os.path.isfile(conf):
            return top, conf

    tops = sorted(f for f in os.listdir(work) if f.endswith(".top"))
    for top_name in tops:
        stem_top = top_name[:-4]
        for conf_name in (
            f"{stem_top}.conf",
            f"{stem_top}.dat",
            f"{stem_top}.oxdna",
            "generated.conf",
            "generated.dat",
        ):
            conf = os.path.join(work, conf_name)
            if os.path.isfile(conf):
                return os.path.join(work, top_name), conf
    return None


def _pdb_to_oxrna(pdb_path: str, top_path: str, conf_path: str) -> None:
    """Converts a PDB to oxDNA topology/configuration format."""
    work = os.path.dirname(top_path) or "."
    errors: list[str] = []
    out_base = os.path.join(work, "generated")

    # 1) oxDNA source build with Python bindings (host Python; skip on Windows+WSL)
    if not (oxdna_uses_wsl() and os.name == "nt"):
        try:
            result = subprocess.run(
                [sys.executable, "-m", "oxDNA.utils.pdb_to_oxdna", pdb_path, "RNA"],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=work,
            )
            if result.returncode == 0:
                if _finalize_oxdna_outputs(work, out_base, top_path, conf_path):
                    return
                errors.append("oxDNA.utils.pdb_to_oxdna: output files not found")
            else:
                errors.append(f"oxDNA.utils.pdb_to_oxdna: {(result.stderr or result.stdout or '')[:200]}")
        except FileNotFoundError:
            errors.append("oxDNA.utils.pdb_to_oxdna: module not installed")

    # 2) WSL miniconda Python + oxDNA.utils or PDB_oxDNA (when oxDNA runs in WSL)
    oat_cmd = resolve_oat_command()
    if oxdna_uses_wsl() and os.name == "nt" and oat_cmd and oat_cmd[0] == "wsl":
        oat_bin = oat_cmd[1]
        py = wsl_python_for_oat(oat_bin)
        wsl_pdb = host_path_for_subprocess(pdb_path)
        try:
            result = wsl_run(
                [py, "-m", "oxDNA.utils.pdb_to_oxdna", wsl_pdb, "RNA"],
                cwd=work,
                timeout=120,
            )
            if result.returncode == 0:
                if _finalize_oxdna_outputs(work, out_base, top_path, conf_path):
                    return
                errors.append("WSL oxDNA.utils.pdb_to_oxdna: output files not found")
            else:
                detail = (result.stderr or result.stdout or "").strip()
                if detail and "No module named 'oxDNA'" not in detail:
                    errors.append(f"WSL oxDNA.utils.pdb_to_oxdna: {detail[:300]}")
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
            errors.append(f"WSL oxDNA.utils.pdb_to_oxdna: {exc}")

    # 3) oxDNA-analysis-tools PDB_oxDNA API (script file — avoids shell quoting)
    if oat_cmd:
        if oat_cmd[0] == "wsl":
            oat_bin = oat_cmd[1]
            if not wsl_oat_pdb_module_ok(oat_bin):
                errors.append(WSL_OAT_REINSTALL_HINT)
            else:
                result = run_wsl_pdb_oxdna_script(
                    oat_bin=oat_bin,
                    work_dir=work,
                    pdb_path=pdb_path,
                    out_base=out_base,
                )
                if result.returncode == 0:
                    if _finalize_oxdna_outputs(work, out_base, top_path, conf_path):
                        return
                    errors.append(
                        "WSL PDB_oxDNA script: succeeded but topology/conf not found in "
                        f"{work} (files: {os.listdir(work)[:8]})"
                    )
                else:
                    detail = (result.stderr or result.stdout or "").strip()
                    errors.append(f"WSL PDB_oxDNA script: {detail[:500]}")
        else:
            result = run_native_pdb_oxdna_script(
                work_dir=work,
                pdb_path=pdb_path,
                out_base=out_base,
            )
            if result.returncode == 0:
                if _finalize_oxdna_outputs(work, out_base, top_path, conf_path):
                    return
                errors.append(
                    "PDB_oxDNA script: succeeded but topology/conf not found in "
                    f"{work} (files: {os.listdir(work)[:8]})"
                )
            else:
                detail = (result.stderr or result.stdout or "").strip()
                errors.append(f"PDB_oxDNA script: {detail[:500]}")

        # 4) oxDNA-analysis-tools CLI (last resort — often same backend as script)
        oat_args = ["PDB_oxDNA", pdb_path, "-o", out_base]
        if oat_cmd[0] == "wsl":
            wsl_work = host_path_for_subprocess(work)
            wsl_pdb = host_path_for_subprocess(pdb_path)
            wsl_out = host_path_for_subprocess(out_base)
            oat_bin = oat_cmd[1] if len(oat_cmd) > 1 else "oat"
            inner = _wsl_oat_shell_command(oat_bin, wsl_work, wsl_pdb, wsl_out)
            result = wsl_run_shell(inner, timeout=120)
        else:
            result = subprocess.run(
                [*oat_cmd, *oat_args],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=work,
            )
        if result.returncode == 0:
            if _finalize_oxdna_outputs(work, out_base, top_path, conf_path):
                return
            errors.append(
                "oat PDB_oxDNA CLI: succeeded but topology/conf not found in "
                f"{work} (files: {os.listdir(work)[:8]})"
            )
        else:
            detail = (result.stderr or result.stdout or "").strip()
            if "No module named 'oxDNA_analysis_tools.PDB_oxDNA'" in detail:
                errors.append(WSL_OAT_REINSTALL_HINT)
            elif detail:
                errors.append(f"oat PDB_oxDNA CLI: {detail[:500]}")

    raise RuntimeError(
        "Could not convert PDB to oxRNA format. Install oxDNA-analysis-tools "
        f"(pip install oxDNA-analysis-tools) or build oxDNA with Python support.\n"
        + "\n".join(errors)
    )


def _finalize_oxdna_outputs(
    work: str,
    out_base: str,
    top_path: str,
    conf_path: str,
) -> bool:
    located = _locate_oxdna_outputs(work, out_base)
    if not located:
        return False
    src_top, src_conf = located
    for src, dst in ((src_top, top_path), (src_conf, conf_path)):
        if os.path.abspath(src) == os.path.abspath(dst):
            continue
        if os.path.exists(dst):
            os.remove(dst)
        os.rename(src, dst)
    return True


def _rename_oxdna_outputs(work: str, top_path: str, conf_path: str) -> None:
    _finalize_oxdna_outputs(work, os.path.join(work, "generated"), top_path, conf_path)


def _extract_snapshots_oxrna(
    traj_path: str,
    rna_chains: list,
    interface_residues: list[tuple[str, int]],
    work_dir: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Parses oxRNA trajectory into coordinate and geometry snapshots."""
    frames = _parse_oxrna_trajectory(traj_path)
    iface_indices = rna_residue_to_trajectory_indices(rna_chains, interface_residues)

    coord_list = []
    geom_list  = []

    for frame_coords in frames:
        if frame_coords.shape[0] <= max(iface_indices, default=0):
            continue
        interface_frame = frame_coords[iface_indices]
        coord_list.append(interface_frame)
        geom_vec = _approximate_geometry_from_centers(frame_coords, iface_indices)
        geom_list.append(geom_vec)

    return np.array(coord_list, dtype=float), np.array(geom_list, dtype=float)


def _parse_oxrna_trajectory(traj_path: str) -> list[np.ndarray]:
    """Parses oxDNA/oxRNA trajectory.dat format."""
    frames = []
    current_frame = []
    in_frame = False

    with open(traj_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("t ="):
                if current_frame:
                    frames.append(np.array(current_frame, dtype=float))
                current_frame = []
                in_frame = True
                continue
            if line.startswith("b =") or line.startswith("E ="):
                continue
            if in_frame and line:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        cm = [float(parts[0]), float(parts[1]), float(parts[2])]
                        current_frame.append(cm)
                    except ValueError:
                        pass

    if current_frame:
        frames.append(np.array(current_frame, dtype=float))

    return frames


def _approximate_geometry_from_centers(
    all_coords: np.ndarray,
    iface_indices: list[int],
) -> np.ndarray:
    """Proxy geometry from coarse-grained nucleotide centers."""
    import math
    n_nuc = all_coords.shape[0]
    features = []

    for idx in iface_indices:
        prev_idx = max(0, idx - 1)
        next_idx = min(n_nuc - 1, idx + 1)
        p, c, n = all_coords[prev_idx], all_coords[idx], all_coords[next_idx]
        v1, v2 = c - p, n - c
        d1, d2 = np.linalg.norm(v1), np.linalg.norm(v2)
        angle = 0.0
        if d1 > 1e-6 and d2 > 1e-6:
            angle = math.degrees(math.acos(
                np.clip(np.dot(v1, v2) / (d1 * d2), -1, 1)
            ))
        features.append(np.array([
            d1, d2, angle, c[0], c[1], c[2], v1[0], v1[1], v1[2]
        ]))

    return np.concatenate(features)


def smoke_test_simulation(
    *,
    n_steps: int = 5_000,
    output_freq: int = 500,
) -> tuple[bool, str]:
    """
    Run a short oxRNA MD on a tiny folded strand (same path as the pipeline).

    Returns ``(ok, message)`` for ``verify_oxrna --test-simulation``.
    """
    if not oxrna_ready():
        return False, "oxRNA not ready (oxDNA binary or PDB converter missing)"

    sequence = "AUGCAUGCAUGCAUGCA"
    with tempfile.TemporaryDirectory(prefix="physrna_simtest_") as work_dir:
        start_pdb = os.path.join(work_dir, "free_rna_start.pdb")
        fold_rna_sequence(sequence, start_pdb, method="extended")
        try:
            traj_path = _run_oxrna(
                start_pdb,
                work_dir,
                n_steps=n_steps,
                output_freq=output_freq,
                temperature=300.0,
            )
        except RuntimeError as exc:
            log_hint = os.path.join(work_dir, "oxdna_run.log")
            if os.path.isfile(log_hint):
                return False, f"{exc}\n(full log: {log_hint})"
            return False, str(exc)

        if not os.path.isfile(traj_path) or os.path.getsize(traj_path) == 0:
            return False, f"oxRNA finished but trajectory is empty: {traj_path}"

        frames = _parse_oxrna_trajectory(traj_path)
        return True, (
            f"OK — {len(frames)} frames in {traj_path} "
            f"({n_steps} steps, work_dir={work_dir})"
        )

