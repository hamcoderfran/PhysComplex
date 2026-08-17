"""
Introduces point mutations into a protein structure computationally.

Used in the mutation sensitivity test: given a wild-type complex and a
ProNAB mutation entry, produce the mutant structure, then score both with
the filter and compare the score delta to experimental delta-delta-G.

Tries PyRosetta first (preferred — does local energy minimization).
Falls back to a simple coordinate-level swap if PyRosetta is unavailable.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from Bio.PDB import MMCIFIO, PDBIO, Select

from .parse_complex import _load_structure


# PDBFixer's addMissingAtoms() implicitly invokes OpenMM minimization.  Structures
# at 6D8P's 14,004 records can spend tens of minutes in that unbounded operation,
# so keep mutation generation on the coordinate-level path above this limit.
PDBFIXER_MAX_ATOM_RECORDS = 10_000
_PDB_CHAIN_ID_CHARS = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
)


def _plan_chain_id_remap(structure) -> dict[str, str]:
    """Map overlong mmCIF chain IDs onto single-character PDB identifiers."""
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for model in structure:
        for chain in model:
            chain_id = chain.id
            if len(chain_id) <= 1:
                mapping[chain_id] = chain_id
                used.add(chain_id)
                continue
            new_id = next(c for c in _PDB_CHAIN_ID_CHARS if c not in used)
            mapping[chain_id] = new_id
            used.add(new_id)
    return mapping


def _apply_chain_id_remap(structure, mapping: dict[str, str]) -> None:
    for model in structure:
        for chain in model:
            chain.id = mapping[chain.id]


def _save_structure(structure, output_path: str, *, source_path: str) -> str:
    """Write a mutated structure, upgrading to mmCIF when PDB chain IDs are invalid."""
    source = Path(source_path)
    use_mmcif = source.suffix.lower() in {".cif", ".mmcif"}
    if not use_mmcif:
        mapping = _try_plan_chain_id_remap(structure)
        if mapping is None:
            use_mmcif = True
        else:
            _apply_chain_id_remap(structure, mapping)
    if use_mmcif:
        target = str(Path(output_path).with_suffix(".cif"))
        io = MMCIFIO()
        io.set_structure(structure)
        io.save(target)
        return target
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_path)
    return output_path


def _try_plan_chain_id_remap(structure) -> dict[str, str] | None:
    try:
        return _plan_chain_id_remap(structure)
    except StopIteration:
        return None


def _count_pdb_atom_records(pdb_path: str) -> int:
    """Count coordinate records without constructing a full structure object."""
    with open(pdb_path, errors="replace") as handle:
        return sum(line.startswith(("ATOM  ", "HETATM")) for line in handle)


def _debug_log(hypothesis_id: str, location: str, message: str, **data) -> None:
    # region agent log
    with open("/opt/cursor/logs/debug.log", "a") as handle:
        handle.write(json.dumps({
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }) + "\n")
    # endregion


# Parent standard names for common MODRES entries (MODRES record in PDB).
_MODRES_PARENT: dict[str, str] = {
    "MLZ": "LYS",
    "M3L": "LYS",
    "LLY": "LYS",
    "MSE": "MET",
}


def _protein_residue_matches(residue, wt_aa: str, alt_aa: str | None = None) -> bool:
    resname = residue.resname.strip().upper()
    expected = _ONE_TO_THREE.get(wt_aa.upper())
    if resname == expected:
        return True
    parent = _MODRES_PARENT.get(resname)
    if parent == expected:
        return True
    if alt_aa:
        alternate = _ONE_TO_THREE.get(alt_aa.upper())
        if resname == alternate:
            return True
        if parent and parent == alternate:
            return True
    return False


def find_mutation_chain(
    pdb_path: str,
    position: int,
    wt_aa: str,
    alt_aa: str | None = None,
) -> str:
    """
    Finds the protein chain containing a residue at `position` whose identity
    matches `wt_aa` (one-letter code).

    Used when the source dataset doesn't provide a chain ID.

    Some ProNAB PDBs are already engineered to the measured mutant state
    (e.g. PDB SEQADV records mark Cys->Ser).  In those cases `alt_aa` lets the
    caller recover the chain and then reconstruct the WT structure separately.
    """
    structure = _load_structure(pdb_path)
    fallback_chain = None

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[1] != position:
                    continue
                if _protein_residue_matches(residue, wt_aa):
                    return chain.id
                if alt_aa and _protein_residue_matches(residue, wt_aa, alt_aa=alt_aa):
                    fallback_chain = fallback_chain or chain.id

    if fallback_chain is not None:
        return fallback_chain

    expected_label = wt_aa if alt_aa is None else f"{wt_aa} or {alt_aa}"
    raise ValueError(f"No chain in {pdb_path} has {expected_label}{position}")


_RNA_ONE_TO_THREE = {"A": "ADE", "U": "URA", "G": "GUA", "C": "CYT"}
_RNA_THREE_TO_ONE = {
    "ADE": "A", "URA": "U", "GUA": "G", "CYT": "C",
    "A": "A", "U": "U", "G": "G", "C": "C",
}


def _rna_residue_matches(residue, wt_base: str) -> bool:
    observed = _RNA_THREE_TO_ONE.get(residue.resname.strip().upper(), "")
    return observed == wt_base.upper()


def find_mutation_rna_chain(
    pdb_path: str,
    position: int,
    wt_base: str,
    alt_base: str | None = None,
) -> str:
    """Find the RNA chain carrying a nucleotide at ``position`` matching ``wt_base``."""
    structure = _load_structure(pdb_path)
    fallback_chain = None

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[1] != position:
                    continue
                if _rna_residue_matches(residue, wt_base):
                    return chain.id
                if alt_base and _RNA_THREE_TO_ONE.get(residue.resname.strip().upper(), "") == alt_base.upper():
                    fallback_chain = fallback_chain or chain.id

    if fallback_chain is not None:
        return fallback_chain

    expected_label = wt_base if alt_base is None else f"{wt_base} or {alt_base}"
    raise ValueError(f"No RNA chain in {pdb_path} has {expected_label}{position}")


def _rna_site_on_chain(
    pdb_path: str,
    chain_id: str,
    position: int,
    wt_base: str,
) -> bool:
    structure = _load_structure(pdb_path)
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for residue in chain:
                if residue.id[1] == position and _rna_residue_matches(residue, wt_base):
                    return True
    return False


def resolve_rna_mutation_chain(
    pdb_path: str,
    chain_id: str,
    position: int,
    wt_base: str,
    mut_base: str,
    *,
    protein_chain_ids: set[str],
    rna_chain_ids: set[str],
) -> str:
    """
    Resolve the deposited RNA chain for a Nabe-style mutation label.

    Nabe ``chain`` metadata often names the logical RNA strand (e.g. ``A``)
    rather than the PDB chain ID (e.g. ``C``/``D`` for 1T0K).  When the
    declared chain is protein-only or lacks the target residue, fall back to
    scanning all RNA chains (e.g. 4lck G87U on chain C when B ends at 79).
    """
    if (
        chain_id
        and chain_id in rna_chain_ids
        and chain_id not in protein_chain_ids
        and _rna_site_on_chain(pdb_path, chain_id, position, wt_base)
    ):
        return chain_id
    return find_mutation_rna_chain(pdb_path, position, wt_base, alt_base=mut_base)


def _mutate_rna_simple(
    pdb_path: str,
    chain_id: str,
    position: int,
    mutant_base: str,
    output_path: str,
) -> str:
    """Coordinate-level RNA point mutation by residue rename."""
    structure = _load_structure(pdb_path)
    target_resname = _RNA_ONE_TO_THREE.get(mutant_base.upper(), mutant_base.upper())

    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for residue in chain:
                if residue.id[1] != position:
                    continue
                residue.resname = target_resname

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    return _save_structure(structure, output_path, source_path=pdb_path)


def prepare_fixed_structure(
    pdb_path: str,
    chain_id: str,
    position: int,
    aa: str,
    output_path: str,
    minimize: bool = True,
) -> str:
    """
    Runs the wild-type structure through the same PDBFixer
    (removeHeterogens + addMissingAtoms + restrained minimization) pipeline
    used for mutants, with no residue substitution (target == current AA at
    `position`). This keeps WT and mutant structures on equal footing for
    the contact-score comparison.

    Falls back to copying `pdb_path` unchanged if PDBFixer is unavailable
    (the coordinate-level fallback would needlessly truncate the WT
    sidechain to a Cbeta stub).
    """
    atom_records = _count_pdb_atom_records(pdb_path)
    # region agent log
    _debug_log(
        "A", "mutate.py:prepare_fixed_structure", "WT preparation entered",
        atom_records=atom_records, threshold=PDBFIXER_MAX_ATOM_RECORDS,
    )
    # endregion
    if atom_records > PDBFIXER_MAX_ATOM_RECORDS:
        print(
            f"  {os.path.basename(pdb_path)} has {atom_records:,} atom records; "
            "skipping PDBFixer relaxation and preserving deposited WT coordinates"
        )
        # region agent log
        _debug_log(
            "A", "mutate.py:prepare_fixed_structure", "large structure copied",
            atom_records=atom_records, fallback="copy_wt",
        )
        # endregion
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        shutil.copy(pdb_path, output_path)
        return output_path
    try:
        return _mutate_pdbfixer(pdb_path, chain_id, position, aa, output_path, minimize=minimize)
    except ImportError:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        shutil.copy(pdb_path, output_path)
        return output_path


def introduce_mutation(
    pdb_path: str,
    chain_id: str,
    position: int,
    mutant_aa: str,
    output_path: str,
    minimize: bool = True,
) -> str:
    """
    Introduces a single point mutation into a protein chain.
    Returns path to the mutant PDB file.
    """
    atom_records = _count_pdb_atom_records(pdb_path)
    # region agent log
    _debug_log(
        "A", "mutate.py:introduce_mutation", "mutation preparation entered",
        atom_records=atom_records, threshold=PDBFIXER_MAX_ATOM_RECORDS,
    )
    # endregion
    if atom_records > PDBFIXER_MAX_ATOM_RECORDS:
        print(
            f"  {os.path.basename(pdb_path)} has {atom_records:,} atom records; "
            "using coordinate-level mutation to avoid unbounded PDBFixer relaxation"
        )
        # region agent log
        _debug_log(
            "A", "mutate.py:introduce_mutation", "large structure fallback selected",
            atom_records=atom_records, fallback="coordinate_mutation",
        )
        # endregion
        return _mutate_simple(pdb_path, chain_id, position, mutant_aa, output_path)
    try:
        return _mutate_pyrosetta(pdb_path, chain_id, position, mutant_aa, output_path)
    except ImportError:
        pass

    try:
        return _mutate_pdbfixer(pdb_path, chain_id, position, mutant_aa, output_path, minimize=minimize)
    except ImportError:
        print("PyRosetta/PDBFixer not available — using coordinate-level mutation (less accurate)")
        return _mutate_simple(pdb_path, chain_id, position, mutant_aa, output_path)


def _mutate_pyrosetta(
    pdb_path: str,
    chain_id: str,
    position: int,
    mutant_aa: str,
    output_path: str,
) -> str:
    """
    PyRosetta mutation with local sidechain minimization.
    Requires a PyRosetta license (free for academic use).
    """
    import pyrosetta  # type: ignore
    import pyrosetta.rosetta.protocols.simple_moves as simple_moves  # type: ignore
    import pyrosetta.rosetta.protocols.minimization_packing as minpack  # type: ignore

    pyrosetta.init(silent=True)

    pose = pyrosetta.pose_from_pdb(pdb_path)
    res_num = pose.pdb_info().pdb2pose(chain_id, position)

    if res_num == 0:
        raise ValueError(f"Residue {chain_id}:{position} not found in structure")

    mutator = simple_moves.MutateResidue()
    mutator.set_res_and_target(res_num, mutant_aa)
    mutator.apply(pose)

    # minimize sidechains only around the mutation site
    scorefxn = pyrosetta.get_fa_scorefxn()
    movemap = pyrosetta.MoveMap()
    movemap.set_bb(False)
    movemap.set_chi(True)

    min_mover = minpack.MinMover()
    min_mover.movemap(movemap)
    min_mover.score_function(scorefxn)
    min_mover.apply(pose)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pose.dump_pdb(output_path)
    return output_path


def _mutate_pdbfixer(
    pdb_path: str,
    chain_id: str,
    position: int,
    mutant_aa: str,
    output_path: str,
    minimize: bool = True,
) -> str:
    """
    PDBFixer-based mutation: replaces the residue with the correct target
    template and builds the missing sidechain heavy atoms from ideal
    geometry. Open-source, no license required.

    Unlike the coordinate-level fallback (which truncates every mutation to
    a Cbeta stub), this produces a distinct structure for each target amino
    acid — required for non-alanine mutations (K, E, I, L, ...).

    After building the mutant sidechain, runs a restrained local energy
    minimization around the mutation site (amber14) to relieve steric
    clashes introduced by rotamer placement. If minimization fails for any
    reason (e.g. unsupported residue templates), falls back to the
    unminimized PDBFixer output.
    """
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile

    # region agent log
    _debug_log(
        "B", "mutate.py:_mutate_pdbfixer", "PDBFixer preparation started",
        chain_id=chain_id, position=position, minimize=minimize,
    )
    # endregion
    fixer = PDBFixer(filename=pdb_path)
    fixer.removeHeterogens(keepWater=False)

    old_resname = None
    for chain in fixer.topology.chains():
        if chain.id != chain_id:
            continue
        for residue in chain.residues():
            if residue.id == str(position):
                old_resname = residue.name
                break
        if old_resname is not None:
            break

    if old_resname is None:
        raise ValueError(f"Residue {chain_id}:{position} not found in {pdb_path}")

    new_resname = _one_to_three(mutant_aa)

    if new_resname != old_resname:
        fixer.applyMutations([f"{old_resname}-{position}-{new_resname}"], chain_id)

    fixer.findMissingResidues()
    fixer.missingResidues = {}  # don't model missing loops, only sidechain atoms
    fixer.findMissingAtoms()
    # region agent log
    _debug_log(
        "B", "mutate.py:_mutate_pdbfixer", "PDBFixer adding missing atoms",
        chain_id=chain_id, position=position,
    )
    # endregion
    fixer.addMissingAtoms()
    # region agent log
    _debug_log(
        "B", "mutate.py:_mutate_pdbfixer", "PDBFixer missing atoms added",
        chain_id=chain_id, position=position,
    )
    # endregion

    if minimize:
        try:
            _minimize_mutation_site(fixer, chain_id, position)
        except Exception as e:
            print(f"  minimization skipped ({type(e).__name__}: {e})")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)

    return output_path


def _minimize_mutation_site(fixer, chain_id: str, position: int, radius_angstrom: float = 8.0) -> None:
    """
    Restrained local energy minimization (amber14) around the mutation site.

    Builds the OpenMM system from PROTEIN ATOMS ONLY (RNA residues are
    deleted via Modeller first). RNA terminal residues in deposited PDB
    files frequently don't match amber RNA.OL3 templates (missing/extra
    phosphate or O3'/O5' atoms depending on how the structure was
    deposited), which previously made createSystem fail for ~all entries.
    Since RNA coordinates aren't being changed anyway (only the mutated
    protein sidechain needs relaxing), excluding RNA from the system
    sidesteps this entirely.

    All protein atoms further than `radius_angstrom` from the mutated
    residue's CA are harmonically restrained to their current positions, so
    only the mutation site and its immediate surroundings relax. Mutates
    `fixer` in place (updates fixer.positions and adds hydrogens to
    fixer.topology).
    """
    import numpy as np
    import openmm
    from openmm import unit, Vec3, CustomExternalForce, LocalEnergyMinimizer, VerletIntegrator, Platform
    from openmm.app import ForceField, NoCutoff, HBonds, Modeller

    fixer.addMissingHydrogens(7.0)

    protein_resnames = set(_ONE_TO_THREE.values())
    keep_mask = [atom.residue.name in protein_resnames for atom in fixer.topology.atoms()]
    if not any(keep_mask):
        raise ValueError("no protein atoms found for minimization")

    modeller = Modeller(fixer.topology, fixer.positions)
    to_delete = [atom for atom, keep in zip(modeller.topology.atoms(), keep_mask) if not keep]
    modeller.delete(to_delete)

    forcefield = ForceField("amber14-all.xml")
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=NoCutoff, constraints=HBonds)

    positions_nm = np.array(modeller.positions.value_in_unit(unit.nanometer))

    site_idx = None
    for chain in modeller.topology.chains():
        if chain.id != chain_id:
            continue
        for residue in chain.residues():
            if residue.id == str(position):
                for atom in residue.atoms():
                    if atom.name == "CA":
                        site_idx = atom.index
    if site_idx is None:
        raise ValueError(f"mutation site CA not found for {chain_id}:{position}")

    site_pos = positions_nm[site_idx]
    radius_nm = radius_angstrom * 0.1

    restraint = CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restraint.addGlobalParameter("k", 1.0e5)  # kJ/mol/nm^2
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    for i, pos in enumerate(positions_nm):
        if np.linalg.norm(pos - site_pos) > radius_nm:
            restraint.addParticle(i, list(pos))

    system.addForce(restraint)

    integrator = VerletIntegrator(0.001)
    platform = Platform.getPlatformByName("CPU")
    context = openmm.Context(system, integrator, platform)
    context.setPositions(modeller.positions)

    LocalEnergyMinimizer.minimize(context, tolerance=10, maxIterations=500)

    min_positions_nm = np.array(context.getState(getPositions=True).getPositions().value_in_unit(unit.nanometer))

    # write minimized protein positions back into fixer.positions
    full_positions_nm = np.array(fixer.positions.value_in_unit(unit.nanometer))
    j = 0
    for i, keep in enumerate(keep_mask):
        if keep:
            full_positions_nm[i] = min_positions_nm[j]
            j += 1

    fixer.positions = unit.Quantity([Vec3(*xyz) for xyz in full_positions_nm], unit.nanometer)


def _mutate_simple(
    pdb_path: str,
    chain_id: str,
    position: int,
    mutant_aa: str,
    output_path: str,
) -> str:
    """
    Coordinate-level mutation: replaces residue name only, keeps backbone atoms.
    Sidechain is truncated to Alanine Cbeta position.

    Less accurate than PyRosetta but requires no external dependencies.
    Suitable for alanine-scanning benchmarks (most ProNAB entries are X->A).
    """
    structure = _load_structure(pdb_path)

    three_letter = _one_to_three(mutant_aa)

    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for residue in chain:
                if residue.id[1] != position:
                    continue
                # rename residue
                residue.resname = three_letter
                # remove sidechain atoms beyond Cbeta for simplicity
                _truncate_to_cbeta(residue)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    return _save_structure(structure, output_path, source_path=pdb_path)


def _truncate_to_cbeta(residue) -> None:
    backbone = {"N", "CA", "C", "O", "CB", "OXT"}
    to_remove = [a.id for a in residue if a.name not in backbone]
    for aid in to_remove:
        try:
            residue.detach_child(aid)
        except Exception:
            pass


_ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def _one_to_three(one: str) -> str:
    return _ONE_TO_THREE.get(one.upper(), "ALA")
