"""
Internal coarse-grained RNA molecular dynamics (C4' bead chain).

Used when oxRNA is not installed.  Unlike the old mock trajectory, this does
NOT use AF3 bound coordinates — it folds from sequence (extended A-form) and
runs Langevin dynamics to produce an independent free-RNA ensemble.

Literature basis: simplified oxRNA/SimRNA-style coarse graining (Sulc et al. 2014;
Boniecki et al. 2016).  Not a replacement for oxRNA but eliminates the
circular entropic scoring problem.
"""
from __future__ import annotations

import math

import numpy as np
from Bio.PDB import PDBParser

from ..structure.extract_interface import extract_interface_coords


def run_cg_langevin_md(
    start_pdb: str,
    interface_residues: list[tuple[str, int]],
    n_frames: int = 500,
    dt: float = 0.002,
    temperature: float = 300.0,
    friction: float = 1.0,
    seed: int = 42,
    rna_chains: list | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run coarse-grained Langevin MD on C4' beads from a sequence-folded start.

    Returns:
        coord_snapshots: (n_frames, n_interface, 3)
        geom_snapshots:  (n_frames, n_interface * 9)
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("rna", start_pdb)

    # Full-chain C4' coordinates in residue order
    all_residues: list[tuple[str, int]] = []
    all_coords: list[np.ndarray] = []

    for chain in structure[0]:
        for residue in chain:
            if "C4'" in residue:
                all_residues.append((chain.id, residue.id[1]))
                all_coords.append(residue["C4'"].coord.copy())

    if not all_coords:
        raise ValueError(f"No C4' atoms in {start_pdb}")

    coords = np.array(all_coords, dtype=float)
    n_beads = coords.shape[0]

    # Map interface residue IDs to bead indices
    res_to_idx = {rid: i for i, rid in enumerate(all_residues)}
    iface_indices = [
        res_to_idx[rid] for rid in interface_residues if rid in res_to_idx
    ]
    if not iface_indices:
        iface_indices = list(range(min(len(interface_residues), n_beads)))

    # Bond rest lengths from starting structure
    bonds = [(i, i + 1) for i in range(n_beads - 1)]
    rest_lengths = np.array([
        np.linalg.norm(coords[j] - coords[i]) for i, j in bonds
    ])

    rng = np.random.default_rng(seed)
    kT = 0.001987 * temperature   # kcal/mol
    kb = 50.0    # bond spring constant
    ka = 8.0     # angle bending constant
    mass = 300.0 # amu per nucleotide bead

    pos = coords.copy()
    vel = rng.normal(scale=0.01, size=pos.shape)

    coord_snaps = []
    geom_snaps = []
    sample_every = max(1, n_frames // 500)

    for step in range(n_frames * sample_every):
        forces = np.zeros_like(pos)

        # harmonic bonds
        for bi, (i, j) in enumerate(bonds):
            vec = pos[j] - pos[i]
            d = np.linalg.norm(vec) + 1e-8
            f_mag = kb * (d - rest_lengths[bi])
            f_vec = f_mag * vec / d
            forces[i] += f_vec
            forces[j] -= f_vec

        # soft angle bending (three consecutive beads)
        for i in range(n_beads - 2):
            v1 = pos[i + 1] - pos[i]
            v2 = pos[i + 2] - pos[i + 1]
            d1 = np.linalg.norm(v1) + 1e-8
            d2 = np.linalg.norm(v2) + 1e-8
            cos_a = np.clip(np.dot(v1, v2) / (d1 * d2), -1, 1)
            angle = math.acos(cos_a)
            target = math.radians(150.0)   # A-form-like
            if angle > 1e-6:
                coeff = ka * (angle - target) / angle
                forces[i]     -= coeff * v1 / d1
                forces[i + 1] += coeff * (v1 / d1 - v2 / d2)
                forces[i + 2] += coeff * v2 / d2

        # excluded volume (soft repulsion) — local window only for speed
        for i in range(n_beads):
            j_start = max(0, i - 8)
            j_end = min(n_beads, i + 9)
            for j in range(j_start, j_end):
                if j <= i + 1:
                    continue
                vec = pos[j] - pos[i]
                d = np.linalg.norm(vec) + 1e-8
                if d < 4.0:
                    rep = 2.0 * (4.0 - d) / d * vec
                    forces[i] -= rep
                    forces[j] += rep

        # Langevin thermostat
        noise = rng.normal(scale=math.sqrt(2 * friction * kT * dt / mass), size=pos.shape)
        acc = forces / mass
        vel = vel + acc * dt - friction * vel * dt + noise
        pos = pos + vel * dt

        if step % sample_every == 0 and len(coord_snaps) < n_frames:
            iface_coords = pos[iface_indices]
            coord_snaps.append(iface_coords.copy())
            geom_snaps.append(_geometry_from_coords(pos, iface_indices))

    return np.array(coord_snaps), np.array(geom_snaps)


def _geometry_from_coords(
    all_coords: np.ndarray,
    iface_indices: list[int],
) -> np.ndarray:
    """9-element proxy geometry per interface nucleotide."""
    features = []
    n = all_coords.shape[0]
    for idx in iface_indices:
        prev_idx = max(0, idx - 1)
        next_idx = min(n - 1, idx + 1)
        p, c, nxt = all_coords[prev_idx], all_coords[idx], all_coords[next_idx]
        v1, v2 = c - p, nxt - c
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


def extract_interface_from_folded_pdb(
    folded_pdb: str,
    interface_residues: list[tuple[str, int]],
) -> np.ndarray:
    """Get interface C4' coords from a sequence-folded PDB."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("rna", folded_pdb)
    chains = list(structure[0])
    coords, _ = extract_interface_coords(chains, interface_residues)
    return coords
