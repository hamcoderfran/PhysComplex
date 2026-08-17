"""
Extracts per-nucleotide local geometry features from RNA structures.

For each interface nucleotide this module computes:
  - Seven backbone torsion angles (alpha, beta, gamma, delta, epsilon, zeta, chi)
  - Sugar pseudorotation phase P and amplitude nu_max
  - Base stacking distance and angle to nearest neighbour

These features go into the geometry-branch clustering pipeline, which compares
the AF3 bound conformation's local geometry against what was observed in the
free-RNA simulation ensemble.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from Bio.PDB import PDBParser

from .parse_complex import RNA_RESIDUE_NAMES


PURINE_NAMES  = {"A", "G", "ADE", "GUA"}
PYRIMIDINE_NAMES = {"U", "C", "URA", "CYT"}


@dataclass
class NucleotideGeometry:
    """Local geometry for a single nucleotide."""
    residue_id: tuple[str, int]
    resname: str

    # backbone torsions in degrees, None if atoms missing
    alpha:   float | None = None   # O3'(i-1) - P - O5' - C5'
    beta:    float | None = None   # P - O5' - C5' - C4'
    gamma:   float | None = None   # O5' - C5' - C4' - C3'
    delta:   float | None = None   # C5' - C4' - C3' - O3'
    epsilon: float | None = None   # C4' - C3' - O3' - P(i+1)
    zeta:    float | None = None   # C3' - O3' - P(i+1) - O5'(i+1)
    chi:     float | None = None   # glycosidic torsion

    # sugar pucker
    pseudorotation_P:    float | None = None   # phase angle, degrees
    pseudorotation_vmax: float | None = None   # amplitude, degrees

    # stacking (relative to sequential neighbour)
    stack_distance: float | None = None   # C1'-C1' inter-residue distance, Angstroms
    stack_angle:    float | None = None   # angle between base normals, degrees

    def to_vector(self) -> np.ndarray:
        """
        Returns a fixed-length float vector of all torsion and pseudorotation
        features. Missing values are filled with 0.0.

        Used as the feature vector for geometry-branch clustering.
        Shape: (9,)  [alpha beta gamma delta epsilon zeta chi P vmax]
        """
        fields = [
            self.alpha, self.beta, self.gamma, self.delta,
            self.epsilon, self.zeta, self.chi,
            self.pseudorotation_P, self.pseudorotation_vmax,
        ]
        return np.array([f if f is not None else 0.0 for f in fields], dtype=float)

    def sugar_pucker_class(self) -> str:
        """
        Classifies sugar pucker from pseudorotation phase P.

        C3'-endo (P ~ 0-36 deg) is normal for A-form RNA.
        C2'-endo (P ~ 144-190 deg) is unusual and flags a suspicious conformation.
        """
        if self.pseudorotation_P is None:
            return "unknown"
        P = self.pseudorotation_P % 360
        if P < 36 or P > 324:
            return "C3'-endo"
        elif 144 <= P <= 190:
            return "C2'-endo"
        elif 72 <= P <= 108:
            return "C4'-exo"
        else:
            return "other"


def extract_geometry(
    rna_chains: list,
    interface_residues: list[tuple[str, int]],
) -> dict[tuple[str, int], NucleotideGeometry]:
    """
    Computes local geometry for every interface nucleotide.

    Needs the full chain list (not just interface residues) because torsion
    angles alpha, epsilon, and zeta span the bond to the adjacent nucleotide.

    Returns dict mapping residue_id -> NucleotideGeometry.
    """
    interface_set = set(interface_residues)
    results: dict[tuple[str, int], NucleotideGeometry] = {}

    for chain in rna_chains:
        residues = [
            r for r in chain.get_residues()
            if r.id[0] == " " and r.resname.strip().upper() in RNA_RESIDUE_NAMES
        ]
        for i, residue in enumerate(residues):
            rid = (chain.id, residue.id[1])
            if rid not in interface_set:
                continue

            prev_res = residues[i - 1] if i > 0 else None
            next_res = residues[i + 1] if i < len(residues) - 1 else None

            geom = NucleotideGeometry(
                residue_id=rid,
                resname=residue.resname.strip(),
            )

            atoms = _get_atoms(residue)
            prev_atoms = _get_atoms(prev_res) if prev_res else {}
            next_atoms = _get_atoms(next_res) if next_res else {}

            # backbone torsions
            geom.alpha   = _torsion(prev_atoms.get("O3'"), atoms.get("P"),
                                    atoms.get("O5'"), atoms.get("C5'"))
            geom.beta    = _torsion(atoms.get("P"),   atoms.get("O5'"),
                                    atoms.get("C5'"), atoms.get("C4'"))
            geom.gamma   = _torsion(atoms.get("O5'"), atoms.get("C5'"),
                                    atoms.get("C4'"), atoms.get("C3'"))
            geom.delta   = _torsion(atoms.get("C5'"), atoms.get("C4'"),
                                    atoms.get("C3'"), atoms.get("O3'"))
            geom.epsilon = _torsion(atoms.get("C4'"), atoms.get("C3'"),
                                    atoms.get("O3'"), next_atoms.get("P"))
            geom.zeta    = _torsion(atoms.get("C3'"), atoms.get("O3'"),
                                    next_atoms.get("P"), next_atoms.get("O5'"))
            geom.chi     = _compute_chi(residue, atoms)

            # sugar pucker
            P, vmax = _compute_pseudorotation(atoms)
            geom.pseudorotation_P    = P
            geom.pseudorotation_vmax = vmax

            # stacking to next residue
            if next_res and "C1'" in residue and "C1'" in next_res:
                c1_curr = residue["C1'"].coord
                c1_next = next_res["C1'"].coord
                geom.stack_distance = float(np.linalg.norm(c1_curr - c1_next))
                geom.stack_angle    = _base_stacking_angle(residue, next_res)

            results[rid] = geom

    return results


def geometry_to_matrix(
    geometry_map: dict[tuple[str, int], NucleotideGeometry],
    residue_order: list[tuple[str, int]],
) -> np.ndarray:
    """
    Assembles per-nucleotide geometry vectors into a flat feature matrix row.

    Shape: (1, len(residue_order) * 9) — suitable for clustering across frames.
    """
    vectors = []
    for rid in residue_order:
        if rid in geometry_map:
            vectors.append(geometry_map[rid].to_vector())
        else:
            vectors.append(np.zeros(9, dtype=float))
    return np.concatenate(vectors)


# ── internal helpers ──────────────────────────────────────────────────────────

def _get_atoms(residue) -> dict[str, np.ndarray]:
    if residue is None:
        return {}
    return {atom.name: atom.coord.copy() for atom in residue}


def _torsion(a, b, c, d) -> float | None:
    """
    Computes the dihedral angle defined by four coordinate points.
    Returns degrees in range [-180, 180], or None if any point is missing.
    """
    if any(x is None for x in (a, b, c, d)):
        return None

    a, b, c, d = (np.array(x, dtype=float) for x in (a, b, c, d))

    b1 = b - a
    b2 = c - b
    b3 = d - c

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    b2_norm = np.linalg.norm(b2)

    if n1_norm < 1e-8 or n2_norm < 1e-8 or b2_norm < 1e-8:
        return None

    n1 /= n1_norm
    n2 /= n2_norm
    b2 /= b2_norm

    m1 = np.cross(n1, b2)
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)

    angle = math.degrees(math.atan2(y, x))
    return angle


def _compute_chi(residue, atoms: dict) -> float | None:
    """
    Computes the glycosidic torsion chi.

    Purines:     O4' - C1' - N9 - C4
    Pyrimidines: O4' - C1' - N1 - C2
    """
    resname = residue.resname.strip().upper()

    if resname in PURINE_NAMES:
        return _torsion(
            atoms.get("O4'"), atoms.get("C1'"),
            atoms.get("N9"),  atoms.get("C4"),
        )
    elif resname in PYRIMIDINE_NAMES:
        return _torsion(
            atoms.get("O4'"), atoms.get("C1'"),
            atoms.get("N1"),  atoms.get("C2"),
        )
    return None


def _compute_pseudorotation(atoms: dict) -> tuple[float | None, float | None]:
    """
    Computes the pseudorotation phase P and amplitude nu_max from the five
    endocyclic torsion angles of the ribose ring.

    nu0: C4'-O4'-C1'-C2'
    nu1: O4'-C1'-C2'-C3'
    nu2: C1'-C2'-C3'-C4'   (reference torsion)
    nu3: C2'-C3'-C4'-O4'
    nu4: C3'-C4'-O4'-C1'
    """
    nu = [
        _torsion(atoms.get("C4'"), atoms.get("O4'"), atoms.get("C1'"), atoms.get("C2'")),
        _torsion(atoms.get("O4'"), atoms.get("C1'"), atoms.get("C2'"), atoms.get("C3'")),
        _torsion(atoms.get("C1'"), atoms.get("C2'"), atoms.get("C3'"), atoms.get("C4'")),
        _torsion(atoms.get("C2'"), atoms.get("C3'"), atoms.get("C4'"), atoms.get("O4'")),
        _torsion(atoms.get("C3'"), atoms.get("C4'"), atoms.get("O4'"), atoms.get("C1'")),
    ]

    if any(v is None for v in nu):
        return None, None

    nu = [math.radians(v) for v in nu]

    sin36 = math.sin(math.radians(36))
    sin72 = math.sin(math.radians(72))

    numerator   = (nu[4] + nu[1]) - (nu[3] + nu[0])
    denominator = 2.0 * nu[2] * (sin36 + sin72)

    if abs(denominator) < 1e-8:
        return None, None

    P = math.degrees(math.atan2(numerator, denominator))

    nu2 = nu[2]
    if abs(math.cos(math.radians(P))) < 1e-8:
        vmax = None
    else:
        vmax = math.degrees(nu2 / math.cos(math.radians(P)))

    return P, vmax


def _base_stacking_angle(res1, res2) -> float | None:
    """
    Estimates the angle between the planes of two adjacent bases.
    Uses three base atoms to define each plane.
    Small angle (<30°) means good stacking.
    """
    plane1 = _base_plane_normal(res1)
    plane2 = _base_plane_normal(res2)

    if plane1 is None or plane2 is None:
        return None

    cos_angle = np.clip(np.dot(plane1, plane2), -1.0, 1.0)
    angle = math.degrees(math.acos(abs(cos_angle)))
    return angle


def _base_plane_normal(residue) -> np.ndarray | None:
    """Computes a normal vector to the nucleobase plane using three ring atoms."""
    resname = residue.resname.strip().upper()

    if resname in PURINE_NAMES:
        atom_names = ["N9", "C4", "C8"]
    elif resname in PYRIMIDINE_NAMES:
        atom_names = ["N1", "C2", "C6"]
    else:
        return None

    coords = []
    for name in atom_names:
        if name in residue:
            coords.append(residue[name].coord.copy())

    if len(coords) < 3:
        return None

    v1 = coords[1] - coords[0]
    v2 = coords[2] - coords[0]
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)

    if norm < 1e-8:
        return None

    return normal / norm
