"""
RMSD computation with Kabsch optimal alignment.

The Kabsch algorithm finds the rotation matrix R that minimizes the RMSD
between two sets of N points after centering both at the origin. This removes
rigid-body translation and rotation from the comparison so that the resulting
RMSD measures pure conformational difference.
"""

from __future__ import annotations

import numpy as np


def kabsch_rmsd(
    mobile: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Computes minimum RMSD after optimal rigid-body alignment.

    Args:
        mobile:    (N, 3) coordinate array to align
        reference: (N, 3) coordinate array to align onto

    Returns:
        aligned_mobile  — (N, 3) coordinates of mobile after alignment
        rmsd            — scalar RMSD in Angstroms
    """
    if mobile.shape != reference.shape:
        raise ValueError(
            f"Shape mismatch: mobile {mobile.shape} vs reference {reference.shape}"
        )

    ref_center = reference.mean(axis=0)
    mob_center = mobile.mean(axis=0)

    ref_c = reference - ref_center
    mob_c = mobile    - mob_center

    H = mob_c.T @ ref_c

    U, S, Vt = np.linalg.svd(H)

    # det check avoids improper rotation (reflection)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])

    R = Vt.T @ D @ U.T
    mob_aligned = mob_c @ R.T

    diff = ref_c - mob_aligned
    rmsd = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))

    return mob_aligned, rmsd


def compute_rmsd_to_ensemble(
    reference_coords: np.ndarray,
    ensemble_snapshots: np.ndarray,
) -> np.ndarray:
    """
    Computes Kabsch-aligned RMSD between a reference structure and every
    snapshot in a conformational ensemble.

    Args:
        reference_coords:   (N, 3)           — AF3 bound pose
        ensemble_snapshots: (n_frames, N, 3) — free simulation frames

    Returns:
        rmsds — (n_frames,) array of RMSD values
    """
    rmsds = np.empty(len(ensemble_snapshots), dtype=float)

    for i, snapshot in enumerate(ensemble_snapshots):
        _, rmsds[i] = kabsch_rmsd(mobile=snapshot, reference=reference_coords)

    return rmsds


def per_residue_rmsd(
    mobile: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """
    Computes per-atom RMSD after global Kabsch alignment.

    Useful for identifying which specific interface nucleotides deviate most
    from the free ensemble rather than just the global interface score.

    Returns:
        (N,) array — per-atom deviation in Angstroms
    """
    mob_aligned, _ = kabsch_rmsd(mobile, reference)
    ref_c = reference - reference.mean(axis=0)
    diff = ref_c - mob_aligned
    return np.sqrt(np.sum(diff ** 2, axis=1))
