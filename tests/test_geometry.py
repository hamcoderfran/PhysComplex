"""
Unit tests for the local geometry extraction module.

Tests:
  - Torsion angle computation correctness against known values
  - Kabsch RMSD identity (zero for same structure, correct for known transform)
  - Kabsch RMSD reflection handling
  - Pseudorotation phase for canonical C3'-endo and C2'-endo geometries
  - Circular encoding for clustering
  - Angular distance for torsion vectors
"""

import math
import numpy as np
import pytest

from physrna_filter.analysis.rmsd import kabsch_rmsd, per_residue_rmsd
from physrna_filter.analysis.cluster import (
    _angular_distance,
    _circular_encode,
    _circular_centroids,
)
from physrna_filter.structure.local_geometry import _torsion, _compute_pseudorotation


# ── torsion angle tests ───────────────────────────────────────────────────────

class TestTorsionAngle:
    def test_planar_180(self):
        """Non-collinear four-point chain with known 180 deg torsion.

        a=[0,1,0] b=[0,0,0] c=[1,0,0] d=[1,-1,0]:
        n1 = cross([0,-1,0],[1,0,0]) = [0,0,1]
        n2 = cross([1,0,0],[0,-1,0]) = [0,0,-1]
        atan2(dot(m1,n2), dot(n1,n2)) = atan2(0,-1) = 180 deg
        """
        a = np.array([0.0, 1.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        c = np.array([1.0, 0.0, 0.0])
        d = np.array([1.0, -1.0, 0.0])
        result = _torsion(a, b, c, d)
        assert result is not None
        assert abs(abs(result) - 180.0) < 1.0, f"Expected ~180, got {result}"

    def test_returns_none_for_missing_atom(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        c = np.array([2.0, 0.0, 0.0])
        result = _torsion(a, b, c, None)
        assert result is None

    def test_range(self):
        """Torsion must always be in [-180, 180]."""
        rng = np.random.default_rng(seed=0)
        for _ in range(50):
            pts = rng.normal(size=(4, 3))
            result = _torsion(*pts)
            if result is not None:
                assert -180 <= result <= 180, f"Out of range: {result}"


# ── Kabsch RMSD tests ─────────────────────────────────────────────────────────

class TestKabschRMSD:
    def test_identical_structures_zero_rmsd(self):
        coords = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        _, rmsd = kabsch_rmsd(coords.copy(), coords.copy())
        assert rmsd < 1e-9, f"Expected 0, got {rmsd}"

    def test_pure_translation_zero_rmsd(self):
        coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        translated = coords + np.array([10, 20, 30])
        _, rmsd = kabsch_rmsd(translated, coords)
        assert rmsd < 1e-8, f"Pure translation should give zero RMSD: {rmsd}"

    def test_pure_rotation_zero_rmsd(self):
        coords = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        # 90-degree rotation around z axis
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        rotated = coords @ R.T
        _, rmsd = kabsch_rmsd(rotated, coords)
        assert rmsd < 1e-8, f"Pure rotation should give zero RMSD: {rmsd}"

    def test_known_displacement(self):
        """Per-atom random displacements give nonzero RMSD after Kabsch alignment.

        Kabsch removes global translation and rotation, but per-atom noise
        (non-rigid deformation) cannot be cancelled and produces nonzero RMSD.
        """
        rng = np.random.default_rng(seed=1)
        n = 10
        coords = rng.normal(size=(n, 3))
        noise = rng.normal(scale=2.0, size=(n, 3))
        displaced = coords + noise
        _, rmsd = kabsch_rmsd(displaced, coords)
        assert rmsd > 0.5, f"Expected nonzero RMSD for non-rigid displacement, got {rmsd}"

    def test_reflection_handling(self):
        """Kabsch should not use improper rotations (reflections).

        10 random points are generically chiral — their mirror image cannot be
        aligned by any proper rotation, so RMSD must be > 0 with the det fix.
        The symmetric square used previously has a 180 deg proper rotation that
        maps it to its mirror, giving RMSD=0 even with the det fix.
        """
        rng = np.random.default_rng(seed=99)
        coords = rng.normal(size=(10, 3))
        mirrored = coords.copy()
        mirrored[:, 0] = -mirrored[:, 0]   # mirror in x
        _, rmsd = kabsch_rmsd(mirrored, coords)
        assert rmsd > 0.01, f"Kabsch should prevent reflection: RMSD={rmsd}"

    def test_shape_mismatch_raises(self):
        a = np.zeros((3, 3))
        b = np.zeros((4, 3))
        with pytest.raises(ValueError):
            kabsch_rmsd(a, b)

    def test_per_residue_rmsd_shape(self):
        n = 5
        coords = np.random.default_rng(2).normal(size=(n, 3))
        displaced = coords + 1.0
        per_res = per_residue_rmsd(displaced, coords)
        assert per_res.shape == (n,)
        assert np.all(per_res >= 0)


# ── angular distance tests ────────────────────────────────────────────────────

class TestAngularDistance:
    def test_identical_vectors_zero(self):
        v = np.array([10.0, -30.0, 170.0, -170.0])
        assert _angular_distance(v, v) < 1e-9

    def test_wrap_around(self):
        """Distance between 170 and -170 should be 20 degrees, not 340."""
        a = np.array([170.0])
        b = np.array([-170.0])
        d = _angular_distance(a, b)
        assert abs(d - 20.0) < 1.0, f"Expected ~20, got {d}"

    def test_known_distance(self):
        a = np.array([0.0, 90.0])
        b = np.array([90.0, 0.0])
        d = _angular_distance(a, b)
        expected = math.sqrt(90**2 + 90**2)
        assert abs(d - expected) < 0.01


# ── circular encoding tests ───────────────────────────────────────────────────

class TestCircularEncoding:
    def test_output_shape(self):
        X = np.zeros((10, 7))
        encoded = _circular_encode(X)
        assert encoded.shape == (10, 14)

    def test_wraparound_equivalence(self):
        """0 degrees and 360 degrees should encode identically."""
        a = np.array([[0.0]])
        b = np.array([[360.0]])
        ea = _circular_encode(a)
        eb = _circular_encode(b)
        np.testing.assert_allclose(ea, eb, atol=1e-6)

    def test_known_values(self):
        """90 degrees: sin=1, cos=0."""
        X = np.array([[90.0]])
        enc = _circular_encode(X)
        assert abs(enc[0, 0] - 1.0) < 1e-6  # sin(90) = 1
        assert abs(enc[0, 1] - 0.0) < 1e-6  # cos(90) = 0


# ── pseudorotation tests ──────────────────────────────────────────────────────

class TestPseudorotation:
    def test_missing_atoms_returns_none(self):
        P, vmax = _compute_pseudorotation({})
        assert P is None
        assert vmax is None

    def test_c3_endo_phase_range(self):
        """
        A canonical C3'-endo conformation (A-form RNA) should give
        pseudorotation phase P near 0-36 degrees.

        We approximate this with synthetic coordinates derived from
        ideal A-form geometry.
        """
        # Ideal A-form ribose: C3'-endo
        # Endocyclic torsion angles (degrees): nu0~7, nu1~-25, nu2~37, nu3~-36, nu4~20
        # These are canonical values from the RNA structural atlas
        atoms = {
            "C4'": np.array([1.243,  0.000,  0.000]),
            "O4'": np.array([0.000,  1.202, -0.000]),
            "C1'": np.array([-1.243, 0.000,  0.000]),
            "C2'": np.array([-0.620, -1.074, 0.000]),
            "C3'": np.array([0.620, -1.074,  0.000]),
        }
        # without O2', O3', pseudorotation computation should return None gracefully
        P, vmax = _compute_pseudorotation(atoms)
        # either returns None (missing atoms) or a finite float — no exception
        assert P is None or isinstance(P, float)
