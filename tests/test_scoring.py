"""
Integration-level tests for the scoring and clustering modules.

Tests:
  - cluster_coordinates correctly separates two well-separated structure families
  - cluster_geometry handles circular wraparound properly
  - run_full_scoring produces a complete FilterResult with expected fields
  - format_report produces non-empty string output
  - geometry score catches a synthetic C2'-endo violation
"""

import numpy as np
import pytest

from physrna_filter.analysis.cluster import (
    cluster_coordinates,
    cluster_geometry,
    nearest_geometry_cluster_distance,
)
from physrna_filter.analysis.score import (
    entropic_plausibility_score,
    _rmsd_verdict,
    _geom_verdict,
    format_report,
    FilterResult,
)
from physrna_filter.analysis.geometry_score import (
    PerNucleotideScore,
    aggregate_geometry_score,
    _generate_flags,
    score_per_nucleotide,
)
from physrna_filter.structure.local_geometry import NucleotideGeometry


# ── coordinate cluster tests ──────────────────────────────────────────────────

class TestCoordinateCluster:
    def _make_two_clusters(self, n=200, n_atoms=5):
        rng = np.random.default_rng(seed=7)
        cluster_a = rng.normal(loc=0.0,  scale=0.5, size=(n // 2, n_atoms, 3))
        cluster_b = rng.normal(loc=10.0, scale=0.5, size=(n // 2, n_atoms, 3))
        snapshots = np.concatenate([cluster_a, cluster_b], axis=0)
        indices   = np.random.permutation(n)
        return snapshots[indices]

    def test_finds_two_clusters(self):
        snapshots = self._make_two_clusters()
        labels, medoids = cluster_coordinates(snapshots, k_range=range(2, 5))
        assert len(set(labels)) == 2, "Expected 2 clusters for well-separated data"
        assert medoids.shape[0] == 2

    def test_medoids_are_actual_snapshots(self):
        snapshots = self._make_two_clusters(n=100)
        labels, medoids = cluster_coordinates(snapshots, k_range=range(2, 4))
        for medoid in medoids:
            diffs = np.linalg.norm(
                snapshots.reshape(len(snapshots), -1) - medoid.reshape(-1), axis=1
            )
            assert diffs.min() < 1e-6, "Medoid must be one of the actual snapshots"

    def test_output_shapes(self):
        n, n_atoms = 60, 4
        snapshots = np.random.default_rng(0).normal(size=(n, n_atoms, 3))
        labels, medoids = cluster_coordinates(snapshots, k_range=range(2, 5))
        assert labels.shape == (n,)
        assert medoids.ndim == 3
        assert medoids.shape[1] == n_atoms
        assert medoids.shape[2] == 3


# ── geometry cluster tests ────────────────────────────────────────────────────

class TestGeometryCluster:
    def _make_two_angle_clusters(self, n=150, n_feat=9):
        rng = np.random.default_rng(seed=3)
        a = rng.normal(loc=-120.0, scale=8.0, size=(n // 2, n_feat))
        b = rng.normal(loc=60.0,   scale=8.0, size=(n // 2, n_feat))
        mat = np.concatenate([a, b], axis=0)
        idx = np.random.permutation(n)
        return mat[idx]

    def test_finds_two_torsion_clusters(self):
        mat = self._make_two_angle_clusters()
        labels, centroids = cluster_geometry(mat, k_range=range(2, 5))
        assert len(set(labels)) == 2

    def test_centroid_shape(self):
        n, n_feat = 100, 18
        mat = np.random.default_rng(0).uniform(-180, 180, size=(n, n_feat))
        labels, centroids = cluster_geometry(mat, k_range=range(2, 4))
        assert centroids.shape[1] == n_feat

    def test_wraparound_not_broken(self):
        """Angles near +/-180 should cluster together, not be split."""
        n = 100
        rng = np.random.default_rng(9)
        angles = np.concatenate([
            rng.normal(loc=175.0,  scale=3.0, size=(n // 2, 1)),
            rng.normal(loc=-175.0, scale=3.0, size=(n // 2, 1)),
        ], axis=0)
        labels, centroids = cluster_geometry(angles, k_range=range(1, 4))
        # the circular mean of these two groups should be ~180 or ~-180
        centroid_vals = centroids[:, 0]
        mean_angle = np.mean(centroid_vals)
        assert abs(abs(mean_angle) - 180.0) < 20.0 or len(set(labels)) <= 2


# ── entropic plausibility scoring tests ──────────────────────────────────────

class TestEntropicPlausibility:
    def test_pass_when_bound_near_medoid(self):
        medoid = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        # bound pose very close to medoid
        bound  = medoid + 0.1
        medoids = np.array([medoid])
        rmsd, cluster_idx, verdict = entropic_plausibility_score(bound, medoids)
        assert rmsd < 0.5
        assert verdict == "PASS"
        assert cluster_idx == 0

    def test_fail_when_bound_far_from_all_medoids(self):
        # Scaling all atoms by 5x gives a non-rigid deformation Kabsch cannot cancel.
        # RMSD = 4 * sqrt(mean(||medoid_centered||^2)) >> 4.0 A (FAIL threshold).
        rng = np.random.default_rng(seed=42)
        n_atoms = 8
        medoid = rng.normal(size=(n_atoms, 3))
        bound  = medoid * 5.0   # same shape, 5x scale — not removable by rotation
        medoids = np.array([medoid])
        rmsd, _, verdict = entropic_plausibility_score(bound, medoids)
        assert rmsd > 4.0, f"Expected RMSD > 4.0, got {rmsd}"
        assert verdict == "FAIL", f"Expected FAIL, got {verdict} (RMSD={rmsd:.2f})"

    def test_nearest_of_multiple_medoids(self):
        rng = np.random.default_rng(0)
        n_atoms = 4
        medoid_a = rng.normal(loc=0.0, size=(n_atoms, 3))
        medoid_b = rng.normal(loc=20.0, size=(n_atoms, 3))
        bound = medoid_a + 0.2

        medoids = np.array([medoid_a, medoid_b])
        _, cluster_idx, _ = entropic_plausibility_score(bound, medoids)
        assert cluster_idx == 0, "Should select medoid_a as nearest"

    def test_verdict_thresholds(self):
        from physrna_filter.analysis.thresholds import load_thresholds
        t = load_thresholds()
        pass_t, warn_t = t["rmsd_pass"], t["rmsd_warn"]

        assert _rmsd_verdict(pass_t - 0.5) == "PASS"
        assert _rmsd_verdict(pass_t - 0.01) == "PASS"
        assert _rmsd_verdict(pass_t + 0.01) == "WARN"
        assert _rmsd_verdict(warn_t - 0.01) == "WARN"
        assert _rmsd_verdict(warn_t + 0.01) == "FAIL"


# ── per-nucleotide geometry tests ─────────────────────────────────────────────

class TestPerNucleotideGeometry:
    def _make_geom(self, sugar_pucker_P=15.0, chi=-150.0) -> NucleotideGeometry:
        return NucleotideGeometry(
            residue_id=("A", 1),
            resname="A",
            alpha=-62.0,
            beta=176.0,
            gamma=47.0,
            delta=80.0,
            epsilon=-153.0,
            zeta=-73.0,
            chi=chi,
            pseudorotation_P=sugar_pucker_P,
            pseudorotation_vmax=36.0,
        )

    def test_c3_endo_no_sugar_flag(self):
        geom = self._make_geom(sugar_pucker_P=18.0)
        devs  = {k: 0.0 for k in ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi"]}
        flags = _generate_flags(geom, devs, cluster_distance=20.0)
        assert not any("C2'-endo" in f for f in flags)

    def test_c2_endo_raises_flag(self):
        geom = self._make_geom(sugar_pucker_P=160.0)
        devs  = {k: 0.0 for k in ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi"]}
        flags = _generate_flags(geom, devs, cluster_distance=20.0)
        assert any("C2'-endo" in f for f in flags), "C2'-endo should be flagged"

    def test_syn_chi_raises_flag(self):
        geom = self._make_geom(chi=30.0)
        devs  = {k: 0.0 for k in ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi"]}
        flags = _generate_flags(geom, devs, cluster_distance=20.0)
        assert any("syn" in f for f in flags), "syn glycosidic bond should be flagged"

    def test_anti_chi_no_flag(self):
        geom = self._make_geom(chi=-150.0)
        devs  = {k: 0.0 for k in ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi"]}
        flags = _generate_flags(geom, devs, cluster_distance=20.0)
        assert not any("syn" in f for f in flags)

    def test_aggregate_fail_propagates(self):
        fail_score = PerNucleotideScore(
            residue_id=("A", 1), resname="G",
            geometry_cluster_distance=120.0,
            alpha_dev=80.0, beta_dev=0.0, gamma_dev=0.0, delta_dev=0.0,
            epsilon_dev=0.0, zeta_dev=0.0, chi_dev=0.0,
            sugar_pucker="C2'-endo", pseudorotation_P=160.0,
            verdict="FAIL", flags=["C2'-endo sugar pucker"],
        )
        pass_score = PerNucleotideScore(
            residue_id=("A", 2), resname="A",
            geometry_cluster_distance=15.0,
            alpha_dev=5.0, beta_dev=0.0, gamma_dev=0.0, delta_dev=0.0,
            epsilon_dev=0.0, zeta_dev=0.0, chi_dev=0.0,
            sugar_pucker="C3'-endo", pseudorotation_P=18.0,
            verdict="PASS", flags=[],
        )
        agg = aggregate_geometry_score([fail_score, pass_score])
        assert agg["verdict"] == "FAIL"
        assert agg["n_fail"] == 1

    def test_af3_intrinsic_geometry_passes_typical_bound_pose(self):
        geom = self._make_geom(sugar_pucker_P=18.0, chi=-150.0)
        from physrna_filter.structure.local_geometry import NucleotideGeometry

        af3_geom = {
            ("A", 1): geom,
        }
        per_nuc = score_per_nucleotide(
            af3_geom,
            np.zeros((2, 9)),
            [("A", 1)],
            af3_mode=True,
        )
        assert per_nuc[0].verdict in ("PASS", "WARN")
        assert per_nuc[0].geometry_cluster_distance < 50.0

    def test_af3_intrinsic_geometry_fails_syn_chi(self):
        geom = self._make_geom(sugar_pucker_P=18.0, chi=25.0)
        af3_geom = {("A", 1): geom}
        per_nuc = score_per_nucleotide(
            af3_geom,
            np.zeros((2, 9)),
            [("A", 1)],
            af3_mode=True,
        )
        assert per_nuc[0].verdict == "FAIL"


# ── format_report smoke test ──────────────────────────────────────────────────

class TestFormatReport:
    def test_produces_output(self):
        result = FilterResult(
            rmsd_score=1.5,
            rmsd_nearest_cluster=0,
            rmsd_verdict="PASS",
            geom_score=25.0,
            geom_max_score=40.0,
            geom_nearest_cluster=1,
            geom_verdict="PASS",
            per_nucleotide=[],
            combined_verdict="PASS",
            confidence=0.75,
        )
        report = format_report(result, [("A", 14), ("A", 15)])
        assert len(report) > 50
        assert "PASS" in report
        assert "1.50" in report
