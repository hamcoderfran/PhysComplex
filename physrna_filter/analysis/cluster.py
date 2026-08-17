"""
Clusters a conformational ensemble in two parallel branches:

  1. Coordinate branch  — k-medoids on C4' Cartesian coordinates (post-Kabsch)
  2. Geometry branch    — k-means on torsion angle feature vectors

Both branches return cluster representatives (medoids / centroids) against
which the AF3 bound pose is compared in the scoring step.

Silhouette scoring selects the optimal k for each branch independently.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances


# ── coordinate-branch clustering ──────────────────────────────────────────────

def cluster_coordinates(
    snapshots: np.ndarray,
    k_range: range = range(2, 20),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Clusters coordinate snapshots and returns medoid coordinates.

    Args:
        snapshots: (n_frames, n_atoms, 3)

    Returns:
        labels   — (n_frames,) cluster assignment per frame
        medoids  — (n_clusters, n_atoms, 3) medoid coordinate arrays
    """
    n_frames = snapshots.shape[0]
    flat = snapshots.reshape(n_frames, -1)

    k = _optimal_k(flat, k_range, label="coordinate")
    km = KMeans(n_clusters=k, random_state=42, n_init=15)
    labels = km.fit_predict(flat)

    medoids = _find_medoids(snapshots, flat, labels, k)
    return labels, medoids


def cluster_geometry(
    geom_matrix: np.ndarray,
    k_range: range = range(2, 15),
    *,
    low_signal: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Clusters torsion-angle geometry feature vectors.

    Torsion angles are circular, so angular distance is used rather than
    Euclidean distance. For k-means we use a circular-aware preprocessing
    step: each angle theta is encoded as (sin(theta), cos(theta)), doubling
    the feature dimension but making Euclidean distance meaningful on the
    circular space.

    Args:
        geom_matrix: (n_frames, n_features)

    Returns:
        labels    — (n_frames,) cluster assignment
        centroids — (n_clusters, n_features) — in original torsion-angle space
    """
    encoded = _circular_encode(geom_matrix)

    if low_signal or len(geom_matrix) < 4:
        k = min(2, max(1, len(geom_matrix) - 1))
        if k < 1:
            k = 1
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(encoded) if len(encoded) else np.zeros(0, dtype=int)
        centroids = _circular_centroids(geom_matrix, labels, k) if len(labels) else geom_matrix[:1]
        print(f"Cluster (geometry): k={k} (low-signal / CG fallback)")
        return labels, centroids

    k = _optimal_k(encoded, k_range, label="geometry")
    km = KMeans(n_clusters=k, random_state=42, n_init=15)
    labels = km.fit_predict(encoded)

    # centroids in original space: take circular mean of each cluster
    centroids = _circular_centroids(geom_matrix, labels, k)
    return labels, centroids


# ── geometry score against free-ensemble clusters ─────────────────────────────

def nearest_geometry_cluster_distance(
    af3_geom_vector: np.ndarray,
    cluster_centroids: np.ndarray,
) -> tuple[float, int]:
    """
    Computes the angular distance between the AF3 bound-pose geometry vector
    and each geometry cluster centroid.

    Angular distance for circular features:
        d = sqrt( sum_i min(|a_i - b_i|, 360 - |a_i - b_i|)^2 )

    Returns:
        min_distance     — scalar in degrees
        nearest_cluster  — index of nearest centroid
    """
    min_dist = np.inf
    nearest  = -1

    for i, centroid in enumerate(cluster_centroids):
        dist = _angular_distance(af3_geom_vector, centroid)
        if dist < min_dist:
            min_dist = dist
            nearest  = i

    return float(min_dist), nearest


# ── internal helpers ──────────────────────────────────────────────────────────

def _optimal_k(flat: np.ndarray, k_range: range, label: str) -> int:
    n_samples = len(flat)
    best_k     = k_range.start
    best_score = -np.inf

    for k in k_range:
        if k >= n_samples:
            break
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = km.fit_predict(flat)

        if len(set(cluster_labels)) < 2:
            continue

        score = silhouette_score(
            flat,
            cluster_labels,
            sample_size=min(2000, n_samples),
            random_state=42,
        )
        if score > best_score:
            best_score = score
            best_k     = k

    print(f"Cluster ({label}): optimal k={best_k}, silhouette={best_score:.3f}")
    return best_k


def _find_medoids(
    snapshots: np.ndarray,
    flat: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """
    Returns the medoid (most central actual snapshot) for each cluster.

    Medoid minimizes the sum of pairwise distances to all other members,
    making it a more robust representative than the k-means centroid which
    may not correspond to any real observed conformation.
    """
    medoids = []

    for k in range(n_clusters):
        members = np.where(labels == k)[0]
        if len(members) == 1:
            medoids.append(snapshots[members[0]])
            continue

        member_flat = flat[members]
        dists = pairwise_distances(member_flat, metric="euclidean")
        local_idx = dists.sum(axis=1).argmin()
        medoids.append(snapshots[members[local_idx]])

    return np.array(medoids)


def _circular_encode(angles: np.ndarray) -> np.ndarray:
    """
    Encodes torsion angles theta as [sin(theta), cos(theta)] pairs.
    Input shape: (n, d)  ->  output shape: (n, 2d)
    """
    rad = np.radians(angles)
    return np.concatenate([np.sin(rad), np.cos(rad)], axis=1)


def _circular_centroids(
    geom_matrix: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """
    Computes the circular mean for each torsion angle dimension within each
    cluster. The circular mean of angles is atan2(mean(sin), mean(cos)).
    """
    n_features = geom_matrix.shape[1]
    centroids  = np.zeros((n_clusters, n_features), dtype=float)

    for k in range(n_clusters):
        members = geom_matrix[labels == k]
        if len(members) == 0:
            continue
        rad = np.radians(members)
        sin_mean = np.sin(rad).mean(axis=0)
        cos_mean = np.cos(rad).mean(axis=0)
        centroids[k] = np.degrees(np.arctan2(sin_mean, cos_mean))

    return centroids


def _angular_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Angular (circular) distance between two torsion-angle vectors in degrees.
    Each component uses the shorter arc on the circle.
    """
    diff = np.abs(a - b)
    diff = np.minimum(diff, 360.0 - diff)
    return float(np.sqrt(np.sum(diff ** 2)))
