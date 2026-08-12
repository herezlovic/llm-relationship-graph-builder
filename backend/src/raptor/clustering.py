"""GMM (+ optional UMAP) soft clustering used by RAPTOR."""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _reduce_dimensions(embeddings: np.ndarray, n_neighbors: int = 10, n_components: int = 10) -> np.ndarray:
    n_samples, n_features = embeddings.shape
    if n_samples < 3:
        return embeddings

    target_components = min(n_components, n_samples - 1, n_features)
    if target_components < 2:
        return embeddings

    try:
        import umap

        n_neighbors = max(2, min(n_neighbors, n_samples - 1))
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=target_components,
            metric="cosine",
            random_state=42,
        )
        return reducer.fit_transform(embeddings)
    except Exception as exc:
        logger.warning("UMAP unavailable (%s); falling back to PCA.", exc)
        from sklearn.decomposition import PCA

        pca = PCA(n_components=target_components, random_state=42)
        return pca.fit_transform(embeddings)


def _bic_select_components(reduced: np.ndarray, max_clusters: int) -> int:
    from sklearn.mixture import GaussianMixture

    n_samples = reduced.shape[0]
    upper = min(max_clusters, n_samples)
    if upper <= 1:
        return 1

    best_k = 1
    best_bic = np.inf
    for k in range(1, upper + 1):
        try:
            gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
            gmm.fit(reduced)
            bic = gmm.bic(reduced)
            if bic < best_bic:
                best_bic = bic
                best_k = k
        except Exception as exc:
            logger.debug("GMM BIC failed for k=%s: %s", k, exc)
            continue
    return best_k


def cluster_embeddings(
    embeddings: Sequence[Sequence[float]],
    max_clusters: int = 10,
    threshold: float = 0.1,
    n_neighbors: int = 10,
) -> List[List[int]]:
    """
    Soft-cluster embedding vectors with GMM.

    Returns a list of clusters; each cluster is a list of member indices.
    Nodes may appear in multiple clusters when membership probability >= threshold.
    """
    if not embeddings:
        return []

    matrix = np.asarray(embeddings, dtype=np.float64)
    n_samples = matrix.shape[0]
    if n_samples == 1:
        return [[0]]
    if n_samples == 2:
        return [[0, 1]]

    reduced = _reduce_dimensions(matrix, n_neighbors=n_neighbors)
    k = _bic_select_components(reduced, max_clusters=max_clusters)

    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
    gmm.fit(reduced)
    probs = gmm.predict_proba(reduced)

    clusters: List[List[int]] = [[] for _ in range(k)]
    for idx in range(n_samples):
        assigned = False
        for cluster_idx, prob in enumerate(probs[idx]):
            if prob >= threshold:
                clusters[cluster_idx].append(idx)
                assigned = True
        if not assigned:
            clusters[int(np.argmax(probs[idx]))].append(idx)

    # Drop empty clusters and enforce uniqueness within a cluster while preserving soft multi-membership across clusters.
    cleaned: List[List[int]] = []
    for members in clusters:
        unique_members = sorted(set(members))
        if unique_members:
            cleaned.append(unique_members)
    return cleaned


def recursive_cluster_indices(
    embeddings: Sequence[Sequence[float]],
    token_counts: Sequence[int],
    max_tokens: int,
    max_clusters: int = 10,
    _depth: int = 0,
    _max_depth: int = 8,
) -> List[List[int]]:
    """
    Cluster embeddings, recursively splitting clusters whose combined tokens exceed max_tokens.
    """
    if not embeddings:
        return []
    if len(embeddings) == 1 or _depth >= _max_depth:
        return [list(range(len(embeddings)))]

    clusters = cluster_embeddings(embeddings, max_clusters=max_clusters)
    final_clusters: List[List[int]] = []

    for members in clusters:
        total_tokens = sum(token_counts[i] for i in members)
        if len(members) <= 1 or total_tokens <= max_tokens:
            final_clusters.append(members)
            continue

        # If clustering did not shrink the member set, stop to avoid infinite recursion.
        if len(members) == len(embeddings) and _depth > 0:
            # Force-split by contiguous halves as a last resort.
            mid = max(1, len(members) // 2)
            final_clusters.append(members[:mid])
            final_clusters.append(members[mid:])
            continue

        sub_embeddings = [embeddings[i] for i in members]
        sub_tokens = [token_counts[i] for i in members]
        sub_clusters = recursive_cluster_indices(
            sub_embeddings,
            sub_tokens,
            max_tokens=max_tokens,
            max_clusters=max_clusters,
            _depth=_depth + 1,
            _max_depth=_max_depth,
        )
        for sub in sub_clusters:
            remapped = [members[i] for i in sub]
            final_clusters.append(remapped)

    return final_clusters
