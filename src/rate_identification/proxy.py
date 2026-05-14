from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def build_cluster_scale_proxy(embeddings: np.ndarray, cluster_count: int) -> np.ndarray:
    sample_count = embeddings.shape[0]
    if sample_count == 1:
        return np.asarray([1.0], dtype=np.float32)

    effective_clusters = max(2, min(cluster_count, sample_count))
    kmeans = KMeans(n_clusters=effective_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    center_scores = np.linalg.norm(kmeans.cluster_centers_, axis=1)
    order = np.argsort(center_scores)
    ranked_values = np.linspace(0.25, 1.0, num=effective_clusters, dtype=np.float32)

    label_to_scale = np.zeros(effective_clusters, dtype=np.float32)
    for rank, cluster_idx in enumerate(order):
        label_to_scale[cluster_idx] = ranked_values[rank]

    return label_to_scale[labels]
