from __future__ import annotations

import numpy as np


def cosine_distance_matrix(query_features: np.ndarray, gallery_features: np.ndarray) -> np.ndarray:
    """Compute cosine distance matrix where lower values are better."""
    query_norm = query_features / np.linalg.norm(query_features, axis=1, keepdims=True).clip(min=1e-12)
    gallery_norm = gallery_features / np.linalg.norm(gallery_features, axis=1, keepdims=True).clip(min=1e-12)
    similarity = np.matmul(query_norm, gallery_norm.T)
    return 1.0 - similarity


def cmc_map_from_distance(
    distance: np.ndarray,
    query_pids: np.ndarray,
    gallery_pids: np.ndarray,
    topk: int = 10,
) -> dict[str, float]:
    """Compute CMC@k and mAP for single-shot person retrieval."""
    num_queries = distance.shape[0]
    topk = min(topk, distance.shape[1])

    cmc_hits = np.zeros(topk, dtype=np.float64)
    average_precisions = []

    for i in range(num_queries):
        order = np.argsort(distance[i])
        matches = (gallery_pids[order] == query_pids[i]).astype(np.int32)

        if matches.sum() == 0:
            continue

        first_hit_idx = np.argmax(matches)
        cmc_hits[first_hit_idx:] += 1

        precision_at_k = np.cumsum(matches) / (np.arange(len(matches)) + 1)
        ap = np.sum(precision_at_k * matches) / matches.sum()
        average_precisions.append(ap)

    valid_queries = max(len(average_precisions), 1)
    cmc = cmc_hits / valid_queries
    mAP = float(np.mean(average_precisions)) if average_precisions else 0.0

    metrics = {f"rank_{k+1}": float(cmc[k]) for k in range(topk)}
    metrics["mAP"] = mAP
    return metrics
