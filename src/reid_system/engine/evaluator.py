from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.reid_system.metrics.retrieval import cmc_map_from_distance, cosine_distance_matrix


def extract_features(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Extract embeddings + pids from a dataloader."""
    model.eval()
    all_features: list[np.ndarray] = []
    all_pids: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            pids = batch["pids"].cpu().numpy()
            outputs = model(images)
            feats = outputs["embeddings"].cpu().numpy()
            all_features.append(feats)
            all_pids.append(pids)

    return np.concatenate(all_features, axis=0), np.concatenate(all_pids, axis=0)


def evaluate_retrieval(
    model: torch.nn.Module,
    query_loader: DataLoader,
    gallery_loader: DataLoader,
    device: torch.device,
    topk: int = 10,
) -> dict[str, float]:
    query_features, query_pids = extract_features(model, query_loader, device)
    gallery_features, gallery_pids = extract_features(model, gallery_loader, device)
    distance = cosine_distance_matrix(query_features, gallery_features)
    return cmc_map_from_distance(distance, query_pids, gallery_pids, topk=topk)


def topk_retrieval(
    query_feature: np.ndarray,
    gallery_features: np.ndarray,
    gallery_items: Iterable[dict],
    topk: int = 5,
) -> list[dict]:
    distances = cosine_distance_matrix(query_feature[None, :], gallery_features)[0]
    order = np.argsort(distances)[:topk]
    items = list(gallery_items)
    return [
        {
            "rank": int(rank + 1),
            "distance": float(distances[idx]),
            "pid": int(items[idx]["pid"]),
            "file_path": items[idx]["file_path"],
            "modality": items[idx]["modality"],
        }
        for rank, idx in enumerate(order)
    ]
