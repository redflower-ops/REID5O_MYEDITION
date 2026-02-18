from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reid_system.data.collate import simple_infer_collate
from src.reid_system.data.infer_dataset import ProtocolImageDataset
from src.reid_system.data.orbench_protocol import OrBenchTestProtocol
from src.reid_system.engine.evaluator import evaluate_retrieval
from src.reid_system.models.tiny_cnn import TinyCNN
from src.reid_system.utils.config import load_config


def main() -> None:
    cfg = load_config(ROOT / "configs" / "orbench.yaml")
    data_root = str((ROOT / cfg["data"]["root"]).resolve())
    protocol = OrBenchTestProtocol(data_root)

    gallery_samples = protocol.parse_gallery()
    query_protocol = cfg["eval"]["query_protocol"]
    query_samples = protocol.parse_query(query_protocol)

    gallery_loader = DataLoader(
        ProtocolImageDataset(data_root, gallery_samples),
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=simple_infer_collate,
    )
    query_loader = DataLoader(
        ProtocolImageDataset(data_root, query_samples),
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=simple_infer_collate,
    )

    model = TinyCNN(num_classes=cfg["model"]["num_classes"], embedding_dim=cfg["model"]["embedding_dim"])
    checkpoint = torch.load(ROOT / cfg["train"]["checkpoint_path"], map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate_retrieval(
        model=model,
        query_loader=query_loader,
        gallery_loader=gallery_loader,
        device=torch.device(cfg["project"]["device"]),
        topk=cfg["eval"]["topk"],
    )

    print(f"Evaluation protocol={query_protocol}")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
