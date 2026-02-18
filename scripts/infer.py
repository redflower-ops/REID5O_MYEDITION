from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reid_system.data.collate import simple_infer_collate
from src.reid_system.data.infer_dataset import ProtocolImageDataset
from src.reid_system.data.orbench_protocol import OrBenchTestProtocol
from src.reid_system.engine.evaluator import extract_features, topk_retrieval
from src.reid_system.models.tiny_cnn import TinyCNN
from src.reid_system.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ORBench retrieval inference")
    parser.add_argument("--query-image", required=True, help="Path to query image")
    parser.add_argument("--topk", type=int, default=5, help="Top-k retrieval")
    parser.add_argument("--output", default="outputs/infer_results.json", help="JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(ROOT / "configs" / "orbench.yaml")
    data_root = str((ROOT / cfg["data"]["root"]).resolve())

    protocol = OrBenchTestProtocol(data_root)
    gallery_samples = protocol.parse_gallery()

    gallery_loader = DataLoader(
        ProtocolImageDataset(data_root, gallery_samples),
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=simple_infer_collate,
    )

    device = torch.device(cfg["project"]["device"])
    model = TinyCNN(num_classes=cfg["model"]["num_classes"], embedding_dim=cfg["model"]["embedding_dim"]).to(device)
    checkpoint = torch.load(ROOT / cfg["train"]["checkpoint_path"], map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    gallery_features, _ = extract_features(model, gallery_loader, device=device)

    transform = T.Compose([T.Resize((384, 128)), T.ToTensor()])
    query_image = Image.open(args.query_image).convert("RGB")
    query_tensor = transform(query_image).unsqueeze(0).to(device)

    with torch.no_grad():
        query_feature = model(query_tensor)["embeddings"].cpu().numpy()[0]

    results = topk_retrieval(query_feature, gallery_features, gallery_samples, topk=args.topk)

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved retrieval results to {output_path}")


if __name__ == "__main__":
    main()
