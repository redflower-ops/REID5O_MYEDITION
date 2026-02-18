from __future__ import annotations

from pathlib import Path
import sys

from torch.utils.data import DataLoader
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reid_system.data.collate import make_train_collate
from src.reid_system.data.orbench_dataset import OrBenchDataset
from src.reid_system.engine.trainer import train_classifier
from src.reid_system.models.tiny_cnn import TinyCNN
from src.reid_system.utils.config import load_config
from src.reid_system.utils.runtime import set_seed


def main() -> None:
    cfg = load_config(ROOT / "configs" / "orbench.yaml")
    set_seed(cfg["project"]["seed"])

    data_root = str((ROOT / cfg["data"]["root"]).resolve())
    device = torch.device(cfg["project"]["device"])

    train_dataset = OrBenchDataset(root=data_root, split="train")
    unique_pids = sorted({item["pid"] for item in train_dataset.samples})
    pid2cls = {pid: idx for idx, pid in enumerate(unique_pids)}

    loader = DataLoader(
        train_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=make_train_collate(pid2cls),
    )

    model = TinyCNN(num_classes=len(unique_pids), embedding_dim=cfg["model"]["embedding_dim"])
    result = train_classifier(
        model=model,
        loader=loader,
        device=device,
        epochs=cfg["train"]["epochs"],
        lr=cfg["train"]["lr"],
        checkpoint_path=(ROOT / cfg["train"]["checkpoint_path"]),
    )

    print(
        f"Train done: epochs={result.epochs}, steps={result.steps}, "
        f"best_loss={result.best_loss:.4f}, ckpt={result.checkpoint_path}"
    )


if __name__ == "__main__":
    main()
