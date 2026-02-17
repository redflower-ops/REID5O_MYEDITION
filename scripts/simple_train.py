from pathlib import Path
import sys
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 让 Python 能 import src/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reid_system.utils.config import load_config
from src.reid_system.data.orbench_dataset import OrBenchDataset
from src.reid_system.data.collate import make_train_collate


def count_modalities(modalities: list) -> dict:
    """统计一个 batch 内每种模态数量"""
    return dict(Counter(modalities))


class TinyCNN(nn.Module):
    """
    最小可训练模型（用于跑通训练闭环，不追求性能）
    image -> feature -> logits -> CrossEntropyLoss
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),   # [B,16,192,64]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # [B,32,96,32]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),                            # [B,32,1,1]
            nn.Flatten(),                                            # [B,32]
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.classifier(feat)
        return logits


def main():
    # 0) 读配置 + 准备路径
    cfg = load_config(ROOT / "configs" / "orbench.yaml")
    data_root = (ROOT / cfg["data"]["root"]).resolve()
    device = torch.device(cfg["project"]["device"])

    print("Data root:", data_root)
    print("Device:", device)

    # 1) Dataset（train）
    train_ds = OrBenchDataset(root=str(data_root), split="train")
    print("Train dataset size:", len(train_ds))

    # 2) pid -> 连续类别 id 映射（CrossEntropy 必须）
    unique_pids = sorted({s["pid"] for s in train_ds.samples})
    pid2cls = {pid: i for i, pid in enumerate(unique_pids)}
    num_classes = len(unique_pids)
    print("Num classes (unique pids):", num_classes)

    # 3) DataLoader（使用“模块化”的 collate_fn）
    batch_size = cfg["data"]["batch_size"]
    collate_fn = make_train_collate(pid2cls)

    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows 先用 0 最稳
        collate_fn=collate_fn
    )

    # 4) 模型 / loss / 优化器
    model = TinyCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    # 5) 训练循环（只跑少量 step 验证闭环）
    model.train()
    max_steps = 100
    step = 0

    for epoch in range(1):
        for batch in loader:
            images = batch["images"].to(device)
            pids = batch["pids"].to(device)

            logits = model(images)
            loss = criterion(logits, pids)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 5 == 0:
                print(f"step={step:03d} loss={loss.item():.4f}")

            if step % 20 == 0:
                print("Modality counts:", count_modalities(batch["modalities"]))

            step += 1
            if step >= max_steps:
                break

    print("Done. Training pipeline OK.")


if __name__ == "__main__":
    main()
