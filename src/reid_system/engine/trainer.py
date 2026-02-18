from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader


@dataclass
class TrainResult:
    epochs: int
    steps: int
    best_loss: float
    checkpoint_path: Path


def train_classifier(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    checkpoint_path: Path,
) -> TrainResult:
    """Train a simple classification baseline and save the latest checkpoint."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)

    best_loss = float("inf")
    steps = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch in loader:
            images = batch["images"].to(device)
            labels = batch["pids"].to(device)

            outputs = model(images)
            loss = criterion(outputs["logits"], labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            steps += 1

        avg_loss = running_loss / max(len(loader), 1)
        if avg_loss < best_loss:
            best_loss = avg_loss

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "avg_loss": avg_loss,
            },
            checkpoint_path,
        )

    return TrainResult(
        epochs=epochs,
        steps=steps,
        best_loss=best_loss,
        checkpoint_path=checkpoint_path,
    )
