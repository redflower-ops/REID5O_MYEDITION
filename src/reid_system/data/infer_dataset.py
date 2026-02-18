from __future__ import annotations

from typing import Dict, List

from PIL import Image
import torchvision.transforms as T

from src.reid_system.data.path_utils import to_abs_path


class ProtocolImageDataset:
    """Wrap parsed protocol samples for gallery/query feature extraction."""

    def __init__(self, root: str, samples: List[Dict]):
        self.root = root
        self.samples = samples
        self.transform = T.Compose([
            T.Resize((384, 128)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]
        image = Image.open(to_abs_path(self.root, sample)).convert("RGB")
        return {
            "image": self.transform(image),
            "pid": int(sample["pid"]),
            "modality": sample["modality"],
            "caption": sample["caption"],
            "source": sample["source"],
            "file_path": sample["file_path"],
        }
