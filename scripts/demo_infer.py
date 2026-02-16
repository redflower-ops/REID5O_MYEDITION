from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reid_system.utils.config import load_config


def main():
    cfg = load_config(ROOT / "configs" / "orbench.yaml")

    topk = cfg["infer"]["topk"]
    device = cfg["project"]["device"]
    modalities = cfg["data"]["modalities"]

    print("=== Infer Demo Script ===")
    print("Device:", device)
    print("Modalities:", modalities)
    print("Infer TopK:", topk)
    print("(NOTE) Real retrieval demo will be implemented later.")


if __name__ == "__main__":
    main()
