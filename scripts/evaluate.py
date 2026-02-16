from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reid_system.utils.config import load_config


def main():
    cfg = load_config(ROOT / "configs" / "orbench.yaml")

    topk = cfg["eval"]["topk"]
    dataset_root = cfg["data"]["root"]
    modalities = cfg["data"]["modalities"]

    print("=== Evaluate Script ===")
    print("Dataset root:", dataset_root)
    print("Modalities:", modalities)
    print("Eval TopK:", topk)
    print("(NOTE) Real evaluation will be implemented later.")


if __name__ == "__main__":
    main()
