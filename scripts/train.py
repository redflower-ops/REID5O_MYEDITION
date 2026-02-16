from pathlib import Path
import sys

# 让 Python 能找到 src/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reid_system.utils.config import load_config


def main():
    # 1. 读取配置
    cfg = load_config(ROOT / "configs" / "orbench.yaml")

    # 2. 使用配置里的内容
    project_name = cfg["project"]["name"]
    device = cfg["project"]["device"]

    batch_size = cfg["data"]["batch_size"]
    modalities = cfg["data"]["modalities"]

    epochs = cfg["train"]["epochs"]
    lr = cfg["train"]["lr"]

    # 3. 打印出来，验证“配置真的进来了”
    print("=== Train Script ===")
    print("Project:", project_name)
    print("Device:", device)
    print("Modalities:", modalities)
    print("Batch size:", batch_size)
    print("Epochs:", epochs)
    print("Learning rate:", lr)


if __name__ == "__main__":
    main()
