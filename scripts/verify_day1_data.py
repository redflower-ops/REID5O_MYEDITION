from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reid_system.data.orbench_dataset import OrBenchDataset
from src.reid_system.data.orbench_protocol import OrBenchTestProtocol
from src.reid_system.data.path_utils import to_abs_path, exists_on_disk


def main():
    data_root = (ROOT / "data" / "ORBench").resolve()
    data_root_str = str(data_root)

    print("Data root:", data_root_str)

    # ----------------------------
    # 1) 验证 train dataset
    # ----------------------------
    train_ds = OrBenchDataset(root=data_root_str, split="train")
    train_sample = train_ds[0]
    print("\n[TRAIN] sample:", train_sample)
    print("[TRAIN] abs path:", to_abs_path(data_root_str, train_sample))
    print("[TRAIN] exists:", exists_on_disk(data_root_str, train_sample))

    # ----------------------------
    # 2) 验证 gallery
    # ----------------------------
    proto = OrBenchTestProtocol(root=data_root_str)
    gallery = proto.parse_gallery()
    g0 = gallery[0]
    print("\n[GALLERY] sample:", g0)
    print("[GALLERY] abs path:", to_abs_path(data_root_str, g0))
    print("[GALLERY] exists:", exists_on_disk(data_root_str, g0))

    # ----------------------------
    # 3) 验证 query（用一个带文本的协议）
    # ----------------------------
    q = proto.parse_query("NIR+TEXT")
    q0 = q[0]
    print("\n[QUERY] sample:", q0)
    print("[QUERY] abs path:", to_abs_path(data_root_str, q0))
    print("[QUERY] exists:", exists_on_disk(data_root_str, q0))
    print("[QUERY] caption head:", q0["caption"][:60])


if __name__ == "__main__":
    main()
