import json
import os
from typing import List, Dict

from PIL import Image
import torchvision.transforms as T

from src.reid_system.data.sample_utils import make_sample
from src.reid_system.data.path_utils import to_abs_path


class OrBenchDataset:
    """
    ORBench 训练集 Dataset（当前版本仅用于 train）

    - 读取 train_annos.json
    - 解析 pid / modality
    - 输出 image tensor + 其它字段
    """

    def __init__(self, root: str, split: str = "train"):
        # 数据集根目录，例如 E:/.../data/ORBench
        self.root = root

        # 数据划分（目前只支持 train）
        self.split = split

        # 样本列表（每条是统一 sample dict）
        self.samples: List[Dict] = []

        # -----------------------------
        # 训练/测试使用不同 transform
        # -----------------------------
        if self.split == "train":
            # 训练：随机增强
            self.transform = T.Compose([
                T.Resize((384, 128)),
                T.RandomHorizontalFlip(p=0.5),
                T.Pad(10),
                T.RandomCrop((384, 128)),
                T.ToTensor(),
                T.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
            ])
        else:
            # 测试/推理：确定性（但本 Dataset 暂不用于 test）
            self.transform = T.Compose([
                T.Resize((384, 128)),
                T.ToTensor(),
            ])

        # -----------------------------
        # 加载标注
        # -----------------------------
        if self.split == "train":
            anno_file = os.path.join(self.root, "train_annos.json")
            self._load_annotations(anno_file)
        else:
            # test 不用这个 Dataset（test 走 OrBenchTestProtocol）
            # 这里不加载，避免 samples 为空导致误用
            pass

    # -----------------------------------------------------
    # 读取 JSON 并构造样本列表
    # -----------------------------------------------------
    def _load_annotations(self, anno_file: str):
        # 打开 JSON 标注文件
        with open(anno_file, "r", encoding="utf-8") as f:
            data = json.load(f)  # 期望 data 是 list，每个元素是 dict

        # 遍历所有标注条目
        for item in data:
            # split 兼容：如果 item 没有 split 字段，默认认为属于当前 split
            item_split = item.get("split", self.split)
            if item_split != self.split:
                continue

            # file_path 例如：vis/0001/xxxx.jpg
            file_path = item["file_path"]

            # pid 从路径第二段解析：vis/0001/... -> 0001
            pid_str = file_path.split("/")[1]
            pid = int(pid_str)

            # 文本描述（可能不存在）
            caption = item.get("caption", "")

            # 统一构造 sample（存相对路径）
            sample = make_sample(
                pid=pid,
                file_path=file_path,
                caption=caption,
                source="train"
            )

            # ✅ 只 append 一次（你之前重复 append 了）
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        # 取 sample（包含 file_path 等）
        sample = self.samples[index]

        # 拼接绝对路径
        abs_path = to_abs_path(self.root, sample)

        # 读取图像（统一转 RGB 三通道）
        img = Image.open(abs_path).convert("RGB")

        # 图像预处理 -> tensor
        img_tensor = self.transform(img)

        # 返回训练用样本（image tensor + 标签等）
        return {
            "image": img_tensor,  # Tensor [3,384,128]
            "pid": sample["pid"],
            "modality": sample["modality"],
            "caption": sample["caption"],
            "source": sample["source"]
        }
