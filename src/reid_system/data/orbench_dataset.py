import json
import os
from typing import List, Dict
from src.reid_system.data.sample_utils import make_sample


class OrBenchDataset:
    def __init__(self, root: str, split: str = "train"):
        # 数据集根目录，例如 ./data/ORBench
        self.root = root

        # 数据划分（train 或 test）
        self.split = split

        # 用来存放所有样本的列表
        # 每个元素是一个字典 sample
        self.samples: List[Dict] = []

        # 根据 split 选择对应的标注文件
        if split == "train":
            anno_file = os.path.join(root, "train_annos.json")
        else:
            anno_file = os.path.join(root, "test_gallery_and_queries.json")

        # 加载标注文件
        self._load_annotations(anno_file)

    # -----------------------------------------------------
    # 读取 JSON 并构造样本列表
    # -----------------------------------------------------
    def _load_annotations(self, anno_file: str):

        # 打开 JSON 标注文件
        with open(anno_file, "r", encoding="utf-8") as f:
            data = json.load(f)   # data 是一个列表，每个元素是一个字典

        # 遍历所有标注条目
        for item in data:

            # 确保数据属于当前 split（train/test）
            # 防止 test 数据混进 train
            if item.get("split") != self.split:
                continue

            # 获取文件相对路径，例如：
            # vis/0001/xxxx.jpg
            file_path = item["file_path"]

            # -------------------------
            # 解析模态
            # -------------------------
            # vis / nir / cp / sk
            modality = file_path.split("/")[0]

            # -------------------------
            # 解析 person id
            # -------------------------
            # 路径第二层就是 pid 文件夹
            # vis/0001/xxx.jpg → 0001
            pid_str = file_path.split("/")[1]
            pid = int(pid_str)   # 转换为整数，作为训练标签

            # 拼接成完整图片路径
            full_path = os.path.join(self.root, file_path)

            # caption 可能不存在，所以用 get，默认空字符串
            caption = item.get("caption", "")

            # 用统一函数构造 sample
            # 注意：这里存的是相对路径 file_path，不再拼接绝对路径
            # 因为绝对路径属于“具体运行环境”，最好交给上层处理
            sample = make_sample(
                pid=pid,
                file_path=file_path,
                caption=caption,
                source="train"
            )

            self.samples.append(sample)

            # 添加到样本列表
            self.samples.append(sample)

    # 返回数据集总长度
    def __len__(self):
        return len(self.samples)

    # 根据索引返回单条样本
    def __getitem__(self, index: int):
        return self.samples[index]
