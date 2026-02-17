import torch


def make_train_collate(pid2cls: dict):
    """
    训练用 collate_fn 工厂函数（推荐）

    为什么要“工厂函数”？
    - CrossEntropyLoss 要求 target 必须是 [0, num_classes-1]
    - 原始 pid 不一定连续
    - 在 collate 阶段做 pid 映射最稳定（一定会执行）

    输入：
      pid2cls: dict，例如 {601: 0, 602: 1, ...}

    输出：
      collate_fn(batch) -> dict:
      {
        "images": Tensor [B,3,384,128],
        "pids": Tensor [B],            # 连续类别 id
        "modalities": list[str],
        "captions": list[str],
        "sources": list[str],
      }
    """

    def collate(batch: list) -> dict:
        # 1) 堆叠图像 -> [B,3,384,128]
        images = torch.stack([x["image"] for x in batch], dim=0)

        # 2) pid 映射为连续类别 id（CrossEntropy 必须）
        pids = torch.tensor([pid2cls[int(x["pid"])] for x in batch], dtype=torch.long)

        # 3) 其它字段先保留为 list（文本 tokenizing 以后再做）
        modalities = [x["modality"] for x in batch]
        captions = [x["caption"] for x in batch]
        sources = [x["source"] for x in batch]

        return {
            "images": images,
            "pids": pids,
            "modalities": modalities,
            "captions": captions,
            "sources": sources,
        }

    return collate


def simple_infer_collate(batch: list) -> dict:
    """
    推理/评估用的简单 collate（不做 pid 映射）
    后续做 query/gallery 特征提取时会用到。

    输出：
      {
        "images": Tensor [B,3,384,128],
        "pids": Tensor [B],          # 原始 pid（不映射）
        "modalities": list[str],
        "captions": list[str],
        "sources": list[str],
      }
    """
    images = torch.stack([x["image"] for x in batch], dim=0)
    pids = torch.tensor([int(x["pid"]) for x in batch], dtype=torch.long)
    modalities = [x["modality"] for x in batch]
    captions = [x["caption"] for x in batch]
    sources = [x["source"] for x in batch]
    return {
        "images": images,
        "pids": pids,
        "modalities": modalities,
        "captions": captions,
        "sources": sources,
    }
