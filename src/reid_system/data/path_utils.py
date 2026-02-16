import os


def to_abs_path(dataset_root: str, sample: dict) -> str:
    """
    把 sample["file_path"] 拼成绝对路径
    dataset_root: ORBench 根目录，例如 E:/.../data/ORBench
    """
    return os.path.join(dataset_root, sample["file_path"])


def exists_on_disk(dataset_root: str, sample: dict) -> bool:
    """
    检查 sample 对应的文件是否真实存在
    """
    abs_path = to_abs_path(dataset_root, sample)
    return os.path.exists(abs_path)
