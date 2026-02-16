def infer_modality(file_path: str) -> str:
    """
    从 file_path 推断模态：
    例如 "vis/0001/xxx.jpg" -> "vis"
    """
    return file_path.split("/")[0]


def make_sample(pid: int, file_path: str, caption: str = "", source: str = "train") -> dict:
    """
    统一构造 sample（全项目只用这一种结构）
    """
    return {
        "pid": int(pid),                    # 确保是 int
        "file_path": file_path,             # 相对路径
        "modality": infer_modality(file_path),  # 从路径第一段推断模态
        "caption": caption if caption else "",  # 统一空字符串
        "source": source                    # train/gallery/query
    }
