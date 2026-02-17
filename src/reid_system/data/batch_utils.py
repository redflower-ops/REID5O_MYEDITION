from collections import Counter


def count_modalities(modalities: list) -> dict:
    """
    输入：modalities，例如 ["vis","vis","nir",...]
    输出：dict，例如 {"vis": 6, "nir": 2}
    """
    return dict(Counter(modalities))
