import json
import os
from src.reid_system.data.sample_utils import make_sample



class OrBenchTestProtocol:
    """
    ORBench 测试协议解析器（最终版、对齐版）

    test_gallery_and_queries.json 顶层是 dict：
    - "RGB_GALLERY": list，每条是 [pid, file_path]
    - 其它 key（如 "NIR"、"NIR+TEXT"、"CP+SK"...）: list
        - 每条可能是 [pid, file_path]
        - 或 [pid, file_path, caption]
    """

    def __init__(self, root: str):
        # ORBench 根目录，例如：E:/.../data/ORBench
        self.root = root

        # test 协议文件路径
        self.protocol_file = os.path.join(root, "test_gallery_and_queries.json")

        # 存储 json 读出来的内容（dict）
        self.data = {}

        # 初始化时直接加载
        self._load()

    def _load(self):
        """读取 test_gallery_and_queries.json 到 self.data"""
        with open(self.protocol_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def list_keys(self):
        """列出所有协议 key（包括 RGB_GALLERY）"""
        return list(self.data.keys())

    def parse_gallery(self):
        """
        解析图库 RGB_GALLERY。
        返回：list[dict]
        每条结构：
        {
            "pid": int,
            "file_path": str,
            "caption": ""   # gallery 默认无 caption，统一字段方便后续处理
        }
        """
        gallery_raw = self.data["RGB_GALLERY"]

        gallery = []
        for item in gallery_raw:
            # item 形如：[pid, file_path]
            pid = int(item[0])
            file_path = item[1]

            # gallery 来自 RGB_GALLERY，因此 source="gallery"
            gallery.append(
                make_sample(pid=pid, file_path=file_path, caption="", source="gallery")
            )

        return gallery

    def parse_query(self, protocol_name: str):
        """
        统一解析 query（对齐 ORBench 的两种 item 格式）：

        - [pid, file_path]               → caption = ""
        - [pid, file_path, caption]      → caption = item[2]

        返回：list[dict]
        每条结构：
        {
            "pid": int,
            "file_path": str,
            "caption": str
        }
        """
        # 取出对应协议的原始列表，例如 data["NIR"] 或 data["NIR+TEXT"]
        query_raw = self.data[protocol_name]

        query = []
        for item in query_raw:
            pid = int(item[0])
            file_path = item[1]

            # 如果第三项存在，就当做 caption；否则 caption 为空
            caption = item[2] if len(item) >= 3 else ""

            # query 是查询项，因此 source="query"
            query.append(
                make_sample(pid=pid, file_path=file_path, caption=caption, source="query")
            )

        return query

    def abs_path(self, sample: dict):
        """
        把 sample["file_path"] 从相对路径拼成绝对路径
        """
        return os.path.join(self.root, sample["file_path"])

