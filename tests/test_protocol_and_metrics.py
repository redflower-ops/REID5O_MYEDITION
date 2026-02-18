import json

import numpy as np

from src.reid_system.data.orbench_protocol import OrBenchTestProtocol
from src.reid_system.engine.evaluator import topk_retrieval
from src.reid_system.metrics.retrieval import cmc_map_from_distance, cosine_distance_matrix


def test_protocol_parser(tmp_path):
    root = tmp_path
    protocol = {
        "RGB_GALLERY": [[1, "vis/0001/a.jpg"], [2, "vis/0002/b.jpg"]],
        "NIR": [[1, "nir/0001/q1.jpg"], [2, "nir/0002/q2.jpg", "caption"]],
    }
    (root / "test_gallery_and_queries.json").write_text(json.dumps(protocol), encoding="utf-8")

    parser = OrBenchTestProtocol(str(root))
    gallery = parser.parse_gallery()
    query = parser.parse_query("NIR")

    assert len(gallery) == 2
    assert gallery[0]["source"] == "gallery"
    assert query[1]["caption"] == "caption"
    assert query[0]["source"] == "query"


def test_metrics_and_topk():
    query = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    gallery = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]], dtype=np.float32)
    query_pids = np.array([1, 2])
    gallery_pids = np.array([1, 2, 1])

    dist = cosine_distance_matrix(query, gallery)
    metrics = cmc_map_from_distance(dist, query_pids, gallery_pids, topk=2)

    assert metrics["rank_1"] == 1.0
    assert metrics["mAP"] > 0.9

    items = [
        {"pid": 1, "file_path": "a.jpg", "modality": "vis"},
        {"pid": 2, "file_path": "b.jpg", "modality": "vis"},
        {"pid": 1, "file_path": "c.jpg", "modality": "vis"},
    ]
    result = topk_retrieval(query[0], gallery, items, topk=2)

    assert result[0]["file_path"] == "a.jpg"
    assert len(result) == 2
