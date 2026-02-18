# ReID5O_MyEdition

基于 ORBench 的可运行 ReID 项目骨架（训练 / 评估 / 检索推理全流程）。

## 功能
- ✅ `scripts/train.py`：训练 TinyCNN 分类基线并保存 checkpoint
- ✅ `scripts/evaluate.py`：按 ORBench 协议做检索评估（CMC + mAP）
- ✅ `scripts/infer.py`：给定单张 query 图，在 RGB Gallery 里做 Top-k 检索

## 目录结构
```text
configs/
  orbench.yaml
scripts/
  train.py
  evaluate.py
  infer.py
src/reid_system/
  data/
  engine/
  metrics/
  models/
  utils/
```

## 数据准备
将 ORBench 数据放在：

```text
data/ORBench/
  train_annos.json
  test_gallery_and_queries.json
  vis/
  nir/
  ...
```

默认路径来自 `configs/orbench.yaml` 的 `data.root`。

## 安装依赖
```bash
pip install -r requirements.txt
```

## 训练
```bash
python scripts/train.py
```

输出 checkpoint 默认保存在：`checkpoints/tiny_cnn.pt`。

## 评估
> `model.num_classes` 需要和训练时类别数一致。

```bash
python scripts/evaluate.py
```

`eval.query_protocol` 可改成 ORBench 协议 key（例如 `NIR`、`NIR+TEXT` 等）。

## 推理检索
```bash
python scripts/infer.py --query-image path/to/query.jpg --topk 5
```

输出 JSON 默认写入 `outputs/infer_results.json`。

## 测试
```bash
python -m pytest -q
```

当前测试覆盖：
- 协议解析（gallery/query）
- 检索指标与 Top-k 排序
