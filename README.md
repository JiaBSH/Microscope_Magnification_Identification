# Rate Identification

用于显微图像连续视觉尺度估计与自适应滑窗大小映射的最小可运行工程实现。

## 功能

- 连续尺度估计主链路：特征提取 -> PCA -> 无监督伪尺度库 -> kNN 回归 -> 滑窗映射
- 默认内置轻量级鲁棒特征提取器，无需大模型即可跑通
- 可选切换到 `timm` 的 DINOv2 特征后端
- 支持多视图推理，降低超分辨率伪纹理影响
- 支持保存 / 加载训练好的尺度模型

## 安装

```bash
python -m pip install -e .
```

启用 DINOv2 后端：

```bash
python -m pip install -e .[dinov2]
```

## 训练

从图像目录构建无监督尺度库：

```bash
rate-identification fit data --output artifacts/scale_pipeline.joblib
```

使用 DINOv2：

```bash
rate-identification fit data --extractor dinov2_timm --output artifacts/scale_pipeline.joblib
```

## 推理

```bash
rate-identification infer data/example.png --model artifacts/scale_pipeline.joblib
```

示例输出：

```text
predicted_scale=0.436281
window_size=781
```

## 评估

对按倍率目录组织的数据做留一法评估：

```bash
rate-identification evaluate data/images --csv artifacts/eval_predictions.csv --confusion-csv artifacts/eval_confusion.csv
```

输出指标包括：

- 总体倍率分类准确率
- 原始图像准确率
- SwinIR 图像准确率
- 预测连续尺度与真实倍率的相关性
- 文本混淆矩阵

## 说明

- 若未安装 `torch` / `timm`，系统会继续使用默认的手工鲁棒特征提取器。
- 默认窗口映射函数为 `W = clip(alpha * s^(-beta), Wmin, Wmax)`。
- 训练阶段的伪尺度由聚类中心排序后映射得到，可替换为频率代理或外部先验。
