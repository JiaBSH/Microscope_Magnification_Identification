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

## 旋转数据固定划分混淆矩阵实验

`rotation_confusion` 使用固定的 Train / Validation / Test 划分，对下列方法做可比评估：

- 原始灰度像素 + PCA；
- 手工鲁棒特征 + PCA；
- 冻结 DINOv2，不使用 PCA；
- 冻结 DINOv2 + PCA。

运行完整实验：

```bash
mkdir -p /data/home/scvi576/run/JiaBSH/mmdetection_para/outputs/dino_window_supplement/01_scale_ablation/logs
sbatch scripts/run_rotation_confusion_matrix.sh
```

实验只使用 `images/train` 的60张图拟合 PCA、KMeans、kNN 和倍率类别尺度中位数。Validation和Test特征只用于预测，不进入拟合。DINOv2使用公开预训练权重并保持冻结，不进行微调。

主实验使用PCA32。原因是Train只有60个独立源图像；在不把相关视图错误地当作独立样本的前提下，PCA64的后部主成分估计不稳定。多视图特征在每张图内部平均，因此统计单位仍是源图像。

每种方法、每个split输出：

```text
predictions.csv
metrics.json
confusion_counts.csv
confusion_normalized.csv
window_confusion_counts.csv
window_confusion_normalized.csv
confusion_matrix.{png,svg}
window_confusion_matrix.{png,svg}
```

五倍率混淆矩阵评价倍率识别；窗口类别混淆矩阵将50×和100×统一映射到`noSW`，评价识别误差是否真正改变下游窗口策略。

## 说明

- 若未安装 `torch` / `timm`，系统会继续使用默认的手工鲁棒特征提取器。
- 默认窗口映射函数为 `W = clip(alpha * s^(-beta), Wmin, Wmax)`。
- 训练阶段的伪尺度由聚类中心排序后映射得到，可替换为频率代理或外部先验。
