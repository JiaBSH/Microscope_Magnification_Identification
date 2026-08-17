# 2.5×滑窗尺寸—交叠率—分割准确性热力图设计

## 1. 目标

量化2.5×显微图像中滑窗尺寸与滑窗交叠率对实例分割准确性的联合影响，并检验DINOv2预先推荐的窗口大小是否位于高性能区域。该实验作为Figure 2新增子图，为“视觉尺度估计能够指导滑窗参数选择”提供下游任务证据。

## 2. 实验范围

- 数据：先使用`data/syn_multimag/coco_rotation/test2_5_t1/images/2p5x_00016.png`这一幅2.5×图像及其COCO标注进行单图参数敏感性实验。
- 模型：`detectors_htc-r50_custom_coco_instance`，使用现有训练权重，不重新训练。
- 参数组合：5个窗口尺寸 × 5个滑窗交叠率，共25组推理。
- 不比较5×及更高倍率，不对全部分割模型重复完整网格。

## 3. 参数网格

窗口尺寸：

```text
192, 256, 320, 400, 512 px
```

滑窗交叠率：

```text
0.00, 0.10, 0.15, 0.20, 0.30
```

其中：

- `256 px`为DINOv2尺度系统对2.5×合成图像的预先推荐值；
- `400 px`、`0.15`为当前人工固定设置；
- 其余参数围绕两者设置，用于观察局部趋势及联合效应。

## 4. 控制变量

25组实验仅改变：

- `PATCH_SIZE`
- `PATCH_OVERLAP_RATIO`

以下参数固定：

- 模型、checkpoint及数据集；
- `SCORE_THRESH=0.5`；
- `MERGE_OVERLAP_RATIO=0.3`；
- context margin规则；
- COCO max detections及其他后处理参数；
- 随机种子和软件环境。

滑窗交叠率已在当前代码中与context margin解耦，因此本实验中的横轴确实表示用户指定的滑窗几何交叠率。

## 5. 评价指标

每组推理同时计算六项主指标：

```text
segm_mAP
bbox_mAP
pixel_precision
pixel_recall
pixel_f1
pixel_iou
```

其中`segm_mAP`与`bbox_mAP`均为COCO AP@[IoU=0.50:0.95]；其余四项由预测实例mask的像素并集与GT mask像素并集计算。另保存以下辅助指标：

- `segm_mAP_50`
- `segm_mAP_75`
- 推理耗时；
- 滑窗总数。

不得把仅对成功匹配实例计算的per-instance precision/recall/F1/IoU替代上述全图像素并集指标，因为前者会排除未匹配的假阳性和假阴性。

## 6. 最小计算路径

批量实验直接复用以下现有模块：

- `postprocess.run_postprocess._load_model`和`_infer_one_image`：模型加载与滑窗推理；
- `postprocess._coco_eval.COCOResultCollector`和`evaluate_coco_from_predictions`：COCO bbox/segm评价；
- `postprocess._pixel_metrics.compute_pixel_metrics`：像素级Precision、Recall、F1和IoU。

推理阶段禁止生成或执行以下内容：

- overlay、滑窗示意图和mask可视化；
- 六边形/凸包拟合和几何尺寸统计；
- GT实例匹配、直方图、散点图和逐图柱状图；
- 与六项主指标无关的中间图像文件。

每组参数只保存一个紧凑JSON；25组全部完成后再读取JSON生成最终热力图、CSV和汇总JSON。

## 7. 图形设计

输出六张独立5×5热力图以及一张2×3组合图：

- 横轴：Sliding-window overlap ratio；
- 纵轴：Window size (px)；
- 六个面板依次为COCO Segm、COCO Box、Precision、Recall、F1-score和IoU；
- 色阶：黄—橙—红，颜色越深表示对应指标越高；
- 每个格点显示三位小数；
- 用星形边框标出DINOv2设置`(256, 0.15)`；
- 用方形边框标出人工设置`(400, 0.15)`；
- 若全局最大值与DINOv2点不同，可另用细黑框标出，但不得将其表述为重新选择的测试参数。

每个面板导出独立PNG和SVG；另导出组合图、CSV和JSON，便于论文排版与复核。

## 8. 统计解释边界

该单图网格用于先验证参数影响、执行链和作图是否正确，不作为总体统计结论，也不能替代多图测试。论文只能将其表述为代表性图像的参数敏感性案例，除非后续扩展到更多独立图像。

1. 滑窗尺寸与交叠率对掩膜精度存在联合影响；
2. 人工固定参数并非在所有区域都具有相同鲁棒性；
3. DINOv2推荐值是否落在高性能平台区，应由真实结果决定；
4. 如果DINOv2点不是最高值，只能报告其与最优点的差值及其效率优势，不能宣称其达到最优。

## 9. 验收标准

- 25个参数组合全部完成且每个组合均有独立紧凑指标JSON；
- 汇总CSV恰好包含25行，无重复或缺失组合；
- 每行同时包含COCO Segm、COCO Box、Precision、Recall、F1-score和IoU；
- 六张热力图数值与对应原始指标JSON逐格一致；
- DINOv2点与人工设置点标记位置正确；
- 脚本能够从现有指标重绘图形，无需重新推理；
- 25组推理输出目录不包含PNG、SVG、overlay、几何CSV或其他无关中间产物；
- 最终结果明确标注真实数据来源、模型、checkpoint和代码提交版本。
