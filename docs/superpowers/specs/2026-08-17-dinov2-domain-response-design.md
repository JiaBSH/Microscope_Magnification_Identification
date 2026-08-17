# DINOv2 畴区响应可视化设计

## 目标

在不训练或微调 DINOv2 的前提下，从 `vit_small_patch14_dinov2` 提取空间 patch token，检验 DINOv2 的高响应面积是否与 COCO 畴区 union mask 的像素占比相关，并生成可用于论文 Figure 2 的定性与定量图。

## 数据与防泄漏

使用 `syn_multimag/coco_rotation` 的固定划分：Train 60 张、Validation 20 张、Test 20 张。Train mask 仅用于计算畴区/背景特征原型；Validation mask 仅用于选择响应阈值；Test 只用于最终统计。额外给出全数据描述性结果，但不将其表述为独立检验。

同一图像的全部 COCO polygon 必须先栅格化并取 union，不能直接相加 annotation `area`，以免重叠区域重复计数。

## 空间对齐

保持现有分类代码的预处理：先将长边缩放到 1024，再将短边缩放到 518，最后中心裁剪为 `518×518`。mask 使用完全相同的几何变换，但采用 nearest-neighbor 插值。

DINOv2-S/14 的输入为 `518×518`，patch size 为 `14×14`，得到 `37×37=1369` 个 patch token，每个 token 384 维。去除 1 个 CLS prefix token后，将 mask 按 `14×14` 块平均得到 patch occupancy。

## 响应定义

使用 Train 的连续 occupancy 作为权重分别计算畴区和背景原型：

\[
p_d=\operatorname{norm}\left(\sum_i o_i t_i\right),\qquad
p_b=\operatorname{norm}\left(\sum_i (1-o_i)t_i\right)
\]

每个 patch 的响应分数为：

\[
s_i=\cos(t_i,p_d)-\cos(t_i,p_b)
\]

在 Validation 上遍历响应分位数，选择使二值 patch mask Dice 最大的阈值。每张图的 DINO 响应面积占比为超过该阈值的 patch 数量比例；真实畴区占比直接由对齐后的像素级 union mask 计算。

## 输出

1. 每张图的 patch token、patch occupancy、响应分数和面积占比缓存。
2. Test 的 Pearson、Spearman、控制倍率后的组内中心化相关系数、patch AUROC、Dice、IoU、畴区内外平均响应。
3. 每个倍率至少一张四联图：原图、COCO union mask、DINO 响应热力图、响应叠加图。
4. Test 的真实畴区像素占比与 DINO 响应面积占比散点图。
5. 畴区内/外 patch 响应分布图。
6. CSV、JSON、PNG、SVG 和 manifest。

## 解释边界

PCA patch map 仅作为无监督补充图，不能单独证明响应对应畴区。原型相似度属于冻结 DINOv2 表征上的有监督分析探针，不是 DINOv2 微调。只有 Test 相关性和空间重叠指标达到可接受水平时，论文才可以写“DINOv2 响应与畴区像素占比相关”；否则应如实报告其主要编码倍率或全局纹理。

