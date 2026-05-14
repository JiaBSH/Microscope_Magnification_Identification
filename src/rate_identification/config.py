from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WindowMappingConfig:
    w_min: int = 512
    w_max: int = 1024
    alpha: float = 800.0
    beta: float = 1.2


@dataclass
class PipelineConfig:
    resize_long_edge: int = 1024
    pca_components: int = 64
    knn_neighbors: int = 5
    cluster_count: int = 6
    extractor_name: str = "robust_handcrafted"
    use_multiview_inference: bool = True
    window: WindowMappingConfig = field(default_factory=WindowMappingConfig)
