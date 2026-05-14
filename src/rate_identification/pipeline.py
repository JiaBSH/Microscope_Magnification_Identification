from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from .config import PipelineConfig, WindowMappingConfig
from .data import load_rgb_image, make_inference_views
from .features import build_extractor
from .proxy import build_cluster_scale_proxy
from .window import scale_to_window


class ScaleEstimationPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.extractor = build_extractor(
            name=self.config.extractor_name,
            resize_edge=self.config.resize_long_edge,
        )
        self.pca = None  # type: Optional[PCA]
        self.neighbors = None  # type: Optional[NearestNeighbors]
        self.scale_proxy_db = None  # type: Optional[np.ndarray]
        self.reduced_embeddings = None  # type: Optional[np.ndarray]

    def fit(self, image_paths: List[Union[str, Path]]) -> "ScaleEstimationPipeline":
        if not image_paths:
            raise ValueError("No images provided for fitting.")

        embeddings = self.extractor.extract_many(image_paths)
        return self.fit_embeddings(embeddings)

    def fit_embeddings(self, embeddings: np.ndarray) -> "ScaleEstimationPipeline":
        if embeddings.size == 0:
            raise ValueError("No embeddings provided for fitting.")

        component_count = min(
            self.config.pca_components,
            embeddings.shape[0],
            embeddings.shape[1],
        )
        self.pca = PCA(n_components=component_count, random_state=42)
        self.reduced_embeddings = self.pca.fit_transform(embeddings)
        self.scale_proxy_db = build_cluster_scale_proxy(
            self.reduced_embeddings,
            cluster_count=self.config.cluster_count,
        )
        neighbor_count = min(self.config.knn_neighbors, embeddings.shape[0])
        self.neighbors = NearestNeighbors(n_neighbors=neighbor_count)
        self.neighbors.fit(self.reduced_embeddings)
        return self

    def predict_scale(self, image_path: Union[str, Path]) -> float:
        if self.pca is None or self.neighbors is None or self.scale_proxy_db is None:
            raise RuntimeError("Pipeline must be fitted or loaded before inference.")

        if self.config.use_multiview_inference:
            image = load_rgb_image(image_path)
            scales = [self._predict_from_pil(view) for view in make_inference_views(image)]
            return float(np.mean(scales))
        return float(self._predict_from_path(image_path))

    def predict(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        scale = self.predict_scale(image_path)
        window_size = scale_to_window(scale, self.config.window)
        return {"scale": scale, "window_size": window_size}

    def predict_scale_from_embedding(self, embedding: np.ndarray) -> float:
        if self.pca is None or self.neighbors is None or self.scale_proxy_db is None:
            raise RuntimeError("Pipeline must be fitted or loaded before inference.")
        return self._predict_from_embedding(embedding)

    def save(self, path: Union[str, Path]) -> None:
        if self.pca is None or self.neighbors is None or self.scale_proxy_db is None:
            raise RuntimeError("Pipeline must be fitted before saving.")

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "config": self._serialize_config(),
                "pca": self.pca,
                "neighbors": self.neighbors,
                "scale_proxy_db": self.scale_proxy_db,
                "reduced_embeddings": self.reduced_embeddings,
            },
            target,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ScaleEstimationPipeline":
        payload = joblib.load(path)
        config = cls._deserialize_config(payload["config"])
        pipeline = cls(config=config)
        pipeline.pca = payload["pca"]
        pipeline.neighbors = payload["neighbors"]
        pipeline.scale_proxy_db = payload["scale_proxy_db"]
        pipeline.reduced_embeddings = payload.get("reduced_embeddings")
        return pipeline

    def _predict_from_path(self, image_path: Union[str, Path]) -> float:
        embedding = self.extractor.extract(image_path)
        return self._predict_from_embedding(embedding)

    def _predict_from_pil(self, image) -> float:
        embedding = self.extractor.extract_image(image)
        return self._predict_from_embedding(embedding)

    def _predict_from_embedding(self, embedding: np.ndarray) -> float:
        assert self.pca is not None
        assert self.neighbors is not None
        assert self.scale_proxy_db is not None

        reduced = self.pca.transform([embedding])
        distances, indices = self.neighbors.kneighbors(reduced)
        weights = np.exp(-distances[0])
        weights = weights / max(weights.sum(), 1e-8)
        return float(np.sum(weights * self.scale_proxy_db[indices[0]]))

    def _serialize_config(self) -> Dict[str, Any]:
        payload = asdict(self.config)
        payload["window"] = asdict(self.config.window)
        return payload

    @staticmethod
    def _deserialize_config(payload: Dict[str, Any]) -> PipelineConfig:
        window = WindowMappingConfig(**payload["window"])
        return PipelineConfig(
            resize_long_edge=payload["resize_long_edge"],
            pca_components=payload["pca_components"],
            knn_neighbors=payload["knn_neighbors"],
            cluster_count=payload["cluster_count"],
            extractor_name=payload["extractor_name"],
            use_multiview_inference=payload["use_multiview_inference"],
            window=window,
        )
