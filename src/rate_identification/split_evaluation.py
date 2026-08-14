from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.neighbors import NearestNeighbors

from .proxy import build_cluster_scale_proxy


MAGNIFICATION_LABELS = ["2.5x", "5x", "20x", "50x", "100x"]
WINDOW_CLASS_BY_LABEL = {
    "2.5x": "256",
    "5x": "512",
    "20x": "2048",
    "50x": "noSW",
    "100x": "noSW",
}
WINDOW_LABELS = ["256", "512", "2048", "noSW"]
_ROTATION_LABEL_PATTERN = re.compile(r"(\d+(?:p\d+)?)x", re.IGNORECASE)


def parse_rotation_label(path: Union[str, Path]) -> str:
    match = _ROTATION_LABEL_PATTERN.search(Path(path).name)
    if match is None:
        raise ValueError(f"Cannot parse magnification from image name: {path}")
    value = float(match.group(1).lower().replace("p", "."))
    label = f"{value:g}x"
    if label not in MAGNIFICATION_LABELS:
        raise ValueError(f"Unsupported magnification label {label!r} in: {path}")
    return label


def label_to_window_class(label: str) -> str:
    try:
        return WINDOW_CLASS_BY_LABEL[label]
    except KeyError as exc:
        raise ValueError(f"Unsupported magnification label: {label}") from exc


class EmbeddingScaleClassifier:
    """Existing scale-proxy pipeline adapted to a fixed train/evaluation split."""

    def __init__(
        self,
        pca_components: Optional[int],
        knn_neighbors: int = 5,
        cluster_count: int = 6,
        random_state: int = 42,
    ) -> None:
        self.pca_components = pca_components
        self.knn_neighbors = knn_neighbors
        self.cluster_count = cluster_count
        self.random_state = random_state
        self.pca: Optional[PCA] = None
        self.neighbors: Optional[NearestNeighbors] = None
        self.scale_proxy_db: Optional[np.ndarray] = None
        self.class_medians: Dict[str, float] = {}

    def fit(
        self,
        train_embeddings: np.ndarray,
        train_labels: Sequence[str],
    ) -> "EmbeddingScaleClassifier":
        embeddings = _validate_embeddings(train_embeddings, "train_embeddings")
        labels = list(train_labels)
        if embeddings.shape[0] != len(labels):
            raise ValueError("Train embedding and label counts do not match.")
        if embeddings.shape[0] < 2:
            raise ValueError("At least two training samples are required.")
        unknown = sorted(set(labels) - set(MAGNIFICATION_LABELS))
        if unknown:
            raise ValueError(f"Unsupported training labels: {unknown}")

        if self.pca_components is None:
            reduced = embeddings
            self.pca = None
        else:
            if self.pca_components <= 0:
                raise ValueError("pca_components must be positive or None.")
            component_count = min(
                self.pca_components,
                embeddings.shape[0] - 1,
                embeddings.shape[1],
            )
            self.pca = PCA(n_components=component_count, random_state=self.random_state)
            reduced = self.pca.fit_transform(embeddings)

        self.scale_proxy_db = build_cluster_scale_proxy(
            reduced,
            cluster_count=self.cluster_count,
        )
        neighbor_count = min(self.knn_neighbors, reduced.shape[0])
        self.neighbors = NearestNeighbors(n_neighbors=neighbor_count)
        self.neighbors.fit(reduced)

        train_scales = self._predict_scales_from_reduced(reduced)
        grouped: Dict[str, List[float]] = {}
        for scale, label in zip(train_scales, labels):
            grouped.setdefault(label, []).append(float(scale))
        self.class_medians = {
            label: float(np.median(values))
            for label, values in grouped.items()
        }
        return self

    def predict(self, embeddings: np.ndarray) -> Tuple[List[str], np.ndarray]:
        values = _validate_embeddings(embeddings, "embeddings")
        if self.neighbors is None or self.scale_proxy_db is None or not self.class_medians:
            raise RuntimeError("Classifier must be fitted before prediction.")
        reduced = self.pca.transform(values) if self.pca is not None else values
        scales = self._predict_scales_from_reduced(reduced)
        labels = [
            min(
                self.class_medians,
                key=lambda label: abs(float(scale) - self.class_medians[label]),
            )
            for scale in scales
        ]
        return labels, scales

    def _predict_scales_from_reduced(self, reduced: np.ndarray) -> np.ndarray:
        assert self.neighbors is not None
        assert self.scale_proxy_db is not None
        distances, indices = self.neighbors.kneighbors(reduced)
        weights = np.exp(-distances)
        weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
        return np.sum(weights * self.scale_proxy_db[indices], axis=1)


@dataclass
class EvaluationResult:
    sample_count: int
    labels: List[str]
    window_labels: List[str]
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    window_accuracy: float
    confusion_counts: np.ndarray
    confusion_normalized: np.ndarray
    window_confusion_counts: np.ndarray
    window_confusion_normalized: np.ndarray
    rows: List[Dict[str, object]]


def evaluate_embeddings(
    classifier: EmbeddingScaleClassifier,
    embeddings: np.ndarray,
    true_labels: Sequence[str],
    image_paths: Sequence[Union[str, Path]],
) -> EvaluationResult:
    labels = list(true_labels)
    images = [str(path) for path in image_paths]
    if len(labels) != len(images):
        raise ValueError("Evaluation image and label counts do not match.")
    if np.asarray(embeddings).shape[0] != len(labels):
        raise ValueError("Evaluation embedding and label counts do not match.")

    predicted_labels, predicted_scales = classifier.predict(embeddings)
    true_windows = [label_to_window_class(label) for label in labels]
    predicted_windows = [label_to_window_class(label) for label in predicted_labels]

    counts = confusion_matrix(labels, predicted_labels, labels=MAGNIFICATION_LABELS)
    window_counts = confusion_matrix(true_windows, predicted_windows, labels=WINDOW_LABELS)
    rows = []
    for image, true_label, predicted_label, scale, true_window, predicted_window in zip(
        images,
        labels,
        predicted_labels,
        predicted_scales,
        true_windows,
        predicted_windows,
    ):
        rows.append(
            {
                "image": image,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "predicted_scale": float(scale),
                "true_window_class": true_window,
                "predicted_window_class": predicted_window,
                "correct": predicted_label == true_label,
                "window_correct": predicted_window == true_window,
            }
        )

    return EvaluationResult(
        sample_count=len(labels),
        labels=list(MAGNIFICATION_LABELS),
        window_labels=list(WINDOW_LABELS),
        accuracy=float(accuracy_score(labels, predicted_labels)),
        balanced_accuracy=float(balanced_accuracy_score(labels, predicted_labels)),
        macro_f1=float(
            f1_score(
                labels,
                predicted_labels,
                labels=MAGNIFICATION_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        window_accuracy=float(accuracy_score(true_windows, predicted_windows)),
        confusion_counts=counts,
        confusion_normalized=_row_normalize(counts),
        window_confusion_counts=window_counts,
        window_confusion_normalized=_row_normalize(window_counts),
        rows=rows,
    )


def _validate_embeddings(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    totals = values.sum(axis=1, keepdims=True)
    return np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)
