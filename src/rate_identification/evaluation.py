from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import PipelineConfig
from .data import list_images, load_rgb_image, make_inference_views
from .pipeline import ScaleEstimationPipeline


MAGNIFICATION_PATTERN = re.compile(r"(\d+(?:[._]\d+)?)x", re.IGNORECASE)


@dataclass
class LabeledImage:
    image_path: Path
    label_name: str
    magnification: float
    is_swinir: bool


def collect_labeled_images(root: Path) -> List[LabeledImage]:
    samples = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        match = MAGNIFICATION_PATTERN.search(directory.name)
        if match is None:
            continue
        magnification = float(match.group(1).replace("_", "."))
        label_name = f"{magnification:g}x"
        for image_path in list_images(directory):
            lower_name = image_path.name.lower()
            is_swinir = "swinir" in directory.name.lower() or "swinir" in lower_name
            samples.append(
                LabeledImage(
                    image_path=image_path,
                    label_name=label_name,
                    magnification=magnification,
                    is_swinir=is_swinir,
                )
            )
    return samples


def _class_medians(scales: Sequence[float], samples: Sequence[LabeledImage]) -> Dict[str, float]:
    grouped = {}
    for scale, sample in zip(scales, samples):
        grouped.setdefault(sample.label_name, []).append(scale)
    return {label: float(np.median(values)) for label, values in grouped.items()}


def _predict_label_from_medians(scale: float, medians: Dict[str, float]) -> str:
    return min(medians, key=lambda label: abs(scale - medians[label]))


def _safe_corr(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2:
        return 1.0
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if np.allclose(x_array, x_array[0]) or np.allclose(y_array, y_array[0]):
        return 0.0
    return float(np.corrcoef(x_array, y_array)[0, 1])


def evaluate_leave_one_out(
    root: Path,
    config: Optional[PipelineConfig] = None,
) -> Dict[str, object]:
    active_config = config or PipelineConfig()
    samples = collect_labeled_images(root)
    if len(samples) < 2:
        raise ValueError("At least two labeled images are required for evaluation.")

    base_pipeline = ScaleEstimationPipeline(config=active_config)
    embeddings = np.vstack([
        _extract_embedding(base_pipeline, sample.image_path)
        for sample in samples
    ])

    predicted_scales = []  # type: List[float]
    true_labels = []  # type: List[str]
    true_magnifications = []  # type: List[float]
    per_sample_rows = []  # type: List[Dict[str, object]]

    for holdout_index, holdout in enumerate(samples):
        train_samples = [sample for idx, sample in enumerate(samples) if idx != holdout_index]
        train_embeddings = np.delete(embeddings, holdout_index, axis=0)

        pipeline = ScaleEstimationPipeline(config=active_config).fit_embeddings(train_embeddings)
        train_scales = [pipeline.predict_scale_from_embedding(embedding) for embedding in train_embeddings]
        holdout_scale = pipeline.predict_scale_from_embedding(embeddings[holdout_index])
        medians = _class_medians(train_scales, train_samples)
        predicted_label = _predict_label_from_medians(holdout_scale, medians)

        predicted_scales.append(holdout_scale)
        true_labels.append(holdout.label_name)
        true_magnifications.append(holdout.magnification)
        per_sample_rows.append(
            {
                "image": str(holdout.image_path),
                "true_label": holdout.label_name,
                "predicted_label": predicted_label,
                "predicted_scale": holdout_scale,
                "is_swinir": holdout.is_swinir,
                "correct": predicted_label == holdout.label_name,
            }
        )

    correct = [row["correct"] for row in per_sample_rows]
    swinir_correct = [row["correct"] for row in per_sample_rows if row["is_swinir"]]
    native_correct = [row["correct"] for row in per_sample_rows if not row["is_swinir"]]
    labels = sorted(set(true_labels), key=lambda label: float(label[:-1]))
    confusion_matrix = _build_confusion_matrix(per_sample_rows, labels)

    ordered_by_scale = [sample.label_name for _, sample in sorted(zip(predicted_scales, samples), key=lambda item: item[0])]

    return {
        "sample_count": len(samples),
        "labels": labels,
        "accuracy": float(np.mean(correct)),
        "native_accuracy": float(np.mean(native_correct)) if native_correct else None,
        "swinir_accuracy": float(np.mean(swinir_correct)) if swinir_correct else None,
        "native_sample_count": len(native_correct),
        "swinir_sample_count": len(swinir_correct),
        "scale_magnification_correlation": _safe_corr(predicted_scales, true_magnifications),
        "predicted_scale_min": float(np.min(predicted_scales)),
        "predicted_scale_max": float(np.max(predicted_scales)),
        "predicted_label_order_by_scale": ordered_by_scale,
        "confusion_matrix": confusion_matrix,
        "rows": per_sample_rows,
    }


def _extract_embedding(pipeline: ScaleEstimationPipeline, image_path: Path) -> np.ndarray:
    if pipeline.config.use_multiview_inference:
        image = load_rgb_image(image_path)
        view_embeddings = [
            pipeline.extractor.extract_image(view)
            for view in make_inference_views(image)
        ]
        return np.mean(np.vstack(view_embeddings), axis=0)
    return pipeline.extractor.extract(image_path)


def format_evaluation_report(results: Dict[str, object]) -> str:
    lines = [
        f"samples={results['sample_count']}",
        f"labels={','.join(results['labels'])}",
        f"accuracy={results['accuracy']:.4f}",
        f"native_samples={results['native_sample_count']}",
        f"swinir_samples={results['swinir_sample_count']}",
        f"native_accuracy={_format_optional_metric(results['native_accuracy'])}",
        f"swinir_accuracy={_format_optional_metric(results['swinir_accuracy'])}",
        f"scale_magnification_correlation={results['scale_magnification_correlation']:.4f}",
        f"predicted_scale_range={results['predicted_scale_min']:.4f},{results['predicted_scale_max']:.4f}",
        "confusion_matrix=rows:true,cols:pred",
        _format_confusion_matrix(results["labels"], results["confusion_matrix"]),
    ]
    return "\n".join(lines)


def _format_optional_metric(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def write_rows_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["image", "true_label", "predicted_label", "predicted_scale", "is_swinir", "correct"]
    with path.open("w", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(
                [
                    str(row["image"]),
                    str(row["true_label"]),
                    str(row["predicted_label"]),
                    f"{float(row['predicted_scale']):.8f}",
                    str(row["is_swinir"]),
                    str(row["correct"]),
                ]
            )


def write_confusion_csv(path: Path, labels: Sequence[str], matrix: Dict[str, Dict[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred"] + list(labels))
        for true_label in labels:
            writer.writerow([true_label] + [matrix[true_label][predicted_label] for predicted_label in labels])


def _build_confusion_matrix(rows: Sequence[Dict[str, object]], labels: Sequence[str]) -> Dict[str, Dict[str, int]]:
    matrix = {
        true_label: {predicted_label: 0 for predicted_label in labels}
        for true_label in labels
    }
    for row in rows:
        matrix[str(row["true_label"])][str(row["predicted_label"])] += 1
    return matrix


def _format_confusion_matrix(labels: Sequence[str], matrix: Dict[str, Dict[str, int]]) -> str:
    header = ["true\\pred"] + list(labels)
    rows = [header]
    for true_label in labels:
        rows.append([true_label] + [str(matrix[true_label][predicted_label]) for predicted_label in labels])
    widths = [max(len(row[column]) for row in rows) for column in range(len(header))]
    return "\n".join(
        " ".join(value.rjust(widths[idx]) for idx, value in enumerate(row))
        for row in rows
    )
