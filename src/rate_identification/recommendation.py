from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from .config import PipelineConfig
from .evaluation import _extract_embedding, _class_medians, _predict_label_from_medians, collect_labeled_images
from .pipeline import ScaleEstimationPipeline


def recommend_windows(
    labeled_root: Path,
    refer_root: Path,
    config: Optional[PipelineConfig] = None,
    align_to: int = 32,
    proxy_stat: str = "median",
    square_basis: str = "short",
) -> Dict[str, object]:
    active_config = config or PipelineConfig(extractor_name="dinov2_timm", use_multiview_inference=False)
    labeled_samples = collect_labeled_images(labeled_root)
    if not labeled_samples:
        raise ValueError("No labeled images found for calibration.")

    refer_paths = sorted(path for path in refer_root.iterdir() if path.is_file())
    if not refer_paths:
        raise ValueError("No refer images found for calibration.")

    pipeline = ScaleEstimationPipeline(config=active_config)
    train_embeddings = np.vstack([_extract_embedding(pipeline, sample.image_path) for sample in labeled_samples])
    pipeline.fit_embeddings(train_embeddings)
    train_scales = [pipeline.predict_scale_from_embedding(embedding) for embedding in train_embeddings]
    class_medians = _class_medians(train_scales, labeled_samples)

    refer_rows = []
    width_proxies = []
    height_proxies = []
    short_side_proxies = []
    long_side_proxies = []

    for refer_path in refer_paths:
        embedding = _extract_embedding(pipeline, refer_path)
        predicted_scale = pipeline.predict_scale_from_embedding(embedding)
        predicted_label = _predict_label_from_medians(predicted_scale, class_medians)
        predicted_magnification = float(predicted_label[:-1])
        width, height = Image.open(refer_path).size
        width_proxy = width / predicted_magnification
        height_proxy = height / predicted_magnification
        short_proxy = min(width, height) / predicted_magnification
        long_proxy = max(width, height) / predicted_magnification

        width_proxies.append(width_proxy)
        height_proxies.append(height_proxy)
        short_side_proxies.append(short_proxy)
        long_side_proxies.append(long_proxy)
        refer_rows.append(
            {
                "image": str(refer_path),
                "width": width,
                "height": height,
                "predicted_label": predicted_label,
                "predicted_scale": predicted_scale,
                "predicted_magnification": predicted_magnification,
                "width_proxy": width_proxy,
                "height_proxy": height_proxy,
                "short_side_proxy": short_proxy,
                "long_side_proxy": long_proxy,
            }
        )

    aggregated_width_proxy = _aggregate_proxy(width_proxies, proxy_stat)
    aggregated_height_proxy = _aggregate_proxy(height_proxies, proxy_stat)
    aggregated_short_proxy = _aggregate_proxy(short_side_proxies, proxy_stat)
    aggregated_long_proxy = _aggregate_proxy(long_side_proxies, proxy_stat)
    aggregated_square_proxy = aggregated_short_proxy if square_basis == "short" else aggregated_long_proxy
    labels = sorted({sample.label_name for sample in labeled_samples}, key=lambda label: float(label[:-1]))

    recommendations = []
    for label in labels:
        magnification = float(label[:-1])
        width_window = int(round(aggregated_width_proxy * magnification))
        height_window = int(round(aggregated_height_proxy * magnification))
        short_window = int(round(aggregated_square_proxy * magnification))
        recommendations.append(
            {
                "label": label,
                "magnification": magnification,
                "window_width": width_window,
                "window_height": height_window,
                "square_window": short_window,
                "window_width_aligned": _align(width_window, align_to),
                "window_height_aligned": _align(height_window, align_to),
                "square_window_aligned": _align(short_window, align_to),
            }
        )

    return {
        "refer_image_count": len(refer_rows),
        "proxy_stat": proxy_stat,
        "square_basis": square_basis,
        "aggregated_width_proxy": aggregated_width_proxy,
        "aggregated_height_proxy": aggregated_height_proxy,
        "aggregated_short_proxy": aggregated_short_proxy,
        "aggregated_long_proxy": aggregated_long_proxy,
        "aggregated_square_proxy": aggregated_square_proxy,
        "refer_rows": refer_rows,
        "recommendations": recommendations,
    }


def format_recommendation_report(results: Dict[str, object]) -> str:
    lines = [
        f"refer_image_count={results['refer_image_count']}",
        f"proxy_stat={results['proxy_stat']}",
        f"square_basis={results['square_basis']}",
        f"aggregated_width_proxy={results['aggregated_width_proxy']:.4f}",
        f"aggregated_height_proxy={results['aggregated_height_proxy']:.4f}",
        f"aggregated_short_proxy={results['aggregated_short_proxy']:.4f}",
        f"aggregated_long_proxy={results['aggregated_long_proxy']:.4f}",
        f"aggregated_square_proxy={results['aggregated_square_proxy']:.4f}",
        "recommended_windows=match refer field of view",
        _format_recommendation_table(results["recommendations"]),
    ]
    return "\n".join(lines)


def write_recommendations_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "label",
        "magnification",
        "window_width",
        "window_height",
        "square_window",
        "window_width_aligned",
        "window_height_aligned",
        "square_window_aligned",
    ]
    with path.open("w", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow([row[column] for column in header])


def write_refer_summary_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "image",
        "width",
        "height",
        "predicted_label",
        "predicted_scale",
        "predicted_magnification",
        "width_proxy",
        "height_proxy",
        "short_side_proxy",
        "long_side_proxy",
    ]
    with path.open("w", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow([row[column] for column in header])


def _format_recommendation_table(rows: Sequence[Dict[str, object]]) -> str:
    header = [
        "label",
        "width",
        "height",
        "square",
        "width_aligned",
        "height_aligned",
        "square_aligned",
    ]
    table_rows = [header]
    for row in rows:
        table_rows.append(
            [
                str(row["label"]),
                str(row["window_width"]),
                str(row["window_height"]),
                str(row["square_window"]),
                str(row["window_width_aligned"]),
                str(row["window_height_aligned"]),
                str(row["square_window_aligned"]),
            ]
        )
    widths = [max(len(row[column]) for row in table_rows) for column in range(len(header))]
    return "\n".join(
        " ".join(value.rjust(widths[idx]) for idx, value in enumerate(row))
        for row in table_rows
    )


def _align(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return max(multiple, int(round(value / float(multiple))) * multiple)


def _aggregate_proxy(values: Sequence[float], proxy_stat: str) -> float:
    array = np.asarray(values, dtype=np.float64)
    if proxy_stat == "median":
        return float(np.median(array))
    if proxy_stat == "p75":
        return float(np.quantile(array, 0.75))
    if proxy_stat == "max":
        return float(np.max(array))
    raise ValueError(f"Unsupported proxy_stat: {proxy_stat}")
