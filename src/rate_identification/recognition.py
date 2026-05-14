from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import PipelineConfig
from .data import list_images
from .evaluation import _class_medians, _extract_embedding, _predict_label_from_medians, collect_labeled_images
from .pipeline import ScaleEstimationPipeline
from .recommendation import recommend_windows


def recognize_all_images(
    input_root: Path,
    labeled_root: Path,
    refer_root: Path,
    config: Optional[PipelineConfig] = None,
    align_to: int = 32,
    proxy_stat: str = "median",
    square_basis: str = "long",
) -> Dict[str, object]:
    active_config = config or PipelineConfig(extractor_name="dinov2_timm", use_multiview_inference=False)
    labeled_samples = collect_labeled_images(labeled_root)
    if not labeled_samples:
        raise ValueError("No labeled images found for building the recognition pipeline.")

    pipeline = ScaleEstimationPipeline(config=active_config)
    train_embeddings = np.vstack([_extract_embedding(pipeline, sample.image_path) for sample in labeled_samples])
    pipeline.fit_embeddings(train_embeddings)
    train_scales = [pipeline.predict_scale_from_embedding(embedding) for embedding in train_embeddings]
    class_medians = _class_medians(train_scales, labeled_samples)

    recommendation_results = recommend_windows(
        labeled_root=labeled_root,
        refer_root=refer_root,
        config=active_config,
        align_to=align_to,
        proxy_stat=proxy_stat,
        square_basis=square_basis,
    )
    window_lookup = {
        str(row["label"]): int(row["square_window_aligned"])
        for row in recommendation_results["recommendations"]
    }

    rows = []
    for image_path in list_images(input_root):
        embedding = _extract_embedding(pipeline, image_path)
        predicted_scale = pipeline.predict_scale_from_embedding(embedding)
        predicted_label = _predict_label_from_medians(predicted_scale, class_medians)
        rows.append(
            {
                "image": str(image_path),
                "predicted_label": predicted_label,
                "predicted_scale": predicted_scale,
                "recommended_square_window": window_lookup[predicted_label],
                "proxy_stat": proxy_stat,
                "square_basis": square_basis,
            }
        )

    return {
        "image_count": len(rows),
        "proxy_stat": proxy_stat,
        "square_basis": square_basis,
        "rows": rows,
    }


def write_recognition_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "image",
        "predicted_label",
        "predicted_scale",
        "recommended_square_window",
        "proxy_stat",
        "square_basis",
    ]
    with path.open("w", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(
                [
                    row["image"],
                    row["predicted_label"],
                    f"{float(row['predicted_scale']):.8f}",
                    row["recommended_square_window"],
                    row["proxy_stat"],
                    row["square_basis"],
                ]
            )


def format_recognition_report(results: Dict[str, object]) -> str:
    return "\n".join(
        [
            f"image_count={results['image_count']}",
            f"proxy_stat={results['proxy_stat']}",
            f"square_basis={results['square_basis']}",
            "saved_predictions=per-image scale, label, and recommended square window",
        ]
    )
