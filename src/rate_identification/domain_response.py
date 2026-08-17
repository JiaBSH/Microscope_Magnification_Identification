from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from sklearn.metrics import roc_auc_score

from .data import resize_long_edge
from .split_evaluation import MAGNIFICATION_LABELS, parse_rotation_label


@dataclass(frozen=True)
class CocoImageRecord:
    image_id: int
    image_path: Path
    file_name: str
    width: int
    height: int
    label: str
    split: str
    segmentations: Tuple[Tuple[Tuple[float, ...], ...], ...]


@dataclass
class SplitPatchData:
    records: list[CocoImageRecord]
    tokens: np.ndarray
    occupancy: np.ndarray
    pixel_fraction: np.ndarray


def load_coco_records(data_root: Path, split: str) -> list[CocoImageRecord]:
    """Load one COCO split and group every polygon annotation by image."""

    data_root = Path(data_root)
    annotation_path = data_root / "annotations" / f"instances_{split}.json"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing COCO annotations: {annotation_path}")
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    grouped: Dict[int, list[Tuple[Tuple[float, ...], ...]]] = {}
    for annotation in payload.get("annotations", []):
        segmentation = annotation.get("segmentation")
        if not isinstance(segmentation, list):
            raise ValueError(
                f"Expected polygon segmentation in {annotation_path}; "
                f"got {type(segmentation).__name__}"
            )
        polygons = tuple(tuple(float(value) for value in polygon) for polygon in segmentation)
        grouped.setdefault(int(annotation["image_id"]), []).append(polygons)

    records: list[CocoImageRecord] = []
    image_dir = data_root / "images" / split
    for image in sorted(payload.get("images", []), key=lambda row: row["file_name"]):
        image_id = int(image["id"])
        image_path = image_dir / str(image["file_name"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing COCO image: {image_path}")
        records.append(
            CocoImageRecord(
                image_id=image_id,
                image_path=image_path,
                file_name=str(image["file_name"]),
                width=int(image["width"]),
                height=int(image["height"]),
                label=parse_rotation_label(str(image["file_name"])),
                split=split,
                segmentations=tuple(grouped.get(image_id, [])),
            )
        )
    return records


def rasterize_polygon_union(
    size: Tuple[int, int],
    segmentations: Sequence[Sequence[Sequence[float]]],
) -> np.ndarray:
    """Rasterize all COCO polygon annotations into one binary union mask."""

    canvas = Image.new("L", size, 0)
    draw = ImageDraw.Draw(canvas)
    for polygons in segmentations:
        for polygon in polygons:
            if len(polygon) < 6 or len(polygon) % 2:
                continue
            points = list(zip(polygon[0::2], polygon[1::2]))
            draw.polygon(points, fill=1)
    return np.asarray(canvas, dtype=np.uint8)


def align_image_and_mask(
    image: Image.Image,
    mask: np.ndarray,
    resize_edge: int = 1024,
    input_size: int = 518,
) -> Tuple[Image.Image, np.ndarray]:
    """Apply identical long-edge resize, short-edge resize, and center crop."""

    image = image.convert("RGB")
    if mask.shape != (image.height, image.width):
        raise ValueError(
            f"Mask shape {mask.shape} does not match image size "
            f"{(image.height, image.width)}"
        )

    resized_image = resize_long_edge(image, resize_edge)
    resized_mask = Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
        resized_image.size,
        Image.Resampling.NEAREST,
    )

    width, height = resized_image.size
    scale = input_size / float(min(width, height))
    short_resized = (
        max(input_size, int(round(width * scale))),
        max(input_size, int(round(height * scale))),
    )
    resized_image = resized_image.resize(short_resized, Image.Resampling.BICUBIC)
    resized_mask = resized_mask.resize(short_resized, Image.Resampling.NEAREST)

    left = (short_resized[0] - input_size) // 2
    top = (short_resized[1] - input_size) // 2
    box = (left, top, left + input_size, top + input_size)
    aligned_image = resized_image.crop(box)
    aligned_mask = (np.asarray(resized_mask.crop(box), dtype=np.uint8) > 0).astype(
        np.uint8
    )
    return aligned_image, aligned_mask


def patch_occupancy(mask: np.ndarray, patch_size: int = 14) -> np.ndarray:
    """Average a binary mask within non-overlapping ViT patches."""

    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2-D mask; got shape {mask.shape}")
    height, width = mask.shape
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"Mask shape {mask.shape} is not divisible by patch size {patch_size}"
        )
    return mask.reshape(
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    ).mean(axis=(1, 3))


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero vector")
    return (vector / norm).astype(np.float32)


def fit_weighted_prototypes(
    tokens: np.ndarray,
    occupancy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit L2-normalized domain and background token prototypes."""

    tokens = np.asarray(tokens, dtype=np.float64)
    occupancy = np.asarray(occupancy, dtype=np.float64).reshape(-1)
    if tokens.ndim < 2:
        raise ValueError(f"Expected token matrix; got shape {tokens.shape}")
    tokens = tokens.reshape(-1, tokens.shape[-1])
    if len(tokens) != len(occupancy):
        raise ValueError("Token and occupancy counts differ")
    if np.any((occupancy < 0.0) | (occupancy > 1.0)):
        raise ValueError("Occupancy must lie in [0, 1]")
    domain_weight = float(occupancy.sum())
    background = 1.0 - occupancy
    background_weight = float(background.sum())
    if domain_weight <= 0.0 or background_weight <= 0.0:
        raise ValueError("Both domain and background patches are required")
    domain_prototype = (tokens * occupancy[:, None]).sum(axis=0) / domain_weight
    background_prototype = (tokens * background[:, None]).sum(axis=0) / background_weight
    return _normalize(domain_prototype), _normalize(background_prototype)


def prototype_response(
    tokens: np.ndarray,
    domain_prototype: np.ndarray,
    background_prototype: np.ndarray,
) -> np.ndarray:
    """Return cosine(domain) minus cosine(background) for every patch token."""

    tokens = np.asarray(tokens, dtype=np.float64)
    original_shape = tokens.shape[:-1]
    rows = tokens.reshape(-1, tokens.shape[-1])
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    normalized = rows / np.maximum(norms, 1e-12)
    domain = _normalize(domain_prototype).astype(np.float64)
    background = _normalize(background_prototype).astype(np.float64)
    scores = normalized @ domain - normalized @ background
    return scores.reshape(original_shape).astype(np.float32)


def binary_overlap_metrics(
    scores: np.ndarray,
    occupancy: np.ndarray,
    threshold: float,
    occupancy_threshold: float = 0.5,
) -> Dict[str, float]:
    truth = np.asarray(occupancy).reshape(-1) >= occupancy_threshold
    prediction = np.asarray(scores).reshape(-1) > threshold
    true_positive = int(np.logical_and(truth, prediction).sum())
    false_positive = int(np.logical_and(~truth, prediction).sum())
    false_negative = int(np.logical_and(truth, ~prediction).sum())
    dice_denominator = 2 * true_positive + false_positive + false_negative
    iou_denominator = true_positive + false_positive + false_negative
    return {
        "dice": 2.0 * true_positive / dice_denominator if dice_denominator else 1.0,
        "iou": true_positive / iou_denominator if iou_denominator else 1.0,
        "response_fraction": float(prediction.mean()),
        "truth_fraction": float(truth.mean()),
    }


def select_threshold(
    scores: np.ndarray,
    occupancy: np.ndarray,
    occupancy_threshold: float = 0.5,
    quantile_steps: int = 256,
) -> Tuple[float, Dict[str, float]]:
    """Select the validation threshold with maximum patch-level Dice."""

    flat_scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not len(flat_scores):
        raise ValueError("Cannot select a threshold from no scores")
    candidates = np.unique(
        np.quantile(flat_scores, np.linspace(0.0, 1.0, quantile_steps + 1))
    )
    if len(candidates) > 1:
        candidates = (candidates[:-1] + candidates[1:]) / 2.0
    best_threshold = float(candidates[0])
    best_metrics = binary_overlap_metrics(
        flat_scores,
        occupancy,
        best_threshold,
        occupancy_threshold,
    )
    for candidate in candidates[1:]:
        metrics = binary_overlap_metrics(
            flat_scores,
            occupancy,
            float(candidate),
            occupancy_threshold,
        )
        if (metrics["dice"], metrics["iou"], -float(candidate)) > (
            best_metrics["dice"],
            best_metrics["iou"],
            -best_threshold,
        ):
            best_threshold = float(candidate)
            best_metrics = metrics
    return best_threshold, best_metrics


def split_patch_tokens(features, num_prefix_tokens: int, grid_size: Tuple[int, int]):
    """Remove prefix tokens and validate the expected spatial grid."""

    if len(features.shape) != 3:
        raise ValueError(f"Expected [batch, token, channel]; got {features.shape}")
    expected = int(grid_size[0] * grid_size[1])
    patches = features[:, num_prefix_tokens:]
    if patches.shape[1] != expected:
        raise ValueError(
            f"Expected {expected} patch tokens after {num_prefix_tokens} prefixes; "
            f"got {patches.shape[1]}"
        )
    return patches


def within_group_center(values: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    """Remove each magnification group's mean from a numeric vector."""

    values = np.asarray(values, dtype=np.float64)
    if len(values) != len(labels):
        raise ValueError("Value and label counts differ")
    centered = np.empty_like(values)
    label_array = np.asarray(labels, dtype=object)
    for label in dict.fromkeys(labels):
        mask = label_array == label
        centered[mask] = values[mask] - float(values[mask].mean())
    return centered


def correlation_summary(
    truth_fraction: np.ndarray,
    response_fraction: np.ndarray,
    labels: Sequence[str],
) -> Dict[str, float]:
    """Compute overall rank/linear and within-magnification correlations."""

    from scipy.stats import pearsonr, spearmanr

    truth = np.asarray(truth_fraction, dtype=np.float64)
    response = np.asarray(response_fraction, dtype=np.float64)
    if len(truth) != len(response) or len(truth) != len(labels):
        raise ValueError("Truth, response, and label counts differ")
    if len(truth) < 3:
        raise ValueError("At least three images are required for correlation")
    pearson = pearsonr(truth, response)
    spearman = spearmanr(truth, response)
    centered_truth = within_group_center(truth, labels)
    centered_response = within_group_center(response, labels)
    within = pearsonr(centered_truth, centered_response)
    return {
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_r": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "within_group_pearson_r": float(within.statistic),
        "within_group_pearson_p": float(within.pvalue),
    }


class DinoPatchExtractor:
    """Lazy frozen DINOv2-S/14 patch-token extractor."""

    def __init__(self, model_name: str, input_size: int, patch_size: int) -> None:
        self.model_name = model_name
        self.input_size = input_size
        self.patch_size = patch_size
        self._model = None
        self._torch = None
        self._device = None
        self._mean = None
        self._std = None
        self.grid_size = (input_size // patch_size, input_size // patch_size)
        self.embedding_dim: Optional[int] = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import timm
        import torch

        model = timm.create_model(self.model_name, pretrained=True, num_classes=0)
        config = timm.data.resolve_model_data_config(model)
        if tuple(config["input_size"][-2:]) != (self.input_size, self.input_size):
            raise ValueError(
                f"Model expects {config['input_size']}; configured input is {self.input_size}"
            )
        if tuple(model.patch_embed.patch_size) != (self.patch_size, self.patch_size):
            raise ValueError(
                f"Model patch size is {model.patch_embed.patch_size}; "
                f"configured patch is {self.patch_size}"
            )
        self.grid_size = tuple(int(value) for value in model.patch_embed.grid_size)
        self.embedding_dim = int(model.embed_dim)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model.to(self._device).eval()
        self._torch = torch
        self._mean = torch.tensor(config["mean"], device=self._device).view(1, 3, 1, 1)
        self._std = torch.tensor(config["std"], device=self._device).view(1, 3, 1, 1)

    def extract(self, aligned_image: Image.Image) -> np.ndarray:
        self._ensure_model()
        assert self._torch is not None
        assert self._device is not None
        assert self._mean is not None
        assert self._std is not None
        if aligned_image.size != (self.input_size, self.input_size):
            raise ValueError(f"Expected aligned image size {self.input_size}; got {aligned_image.size}")
        array = np.asarray(aligned_image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = self._torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(self._device)
        tensor = (tensor - self._mean) / self._std
        with self._torch.inference_mode():
            features = self._model.forward_features(tensor)
            patches = split_patch_tokens(
                features,
                num_prefix_tokens=int(self._model.num_prefix_tokens),
                grid_size=self.grid_size,
            )
        result = patches[0].float().cpu().numpy()
        expected = self.grid_size[0] * self.grid_size[1]
        if result.shape != (expected, self.embedding_dim):
            raise ValueError(f"Unexpected patch-token shape: {result.shape}")
        return result.astype(np.float32)


def _cache_metadata(
    data_root: Path,
    split: str,
    records: Sequence[CocoImageRecord],
    model_name: str,
    resize_edge: int,
    input_size: int,
    patch_size: int,
) -> Dict[str, object]:
    annotation = data_root / "annotations" / f"instances_{split}.json"
    return {
        "split": split,
        "model_name": model_name,
        "resize_edge": resize_edge,
        "input_size": input_size,
        "patch_size": patch_size,
        "annotation": {
            "path": str(annotation.resolve()),
            "size": annotation.stat().st_size,
            "mtime_ns": annotation.stat().st_mtime_ns,
        },
        "images": [
            {
                "image_id": record.image_id,
                "file_name": record.file_name,
                "path": str(record.image_path.resolve()),
                "size": record.image_path.stat().st_size,
                "mtime_ns": record.image_path.stat().st_mtime_ns,
                "label": record.label,
            }
            for record in records
        ],
    }


def load_or_extract_split(
    data_root: Path,
    split: str,
    cache_root: Path,
    extractor: DinoPatchExtractor,
    resize_edge: int,
    input_size: int,
    patch_size: int,
    reuse_cache: bool,
) -> SplitPatchData:
    records = load_coco_records(data_root, split)
    metadata = _cache_metadata(
        data_root,
        split,
        records,
        extractor.model_name,
        resize_edge,
        input_size,
        patch_size,
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    npz_path = cache_root / f"{split}.npz"
    json_path = cache_root / f"{split}.json"
    if reuse_cache and npz_path.is_file() and json_path.is_file():
        cached_metadata = json.loads(json_path.read_text(encoding="utf-8"))
        if cached_metadata == metadata:
            with np.load(npz_path) as cached:
                tokens = cached["tokens"].astype(np.float32)
                occupancy = cached["occupancy"].astype(np.float32)
                pixel_fraction = cached["pixel_fraction"].astype(np.float32)
            print(
                f"cache_hit split={split} tokens={tokens.shape} "
                f"occupancy={occupancy.shape}",
                flush=True,
            )
            return SplitPatchData(records, tokens, occupancy, pixel_fraction)

    token_rows: list[np.ndarray] = []
    occupancy_rows: list[np.ndarray] = []
    pixel_fractions: list[float] = []
    started = time.perf_counter()
    for index, record in enumerate(records, start=1):
        with Image.open(record.image_path) as source:
            image = source.convert("RGB")
        if image.size != (record.width, record.height):
            raise ValueError(
                f"COCO size mismatch for {record.image_path}: "
                f"json={(record.width, record.height)} image={image.size}"
            )
        mask = rasterize_polygon_union(image.size, record.segmentations)
        aligned_image, aligned_mask = align_image_and_mask(
            image,
            mask,
            resize_edge=resize_edge,
            input_size=input_size,
        )
        token_rows.append(extractor.extract(aligned_image))
        occupancy_rows.append(patch_occupancy(aligned_mask, patch_size).reshape(-1))
        pixel_fractions.append(float(aligned_mask.mean()))
        print(
            f"patch_tokens split={split} index={index}/{len(records)} "
            f"image={record.file_name}",
            flush=True,
        )

    tokens = np.stack(token_rows).astype(np.float32)
    occupancy = np.stack(occupancy_rows).astype(np.float32)
    pixel_fraction = np.asarray(pixel_fractions, dtype=np.float32)
    np.savez_compressed(
        npz_path,
        tokens=tokens.astype(np.float16),
        occupancy=occupancy.astype(np.float16),
        pixel_fraction=pixel_fraction,
    )
    json_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"cache_saved split={split} tokens={tokens.shape} "
        f"elapsed_seconds={time.perf_counter() - started:.2f}",
        flush=True,
    )
    return SplitPatchData(records, tokens, occupancy, pixel_fraction)


def _safe_number(value: float) -> Optional[float]:
    value = float(value)
    return value if np.isfinite(value) else None


def _format_number(value: Optional[float]) -> str:
    return "NA" if value is None else f"{value:.3f}"


def summarize_split(
    data: SplitPatchData,
    scores: np.ndarray,
    threshold: float,
) -> Tuple[list[Dict[str, object]], Dict[str, object]]:
    rows: list[Dict[str, object]] = []
    for index, record in enumerate(data.records):
        overlap = binary_overlap_metrics(scores[index], data.occupancy[index], threshold)
        patch_truth = data.occupancy[index] >= 0.5
        inside = scores[index][patch_truth]
        outside = scores[index][~patch_truth]
        rows.append(
            {
                "split": record.split,
                "image": record.file_name,
                "magnification": record.label,
                "gt_pixel_fraction": float(data.pixel_fraction[index]),
                "gt_patch_fraction": float(patch_truth.mean()),
                "dino_response_fraction": overlap["response_fraction"],
                "dice": overlap["dice"],
                "iou": overlap["iou"],
                "mean_inside_response": float(inside.mean()) if len(inside) else None,
                "mean_outside_response": float(outside.mean()) if len(outside) else None,
            }
        )

    labels = [record.label for record in data.records]
    truth_fraction = np.asarray([float(row["gt_pixel_fraction"]) for row in rows])
    response_fraction = np.asarray([float(row["dino_response_fraction"]) for row in rows])
    correlations = correlation_summary(truth_fraction, response_fraction, labels)
    patch_truth = data.occupancy.reshape(-1) >= 0.5
    flat_scores = scores.reshape(-1)
    metrics: Dict[str, object] = {
        "sample_count": len(rows),
        "patch_count": int(len(flat_scores)),
        "threshold": float(threshold),
        "patch_auroc": float(roc_auc_score(patch_truth.astype(np.uint8), flat_scores)),
        "global_dice": binary_overlap_metrics(
            flat_scores, data.occupancy.reshape(-1), threshold
        )["dice"],
        "global_iou": binary_overlap_metrics(
            flat_scores, data.occupancy.reshape(-1), threshold
        )["iou"],
        "mean_image_dice": float(np.mean([float(row["dice"]) for row in rows])),
        "mean_image_iou": float(np.mean([float(row["iou"]) for row in rows])),
        "mean_inside_response": float(flat_scores[patch_truth].mean()),
        "mean_outside_response": float(flat_scores[~patch_truth].mean()),
        "correlation": {key: _safe_number(value) for key, value in correlations.items()},
    }
    return rows, metrics


def _write_rows(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _aligned_record(record: CocoImageRecord, resize_edge: int, input_size: int):
    with Image.open(record.image_path) as source:
        image = source.convert("RGB")
    mask = rasterize_polygon_union(image.size, record.segmentations)
    return align_image_and_mask(image, mask, resize_edge=resize_edge, input_size=input_size)


def plot_representative_panels(
    output_dir: Path,
    data: SplitPatchData,
    scores: np.ndarray,
    resize_edge: int,
    input_size: int,
    grid_size: Tuple[int, int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    low, high = np.quantile(scores, [0.02, 0.98])
    for label in MAGNIFICATION_LABELS:
        index = next(i for i, record in enumerate(data.records) if record.label == label)
        record = data.records[index]
        image, mask = _aligned_record(record, resize_edge, input_size)
        score_grid = scores[index].reshape(grid_size)
        score_image = Image.fromarray(score_grid.astype(np.float32), mode="F").resize(
            (input_size, input_size), Image.Resampling.BICUBIC
        )
        dense_scores = np.asarray(score_image, dtype=np.float32)
        figure, axes = plt.subplots(1, 4, figsize=(15.5, 4.1))
        axes[0].imshow(image)
        axes[0].set_title("Aligned image")
        axes[1].imshow(mask, cmap="YlOrBr", vmin=0, vmax=1)
        axes[1].set_title(f"COCO union mask\narea={mask.mean():.3f}")
        heat = axes[2].imshow(dense_scores, cmap="YlOrBr", vmin=low, vmax=high)
        axes[2].set_title("DINOv2 patch response")
        axes[3].imshow(image)
        axes[3].imshow(dense_scores, cmap="YlOrBr", alpha=0.52, vmin=low, vmax=high)
        axes[3].contour(mask, levels=[0.5], colors=["#7f2704"], linewidths=0.8)
        axes[3].set_title("Response overlay")
        for axis in axes:
            axis.axis("off")
        figure.colorbar(heat, ax=axes[2:], fraction=0.025, pad=0.02, label="Prototype response")
        figure.suptitle(f"{label} — {record.file_name}")
        figure.tight_layout()
        stem = output_dir / f"representative_{label.replace('.', 'p')}"
        figure.savefig(stem.with_suffix(".png"), dpi=260, bbox_inches="tight")
        figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
        plt.close(figure)


def plot_area_scatter(
    output_stem: Path,
    rows: Sequence[Dict[str, object]],
    metrics: Dict[str, object],
) -> None:
    colors = {
        "2.5x": "#d95f02",
        "5x": "#e6ab02",
        "20x": "#1b9e77",
        "50x": "#7570b3",
        "100x": "#e7298a",
    }
    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    for label in MAGNIFICATION_LABELS:
        selected = [row for row in rows if row["magnification"] == label]
        axis.scatter(
            [float(row["gt_pixel_fraction"]) for row in selected],
            [float(row["dino_response_fraction"]) for row in selected],
            s=58,
            label=label,
            color=colors[label],
            edgecolor="white",
            linewidth=0.7,
        )
    x = np.asarray([float(row["gt_pixel_fraction"]) for row in rows])
    y = np.asarray([float(row["dino_response_fraction"]) for row in rows])
    if np.ptp(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        line_x = np.linspace(x.min(), x.max(), 100)
        axis.plot(line_x, slope * line_x + intercept, color="#4d4d4d", linewidth=1.4)
    correlation = metrics["correlation"]
    axis.text(
        0.03,
        0.97,
        f"Pearson r={_format_number(correlation['pearson_r'])}\n"
        f"Spearman ρ={_format_number(correlation['spearman_r'])}\n"
        f"Within-mag r={_format_number(correlation['within_group_pearson_r'])}",
        transform=axis.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )
    axis.set_xlabel("Ground-truth domain pixel fraction")
    axis.set_ylabel("DINOv2 high-response patch fraction")
    axis.set_title("Domain area versus DINOv2 response area (Test)")
    axis.legend(title="Magnification", frameon=False)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_stem.with_suffix(".png"), dpi=260, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def plot_response_distribution(
    output_stem: Path,
    scores: np.ndarray,
    occupancy: np.ndarray,
) -> None:
    truth = occupancy.reshape(-1) >= 0.5
    flat_scores = scores.reshape(-1)
    figure, axis = plt.subplots(figsize=(5.2, 4.8))
    boxes = axis.boxplot(
        [flat_scores[~truth], flat_scores[truth]],
        labels=["Background", "Domain"],
        showfliers=False,
        patch_artist=True,
    )
    for patch, color in zip(boxes["boxes"], ["#fee8c8", "#e6550d"]):
        patch.set_facecolor(color)
    axis.set_ylabel("Prototype response")
    axis.set_title("DINOv2 patch response by COCO region (Test)")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_stem.with_suffix(".png"), dpi=260, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def _version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_experiment(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = output_dir / "patch_cache"
    extractor = DinoPatchExtractor(args.model_name, args.input_size, args.patch_size)
    split_data = {
        split: load_or_extract_split(
            data_root,
            split,
            cache_root,
            extractor,
            args.resize_long_edge,
            args.input_size,
            args.patch_size,
            args.reuse_cache,
        )
        for split in ("train", "val", "test")
    }

    domain, background = fit_weighted_prototypes(
        split_data["train"].tokens,
        split_data["train"].occupancy,
    )
    responses = {
        split: prototype_response(data.tokens, domain, background)
        for split, data in split_data.items()
    }
    threshold, validation_threshold_metrics = select_threshold(
        responses["val"],
        split_data["val"].occupancy,
    )

    all_rows: list[Dict[str, object]] = []
    all_metrics: Dict[str, object] = {}
    for split in ("train", "val", "test"):
        rows, metrics = summarize_split(split_data[split], responses[split], threshold)
        all_rows.extend(rows)
        all_metrics[split] = metrics
        _write_rows(output_dir / f"per_image_{split}.csv", rows)

    np.savez_compressed(
        output_dir / "response_artifacts.npz",
        domain_prototype=domain,
        background_prototype=background,
        threshold=np.asarray([threshold], dtype=np.float32),
        train_scores=responses["train"].astype(np.float16),
        val_scores=responses["val"].astype(np.float16),
        test_scores=responses["test"].astype(np.float16),
    )
    _write_rows(output_dir / "per_image_all.csv", all_rows)
    metrics_payload = {
        "model_name": args.model_name,
        "input_size": args.input_size,
        "patch_size": args.patch_size,
        "patch_grid": list(extractor.grid_size),
        "embedding_dim": extractor.embedding_dim,
        "threshold": threshold,
        "validation_threshold_metrics": validation_threshold_metrics,
        "splits": all_metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    plot_representative_panels(
        output_dir / "representatives",
        split_data["test"],
        responses["test"],
        args.resize_long_edge,
        args.input_size,
        extractor.grid_size,
    )
    plot_area_scatter(
        output_dir / "test_area_correlation",
        [row for row in all_rows if row["split"] == "test"],
        all_metrics["test"],
    )
    plot_response_distribution(
        output_dir / "test_response_distribution",
        responses["test"],
        split_data["test"].occupancy,
    )

    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    manifest = {
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "git_commit": git_commit,
        "arguments": vars(args) | {
            "data_root": str(args.data_root),
            "output_dir": str(args.output_dir),
        },
        "versions": {
            name: _version(name)
            for name in ("numpy", "pillow", "matplotlib", "scikit-learn", "scipy", "timm", "torch")
        },
        "split_counts": {split: len(data.records) for split, data in split_data.items()},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    test = all_metrics["test"]
    print(
        "result "
        f"test_patch_auroc={test['patch_auroc']:.4f} "
        f"test_dice={test['global_dice']:.4f} "
        f"test_pearson={test['correlation']['pearson_r']} "
        f"test_spearman={test['correlation']['spearman_r']} "
        f"test_within_mag={test['correlation']['within_group_pearson_r']}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Relate frozen DINOv2 patch response to COCO domain-pixel fraction."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="vit_small_patch14_dinov2")
    parser.add_argument("--resize-long-edge", type=int, default=1024)
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-cache", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    np.random.seed(args.seed)
    run_experiment(args)


if __name__ == "__main__":
    main()
