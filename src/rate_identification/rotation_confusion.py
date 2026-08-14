from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import IMAGE_SUFFIXES, load_rgb_image, make_inference_views
from .features import build_extractor
from .split_evaluation import (
    MAGNIFICATION_LABELS,
    WINDOW_LABELS,
    EmbeddingScaleClassifier,
    EvaluationResult,
    evaluate_embeddings,
    parse_rotation_label,
)


METHOD_NAMES = ["raw_pca", "handcrafted_pca", "dino_no_pca", "dino_pca"]


def collect_split_images(
    image_dir: Path,
    require_per_label: int,
) -> Tuple[List[Path], List[str]]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image split directory does not exist: {image_dir}")
    grouped: Dict[str, List[Path]] = {label: [] for label in MAGNIFICATION_LABELS}
    for path in sorted(image_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        grouped[parse_rotation_label(path)].append(path.resolve())

    counts = {label: len(paths) for label, paths in grouped.items()}
    if any(count != require_per_label for count in counts.values()):
        raise ValueError(
            f"Expected {require_per_label} images per label under {image_dir}; "
            f"found {counts}"
        )

    paths: List[Path] = []
    labels: List[str] = []
    for label in MAGNIFICATION_LABELS:
        label_paths = sorted(grouped[label])
        paths.extend(label_paths)
        labels.extend([label] * len(label_paths))
    return paths, labels


def write_evaluation_artifacts(
    output_dir: Path,
    result: EvaluationResult,
    method: str,
    split: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_predictions_csv(output_dir / "predictions.csv", result.rows)
    metrics = {
        "method": method,
        "split": split,
        "sample_count": result.sample_count,
        "accuracy": result.accuracy,
        "balanced_accuracy": result.balanced_accuracy,
        "macro_f1": result.macro_f1,
        "window_accuracy": result.window_accuracy,
        "labels": result.labels,
        "window_labels": result.window_labels,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_matrix_csv(
        output_dir / "confusion_counts.csv",
        result.labels,
        result.confusion_counts,
        integer=True,
    )
    _write_matrix_csv(
        output_dir / "confusion_normalized.csv",
        result.labels,
        result.confusion_normalized,
        integer=False,
    )
    _write_matrix_csv(
        output_dir / "window_confusion_counts.csv",
        result.window_labels,
        result.window_confusion_counts,
        integer=True,
    )
    _write_matrix_csv(
        output_dir / "window_confusion_normalized.csv",
        result.window_labels,
        result.window_confusion_normalized,
        integer=False,
    )
    _plot_confusion(
        output_dir / "confusion_matrix",
        result.labels,
        result.confusion_counts,
        result.confusion_normalized,
        title=f"{method}: magnification confusion ({split})",
    )
    _plot_confusion(
        output_dir / "window_confusion_matrix",
        result.window_labels,
        result.window_confusion_counts,
        result.window_confusion_normalized,
        title=f"{method}: window-class confusion ({split})",
    )


def _write_predictions_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "image",
        "true_label",
        "predicted_label",
        "predicted_scale",
        "true_window_class",
        "predicted_window_class",
        "correct",
        "window_correct",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload["predicted_scale"] = f"{float(payload['predicted_scale']):.8f}"
            writer.writerow(payload)


def _write_matrix_csv(
    path: Path,
    labels: Sequence[str],
    matrix: np.ndarray,
    integer: bool,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *labels])
        for label, row in zip(labels, np.asarray(matrix)):
            values = [str(int(value)) for value in row] if integer else [f"{float(value):.8f}" for value in row]
            writer.writerow([label, *values])


def _plot_confusion(
    output_stem: Path,
    labels: Sequence[str],
    counts: np.ndarray,
    normalized: np.ndarray,
    title: str,
) -> None:
    figure_size = max(5.2, 0.9 * len(labels) + 2.2)
    fig, ax = plt.subplots(figsize=(figure_size, figure_size - 0.4))
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    for row_index in range(len(labels)):
        for column_index in range(len(labels)):
            percentage = float(normalized[row_index, column_index])
            count = int(counts[row_index, column_index])
            color = "white" if percentage >= 0.5 else "black"
            ax.text(
                column_index,
                row_index,
                f"{count}\n{percentage:.0%}",
                ha="center",
                va="center",
                fontsize=9,
                color=color,
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized fraction")
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _method_spec(method: str, pca_components: int) -> Tuple[str, Optional[int]]:
    mapping = {
        "raw_pca": ("raw_pixels", pca_components),
        "handcrafted_pca": ("robust_handcrafted", pca_components),
        "dino_no_pca": ("dinov2_timm", None),
        "dino_pca": ("dinov2_timm", pca_components),
    }
    try:
        return mapping[method]
    except KeyError as exc:
        raise ValueError(f"Unsupported method: {method}") from exc


def _embedding_cache_metadata(
    extractor_name: str,
    image_paths: Sequence[Path],
    resize_long_edge: int,
    use_multiview: bool,
) -> Dict[str, object]:
    return {
        "extractor": extractor_name,
        "resize_long_edge": resize_long_edge,
        "use_multiview": use_multiview,
        "images": [
            {
                "path": str(path),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in image_paths
        ],
    }


def load_or_extract_embeddings(
    extractor_name: str,
    image_paths: Sequence[Path],
    split: str,
    cache_root: Path,
    resize_long_edge: int,
    use_multiview: bool,
    reuse_cache: bool,
) -> np.ndarray:
    extractor_cache = cache_root / extractor_name
    extractor_cache.mkdir(parents=True, exist_ok=True)
    array_path = extractor_cache / f"{split}.npz"
    metadata_path = extractor_cache / f"{split}.json"
    expected_metadata = _embedding_cache_metadata(
        extractor_name,
        image_paths,
        resize_long_edge,
        use_multiview,
    )

    if reuse_cache and array_path.exists() and metadata_path.exists():
        cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if cached_metadata == expected_metadata:
            with np.load(array_path) as payload:
                embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
            if embeddings.shape[0] != len(image_paths):
                raise ValueError(f"Cached row count mismatch: {array_path}")
            print(f"cache_hit extractor={extractor_name} split={split} shape={embeddings.shape}")
            return embeddings
        print(f"cache_miss_metadata extractor={extractor_name} split={split}")

    extractor = build_extractor(extractor_name, resize_edge=resize_long_edge)
    rows = []
    start = time.monotonic()
    for index, path in enumerate(image_paths, start=1):
        if use_multiview:
            image = load_rgb_image(path)
            view_rows = [extractor.extract_image(view) for view in make_inference_views(image)]
            embedding = np.mean(np.vstack(view_rows), axis=0)
        else:
            embedding = extractor.extract(path)
        rows.append(np.asarray(embedding, dtype=np.float32))
        print(
            f"embedding extractor={extractor_name} split={split} "
            f"index={index}/{len(image_paths)} image={path.name}",
            flush=True,
        )
    embeddings = np.vstack(rows).astype(np.float32)
    np.savez_compressed(array_path, embeddings=embeddings)
    metadata_path.write_text(
        json.dumps(expected_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"cache_saved extractor={extractor_name} split={split} "
        f"shape={embeddings.shape} elapsed_seconds={time.monotonic() - start:.2f}"
    )
    return embeddings


def run_experiment(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = output_dir / "embedding_cache"

    split_data: Dict[str, Tuple[List[Path], List[str]]] = {}
    split_requirements = {"train": 12, "val": 4, "test": 4}
    for split in ["train", *args.splits]:
        split_data[split] = collect_split_images(
            args.data_root / "images" / split,
            require_per_label=split_requirements[split],
        )
        print(f"split={split} images={len(split_data[split][0])}")

    extractor_names = sorted({_method_spec(method, args.pca_components)[0] for method in args.methods})
    embeddings: Dict[Tuple[str, str], np.ndarray] = {}
    for extractor_name in extractor_names:
        for split in ["train", *args.splits]:
            paths, _ = split_data[split]
            embeddings[(extractor_name, split)] = load_or_extract_embeddings(
                extractor_name=extractor_name,
                image_paths=paths,
                split=split,
                cache_root=cache_root,
                resize_long_edge=args.resize_long_edge,
                use_multiview=not args.disable_multiview,
                reuse_cache=args.reuse_cache,
            )

    summary_rows: List[Dict[str, object]] = []
    train_paths, train_labels = split_data["train"]
    del train_paths
    for method in args.methods:
        extractor_name, pca_components = _method_spec(method, args.pca_components)
        classifier = EmbeddingScaleClassifier(
            pca_components=pca_components,
            knn_neighbors=args.knn_neighbors,
            cluster_count=args.cluster_count,
            random_state=args.seed,
        ).fit(embeddings[(extractor_name, "train")], train_labels)
        effective_pca = int(classifier.pca.n_components_) if classifier.pca is not None else None
        for split in args.splits:
            paths, labels = split_data[split]
            result = evaluate_embeddings(
                classifier,
                embeddings[(extractor_name, split)],
                labels,
                paths,
            )
            method_output = output_dir / method / split
            write_evaluation_artifacts(method_output, result, method=method, split=split)
            summary_rows.append(
                {
                    "method": method,
                    "split": split,
                    "extractor": extractor_name,
                    "requested_pca_components": pca_components,
                    "effective_pca_components": effective_pca,
                    "sample_count": result.sample_count,
                    "accuracy": result.accuracy,
                    "balanced_accuracy": result.balanced_accuracy,
                    "macro_f1": result.macro_f1,
                    "window_accuracy": result.window_accuracy,
                }
            )
            print(
                f"result method={method} split={split} accuracy={result.accuracy:.4f} "
                f"macro_f1={result.macro_f1:.4f} window_accuracy={result.window_accuracy:.4f}"
            )

    _write_summary_csv(output_dir / "summary.csv", summary_rows)
    _write_manifest(output_dir / "manifest.json", args, split_data, summary_rows)


def _write_summary_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No summary rows were generated.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(
    path: Path,
    args: argparse.Namespace,
    split_data: Dict[str, Tuple[List[Path], List[str]]],
    summary_rows: Sequence[Dict[str, object]],
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    packages = {}
    for package in ["numpy", "scikit-learn", "Pillow", "matplotlib", "timm", "torch"]:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    payload = {
        "git_commit": git_commit,
        "command": sys.argv,
        "parameters": {
            "data_root": str(args.data_root.resolve()),
            "methods": list(args.methods),
            "splits": list(args.splits),
            "pca_components": args.pca_components,
            "knn_neighbors": args.knn_neighbors,
            "cluster_count": args.cluster_count,
            "resize_long_edge": args.resize_long_edge,
            "use_multiview": not args.disable_multiview,
            "seed": args.seed,
        },
        "packages": packages,
        "images": {
            split: [str(image) for image in paths]
            for split, (paths, _) in split_data.items()
        },
        "summary": list(summary_rows),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate rotation-scale recognition with fixed Train/Validation/Test splits."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHOD_NAMES, default=METHOD_NAMES)
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument("--knn-neighbors", type=int, default=5)
    parser.add_argument("--cluster-count", type=int, default=6)
    parser.add_argument("--resize-long-edge", type=int, default=1024)
    parser.add_argument("--splits", nargs="+", choices=["val", "test"], default=["val", "test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-multiview", action="store_true")
    parser.add_argument("--reuse-cache", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
