from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from rate_identification.features import RawPixelExtractor
from rate_identification.split_evaluation import (
    EvaluationResult,
    MAGNIFICATION_LABELS,
    EmbeddingScaleClassifier,
    evaluate_embeddings,
    label_to_window_class,
    parse_rotation_label,
)
from rate_identification.rotation_confusion import (
    collect_split_images,
    write_evaluation_artifacts,
)


class RawPixelExtractorTests(unittest.TestCase):
    def test_extracts_normalized_64_square_grayscale_vector(self) -> None:
        x = np.linspace(0, 255, 128, dtype=np.uint8)
        row = np.stack([x, np.flip(x), x], axis=1)
        image_array = np.repeat(row[None, :, :], 64, axis=0)
        image = Image.fromarray(image_array, mode="RGB")

        feature = RawPixelExtractor(resize_edge=1024).extract_image(image)

        self.assertEqual(feature.shape, (4096,))
        self.assertEqual(feature.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(feature)), 1.0, places=5)


class SplitEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        train_rows = []
        train_labels = []
        test_rows = []
        for label_index, label in enumerate(MAGNIFICATION_LABELS, start=1):
            center = float(label_index) * 10.0
            for offset in (-0.2, 0.0, 0.2):
                train_rows.append([center + offset, 0.1 * label_index])
                train_labels.append(label)
            test_rows.append([center + 0.05, 0.1 * label_index])
        self.train_embeddings = np.asarray(train_rows, dtype=np.float32)
        self.train_labels = train_labels
        self.test_embeddings = np.asarray(test_rows, dtype=np.float32)
        self.test_labels = list(MAGNIFICATION_LABELS)
        self.test_images = [f"{label.replace('.', 'p')}_00016.png" for label in MAGNIFICATION_LABELS]

    def test_parses_rotation_labels_and_window_classes(self) -> None:
        self.assertEqual(parse_rotation_label("2p5x_00016.png"), "2.5x")
        self.assertEqual(parse_rotation_label("100x_00019.png"), "100x")
        self.assertEqual(label_to_window_class("20x"), "2048")
        self.assertEqual(label_to_window_class("50x"), "noSW")

    def test_fit_uses_train_rows_and_builds_both_confusion_matrices(self) -> None:
        classifier = EmbeddingScaleClassifier(
            pca_components=1,
            knn_neighbors=5,
            cluster_count=6,
        ).fit(self.train_embeddings, self.train_labels)

        result = evaluate_embeddings(
            classifier,
            self.test_embeddings,
            self.test_labels,
            self.test_images,
        )

        self.assertEqual(result.sample_count, 5)
        self.assertEqual(result.labels, MAGNIFICATION_LABELS)
        self.assertEqual(int(result.confusion_counts.sum()), 5)
        self.assertEqual(int(result.window_confusion_counts.sum()), 5)
        self.assertEqual(len(result.rows), 5)
        self.assertNotIn("test_embeddings", classifier.__dict__)

    def test_no_pca_path_predicts_one_label_per_test_row(self) -> None:
        classifier = EmbeddingScaleClassifier(
            pca_components=None,
            knn_neighbors=5,
            cluster_count=6,
        ).fit(self.train_embeddings, self.train_labels)

        predictions, scales = classifier.predict(self.test_embeddings)

        self.assertIsNone(classifier.pca)
        self.assertEqual(len(predictions), len(self.test_labels))
        self.assertEqual(scales.shape, (len(self.test_labels),))


class ArtifactWriterTests(unittest.TestCase):
    def test_collect_split_images_uses_fixed_label_order(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label in ("100x", "2p5x", "50x", "5x", "20x"):
                for index in (17, 16):
                    (root / f"{label}_{index:05d}.png").write_bytes(b"not-read")

            paths, labels = collect_split_images(root, require_per_label=2)

            self.assertEqual(labels[:2], ["2.5x", "2.5x"])
            self.assertEqual(labels[-2:], ["100x", "100x"])
            self.assertEqual(len(paths), 10)

    def test_collect_split_images_rejects_incomplete_label(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label in ("2p5x", "5x", "20x", "50x", "100x"):
                (root / f"{label}_00016.png").write_bytes(b"not-read")
            (root / "2p5x_00017.png").write_bytes(b"not-read")

            with self.assertRaisesRegex(ValueError, "Expected 1 images per label"):
                collect_split_images(root, require_per_label=1)

    def test_writes_numeric_and_visual_confusion_artifacts(self) -> None:
        count_matrix = np.eye(5, dtype=np.int64)
        window_matrix = np.diag([1, 1, 1, 2]).astype(np.int64)
        result = EvaluationResult(
            sample_count=5,
            labels=list(MAGNIFICATION_LABELS),
            window_labels=["256", "512", "2048", "noSW"],
            accuracy=1.0,
            balanced_accuracy=1.0,
            macro_f1=1.0,
            window_accuracy=1.0,
            confusion_counts=count_matrix,
            confusion_normalized=count_matrix.astype(np.float64),
            window_confusion_counts=window_matrix,
            window_confusion_normalized=np.eye(4, dtype=np.float64),
            rows=[
                {
                    "image": "2p5x_00016.png",
                    "true_label": "2.5x",
                    "predicted_label": "2.5x",
                    "predicted_scale": 0.25,
                    "true_window_class": "256",
                    "predicted_window_class": "256",
                    "correct": True,
                    "window_correct": True,
                }
            ],
        )

        with TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            write_evaluation_artifacts(output_dir, result, method="dino_pca", split="test")

            expected = {
                "predictions.csv",
                "metrics.json",
                "confusion_counts.csv",
                "confusion_normalized.csv",
                "window_confusion_counts.csv",
                "window_confusion_normalized.csv",
                "confusion_matrix.png",
                "confusion_matrix.svg",
                "window_confusion_matrix.png",
                "window_confusion_matrix.svg",
            }
            self.assertEqual({path.name for path in output_dir.iterdir()}, expected)
            self.assertTrue(all((output_dir / name).stat().st_size > 0 for name in expected))


if __name__ == "__main__":
    unittest.main()
