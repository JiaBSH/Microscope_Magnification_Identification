from __future__ import annotations

import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from rate_identification.domain_response import (
    align_image_and_mask,
    correlation_summary,
    fit_weighted_prototypes,
    patch_occupancy,
    prototype_response,
    rasterize_polygon_union,
    load_coco_records,
    select_threshold,
    split_patch_tokens,
    within_group_center,
)


class MaskAlignmentTests(unittest.TestCase):
    def test_rasterize_union_does_not_double_count_overlap(self) -> None:
        segmentations = [
            [[0, 0, 9, 0, 9, 9, 0, 9]],
            [[5, 0, 14, 0, 14, 9, 5, 9]],
        ]

        mask = rasterize_polygon_union((20, 10), segmentations)

        self.assertEqual(mask.shape, (10, 20))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(int(mask.sum()), 150)

    def test_align_image_and_mask_returns_square_pair(self) -> None:
        image_array = np.zeros((40, 80, 3), dtype=np.uint8)
        image_array[:, 20:60, 0] = 255
        image = Image.fromarray(image_array, mode="RGB")
        mask = np.zeros((40, 80), dtype=np.uint8)
        mask[:, 20:60] = 1

        aligned_image, aligned_mask = align_image_and_mask(
            image,
            mask,
            resize_edge=100,
            input_size=20,
        )

        self.assertEqual(aligned_image.size, (20, 20))
        self.assertEqual(aligned_mask.shape, (20, 20))
        self.assertTrue(np.all(aligned_mask == 1))

    def test_patch_occupancy_preserves_fraction(self) -> None:
        mask = np.zeros((28, 28), dtype=np.uint8)
        mask[:14, :7] = 1
        mask[14:, 14:] = 1

        occupancy = patch_occupancy(mask, patch_size=14)

        np.testing.assert_allclose(occupancy, [[0.5, 0.0], [0.0, 1.0]])


class PrototypeResponseTests(unittest.TestCase):
    def test_response_is_higher_for_domain_tokens(self) -> None:
        tokens = np.asarray(
            [[1.0, 0.0], [0.8, 0.2], [-1.0, 0.0], [-0.8, -0.2]],
            dtype=np.float32,
        )
        occupancy = np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32)

        domain, background = fit_weighted_prototypes(tokens, occupancy)
        scores = prototype_response(tokens, domain, background)

        self.assertGreater(float(scores[:2].mean()), float(scores[2:].mean()))
        self.assertAlmostEqual(float(np.linalg.norm(domain)), 1.0, places=6)
        self.assertAlmostEqual(float(np.linalg.norm(background)), 1.0, places=6)

    def test_threshold_selection_maximizes_validation_dice(self) -> None:
        scores = np.asarray([-1.0, -0.5, 0.5, 1.0], dtype=np.float32)
        occupancy = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)

        threshold, metrics = select_threshold(scores, occupancy)

        self.assertGreater(threshold, -0.5)
        self.assertLess(threshold, 0.5)
        self.assertEqual(metrics["dice"], 1.0)
        self.assertEqual(metrics["iou"], 1.0)

    def test_split_patch_tokens_removes_prefix(self) -> None:
        features = np.arange(2 * 5 * 3).reshape(2, 5, 3)

        patches = split_patch_tokens(features, num_prefix_tokens=1, grid_size=(2, 2))

        self.assertEqual(patches.shape, (2, 4, 3))
        np.testing.assert_array_equal(patches[:, 0], features[:, 1])

    def test_within_group_center_removes_group_means(self) -> None:
        values = np.asarray([1.0, 3.0, 10.0, 14.0])
        labels = ["a", "a", "b", "b"]

        centered = within_group_center(values, labels)

        np.testing.assert_allclose(centered, [-1.0, 1.0, -2.0, 2.0])


class DatasetAndStatisticsTests(unittest.TestCase):
    def test_load_coco_records_groups_all_polygons_by_image(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "annotations").mkdir()
            image_dir = root / "images" / "train"
            image_dir.mkdir(parents=True)
            Image.new("RGB", (20, 10)).save(image_dir / "2p5x_00000.png")
            payload = {
                "images": [
                    {
                        "id": 7,
                        "file_name": "2p5x_00000.png",
                        "width": 20,
                        "height": 10,
                    }
                ],
                "annotations": [
                    {"image_id": 7, "segmentation": [[0, 0, 4, 0, 4, 4, 0, 4]]},
                    {"image_id": 7, "segmentation": [[10, 0, 14, 0, 14, 4, 10, 4]]},
                ],
            }
            (root / "annotations" / "instances_train.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )

            records = load_coco_records(root, "train")

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].label, "2.5x")
            self.assertEqual(records[0].image_path, image_dir / "2p5x_00000.png")
            self.assertEqual(len(records[0].segmentations), 2)

    def test_correlation_summary_reports_perfect_relationship(self) -> None:
        truth = np.asarray([1.0, 2.0, 10.0, 12.0])
        response = np.asarray([2.0, 4.0, 20.0, 24.0])
        labels = ["a", "a", "b", "b"]

        summary = correlation_summary(truth, response, labels)

        self.assertAlmostEqual(summary["pearson_r"], 1.0)
        self.assertAlmostEqual(summary["spearman_r"], 1.0)
        self.assertAlmostEqual(summary["within_group_pearson_r"], 1.0)


if __name__ == "__main__":
    unittest.main()
