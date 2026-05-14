from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Sequence

from PIL import Image, ImageDraw, ImageOps

from .evaluation import collect_labeled_images


def generate_window_visualizations(
    labeled_root: Path,
    recommendations_csv: Path,
    output_dir: Path,
    display_size: int = 768,
) -> List[Dict[str, object]]:
    recommendations = _read_recommendations(recommendations_csv)
    samples = collect_labeled_images(labeled_root)
    grouped = {}
    for sample in samples:
        variant = "swinir" if sample.is_swinir else "native"
        grouped.setdefault(sample.label_name, {}).setdefault(variant, []).append(sample.image_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for row in recommendations:
        label = str(row["label"])
        crop_size = int(row["square_window_aligned"])
        for variant, paths in sorted(grouped[label].items()):
            image_path = _choose_representative_image(paths)
            image = Image.open(image_path).convert("RGB")
            patch_path = output_dir / f"{label}_{variant}_patch.png"
            preview_path = output_dir / f"{label}_{variant}_preview.png"

            if min(image.size) >= crop_size:
                patch = _center_crop(image, crop_size)
                _save_display_patch(patch, patch_path, display_size, crop_size)
                _save_preview(image, crop_size, preview_path, fits=True)
                visualization_mode = "patch"
            else:
                fitted_crop = min(image.size)
                patch = _center_crop(image, fitted_crop)
                _save_display_patch(patch, patch_path, display_size, crop_size)
                _save_preview(image, crop_size, preview_path, fits=False)
                visualization_mode = "preview_only"

            summary.append(
                {
                    "label": label,
                    "variant": variant,
                    "image": str(image_path),
                    "image_width": image.size[0],
                    "image_height": image.size[1],
                    "recommended_square_window": crop_size,
                    "display_size": display_size,
                    "patch_path": str(patch_path),
                    "preview_path": str(preview_path),
                    "mode": visualization_mode,
                }
            )

    return summary


def write_visualization_summary_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "label",
        "variant",
        "image",
        "image_width",
        "image_height",
        "recommended_square_window",
        "display_size",
        "patch_path",
        "preview_path",
        "mode",
    ]
    with path.open("w", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow([row[column] for column in header])


def format_visualization_report(rows: Sequence[Dict[str, object]]) -> str:
    lines = ["saved_visualizations=one patch and one preview per magnification"]
    lines.append(_format_table(rows))
    return "\n".join(lines)


def _read_recommendations(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            parsed = dict(row)
            parsed["square_window_aligned"] = int(row["square_window_aligned"])
            rows.append(parsed)
        return rows


def _choose_representative_image(paths: Sequence[Path]) -> Path:
    return sorted(paths)[0]


def _center_crop(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    left = max(0, (width - size) // 2)
    top = max(0, (height - size) // 2)
    return image.crop((left, top, left + size, top + size))


def _save_preview(image: Image.Image, crop_size: int, output_path: Path, fits: bool) -> None:
    preview = image.copy()
    preview.thumbnail((1024, 1024), Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(preview)
    scale_x = preview.size[0] / float(image.size[0])
    scale_y = preview.size[1] / float(image.size[1])
    box_size_x = crop_size * scale_x
    box_size_y = crop_size * scale_y
    box_size = min(box_size_x, box_size_y)
    center_x = preview.size[0] / 2.0
    center_y = preview.size[1] / 2.0
    left = center_x - box_size / 2.0
    top = center_y - box_size / 2.0
    right = center_x + box_size / 2.0
    bottom = center_y + box_size / 2.0
    color = (0, 255, 0) if fits else (255, 0, 0)
    draw.rectangle((left, top, right, bottom), outline=color, width=4)
    preview.save(output_path)


def _save_display_patch(patch: Image.Image, output_path: Path, display_size: int, recommended_crop_size: int) -> None:
    canvas = ImageOps.pad(patch, (display_size, display_size), method=Image.Resampling.NEAREST, color=(0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((3, 3, display_size - 4, display_size - 4), outline=(255, 255, 255), width=4)
    draw.text((16, 16), f"window={recommended_crop_size}", fill=(255, 255, 0))
    canvas.save(output_path)


def _format_table(rows: Sequence[Dict[str, object]]) -> str:
    header = ["label", "variant", "image_size", "window", "mode", "patch", "preview"]
    table_rows = [header]
    for row in rows:
        table_rows.append(
            [
                str(row["label"]),
                str(row["variant"]),
                f"{row['image_width']}x{row['image_height']}",
                str(row["recommended_square_window"]),
                str(row["mode"]),
                Path(str(row["patch_path"])).name,
                Path(str(row["preview_path"])).name,
            ]
        )
    widths = [max(len(r[col]) for r in table_rows) for col in range(len(header))]
    return "\n".join(
        " ".join(value.rjust(widths[idx]) for idx, value in enumerate(row))
        for row in table_rows
    )
