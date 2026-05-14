from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from PIL import Image, ImageFilter


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(root: Union[str, Path]) -> List[Path]:
    root_path = Path(root)
    if root_path.is_file():
        return [root_path]
    return sorted(
        path
        for path in root_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def load_rgb_image(path: Union[str, Path]) -> Image.Image:
    return Image.open(path).convert("RGB")


def resize_long_edge(image: Image.Image, long_edge: int) -> Image.Image:
    width, height = image.size
    if max(width, height) == long_edge:
        return image.copy()

    scale = long_edge / float(max(width, height))
    resized = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return image.resize(resized, Image.Resampling.BICUBIC)


def make_inference_views(image: Image.Image) -> list[Image.Image]:
    half = image.resize(
        (max(1, image.size[0] // 2), max(1, image.size[1] // 2)),
        Image.Resampling.BICUBIC,
    )
    half = half.resize(image.size, Image.Resampling.BICUBIC)
    blurred = image.filter(ImageFilter.GaussianBlur(radius=1.5))
    return [image, half, blurred]


def pil_to_array(image: Image.Image, size: Tuple[int, int]) -> np.ndarray:
    return np.asarray(image.resize(size, Image.Resampling.BICUBIC), dtype=np.float32)
