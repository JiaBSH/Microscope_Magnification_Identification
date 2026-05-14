from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from PIL import Image

from .data import load_rgb_image, pil_to_array, resize_long_edge


class FeatureExtractor(ABC):
    name: str

    @abstractmethod
    def extract_image(self, image: Image.Image) -> np.ndarray:
        raise NotImplementedError

    def extract(self, image_path: Union[str, Path]) -> np.ndarray:
        return self.extract_image(load_rgb_image(image_path))

    def extract_many(self, image_paths: List[Union[str, Path]]) -> np.ndarray:
        return np.vstack([self.extract(path) for path in image_paths])


class RobustHandcraftedExtractor(FeatureExtractor):
    name = "robust_handcrafted"

    def __init__(self, resize_edge: int = 1024) -> None:
        self.resize_edge = resize_edge

    def extract_image(self, image: Image.Image) -> np.ndarray:
        image = resize_long_edge(image.convert("RGB"), self.resize_edge)
        rgb = pil_to_array(image, (128, 128)) / 255.0
        gray = rgb.mean(axis=2)

        intensity_hist, _ = np.histogram(gray, bins=16, range=(0.0, 1.0), density=True)

        grad_x = np.diff(gray, axis=1, prepend=gray[:, :1])
        grad_y = np.diff(gray, axis=0, prepend=gray[:1, :])
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_hist, _ = np.histogram(grad_mag, bins=16, range=(0.0, 1.0), density=True)

        channel_stats = np.concatenate([rgb.mean(axis=(0, 1)), rgb.std(axis=(0, 1))])

        pooled = gray.reshape(16, 8, 16, 8).mean(axis=(1, 3)).reshape(-1)

        spectrum = np.fft.fftshift(np.abs(np.fft.fft2(gray)))
        yy, xx = np.indices(spectrum.shape)
        center = np.array(spectrum.shape)[:, None, None] / 2.0
        radius = np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2)
        bins = np.linspace(0, radius.max(), 17)
        radial_energy = []
        for lower, upper in zip(bins[:-1], bins[1:]):
            mask = (radius >= lower) & (radius < upper)
            radial_energy.append(float(spectrum[mask].mean()) if np.any(mask) else 0.0)

        feature = np.concatenate(
            [intensity_hist, grad_hist, channel_stats, pooled, np.asarray(radial_energy)]
        ).astype(np.float32)
        norm = np.linalg.norm(feature)
        return feature / norm if norm > 0 else feature


class DinoV2TimmExtractor(FeatureExtractor):
    name = "dinov2_timm"

    def __init__(self, resize_edge: int = 1024, model_name: str = "vit_small_patch14_dinov2") -> None:
        self.resize_edge = resize_edge
        self.model_name = model_name
        self._model: Optional[object] = None
        self._transform: Optional[object] = None
        self._device: Optional[object] = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        try:
            import timm
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "DINOv2 extractor requires optional dependencies: pip install -e .[dinov2]"
            ) from exc

        self._model = timm.create_model(self.model_name, pretrained=True, num_classes=0)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()
        data_config = timm.data.resolve_model_data_config(self._model)
        self._transform = timm.data.create_transform(**data_config, is_training=False)
        self._torch = torch

    def extract_image(self, image: Image.Image) -> np.ndarray:
        self._ensure_model()
        assert self._transform is not None
        assert self._device is not None
        image = resize_long_edge(image.convert("RGB"), self.resize_edge)
        tensor = self._transform(image).unsqueeze(0).to(self._device)
        with self._torch.inference_mode():
            embedding = self._model(tensor).cpu().numpy()[0].astype(np.float32)
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding


def build_extractor(name: str, resize_edge: int) -> FeatureExtractor:
    if name == RobustHandcraftedExtractor.name:
        return RobustHandcraftedExtractor(resize_edge=resize_edge)
    if name == DinoV2TimmExtractor.name:
        return DinoV2TimmExtractor(resize_edge=resize_edge)
    raise ValueError(f"Unsupported extractor: {name}")
