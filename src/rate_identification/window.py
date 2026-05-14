from __future__ import annotations

import numpy as np

from .config import WindowMappingConfig


def scale_to_window(scale: float, config: WindowMappingConfig) -> int:
    safe_scale = max(scale, 1e-6)
    window = config.alpha * (safe_scale ** (-config.beta))
    return int(np.clip(window, config.w_min, config.w_max))
