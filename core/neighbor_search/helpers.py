from __future__ import annotations

import numpy as np
from typing import Sequence


# вспомогательная функция для периодического расстояния
def minimal_image(dx: np.ndarray, dim: int, L: Sequence[float] | float):
    # dx: numpy array длины dim
    for k in range(dim):
        if dx[k] > 0.5 * L:
            dx[k] -= L
        elif dx[k] < -0.5 * L:
            dx[k] += L
    return dx
