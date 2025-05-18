from dataclasses import dataclass, field
import numpy as np
from typing import List


@dataclass
class Particle:
    """
    Класс частицы для SPH расчётов.
    Атрибуты:
        id: уникальный идентификатор частицы
        m: масса частицы
        rho: плотность
        p: давление
        x: координаты (вектор)
        v: скорость (вектор)
        h: сглаживающая длина
        neigh: список индексов соседних частиц
        neigh_w: список значений ядра W для каждого соседа
        drho_dt: временная производная плотности
        dv_dt: временная производная скорости (ускорение)
        state: 0 - твёрдый, 1 - жидкий, 2 - газ
    """
    id: int
    m: float
    rho: float
    p: float
    x: np.ndarray
    v: np.ndarray
    h: float
    neigh: List[int]
    neigh_w: List[float]
    drho_dt: float
    dv_dt: np.ndarray
    state: int
