from typing import List
import numpy as np

from SPH.configs.config_class import Config
from SPH.core.particle.particle_dataclass import Particle


def apply_pst_monaghan(cfg: Config, particles: List[Particle]) -> None:
    """
    Применяет XSPH-коррекцию Монагана к скоростям частиц.
    particles – список объектов Particle.
    eps – коэффициент XSPH (малое число, например 0.3).
    h – сглаживающая длина ядра.
    """
    if "eps" not in cfg.pst_param.keys():
        raise ValueError("Укажите eps в параметрах pst/monaghan")
    eps = cfg.pst_param["eps"]
    # Создадим массив поправок скоростей
    v_corrections = [np.zeros_like(p.v, dtype=float) for p in particles]

    for i, pi in enumerate(particles):
        for j, wj in zip(pi.neigh, pi.neigh_w):
            pj = particles[j]
            if i == j:
                continue
            # Накопливаем вклад j в коррекцию скорости i:
            v_corrections[i] += (pj.m / pj.rho) * (pj.v - pi.v) * wj

    # Применяем коррекцию: обновляем положение частиц с использованием скорректированных скоростей.
    for i, pi in enumerate(particles):
        # Обновляем скорость частицы (хотя классический XSPH не меняет саму v,
        # а использует скорректированную только для позиции;
        # можно либо изменить p_i.v прямо, либо сохранить старую для расчёта сил).
        pi.v = pi.v + eps * v_corrections[i]
        # Обновляем положение частицы за шаг dt:
        pi.x = pi.x + pi.v * cfg.dt
        # Вариант: можно использовать p_i.v + eps*... прямо в шаге для r,
        # не меняя исходную p_i.v. В зависимости от того, как устроен интегратор.
