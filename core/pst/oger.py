from typing import List
import numpy as np

from SPH.configs.config_class import Config
from SPH.core.particle.particle_dataclass import Particle


def apply_pst_oger(cfg: Config, particles: List[Particle]) -> None:
    """
    Реализует смещение частиц по PST Оже (диффузия частиц).
    particles – список Particle.
    """
    if "D" not in cfg.pst_param.keys():
        raise ValueError("Укажите D в параметрах pst/oger")
    D = cfg.pst_param["D"]
    N = len(particles)
    # Шаг 1, 2: посчитать сглаженную концентрацию (число соседей) для каждой частицы + градиенты
    C = np.zeros(N)  # массив для хранения концентраций n_i
    gradC = [np.zeros_like(p.x) for p in particles]  # лист векторов нулевых
    for i, pi in enumerate(particles):
        for j, wj, grad_wj in zip(pi.neigh, pi.neigh_w, pi.grad_w):
            # if i == j:
            #     continue
            vj = particles[j].m / particles[j].rho
            C[i] += wj * vj
            gradC[i] += grad_wj * vj
    # (Если хотим более физично: можно C[i] не учитывать саму частицу i,
    # но W(0) обычно конечен для ядра, так что можно и учитывать, это просто добавит константу.)

    # Шаг 3: вычислить и применить смещения
    shifts = [np.zeros_like(p.x) for p in particles]
    for i, pi in enumerate(particles):
        # Расчёт величины сдвига. Берём -D * gradC (минус, чтобы от высокого C к низкому)
        # Учтём dt: если D задан без учета dt, то умножим на dt внутри.

        # 1 вариант
        # if np.linalg.norm(D * gradC[i]) < 0.25 * np.linalg.norm(pi.v):
        #     shifts[i] = - D * gradC[i] * cfg.dt
        # else:
        #     shifts[i] = -0.25 * np.linalg.norm(pi.v) * gradC[i] / np.linalg.norm(gradC[i])

        # 2 вариант
        shift = - D * gradC[i] * cfg.dt
        if np.linalg.norm(shift) < 0.2 * cfg.h:
            shifts[i] = shift
        else:
            shifts[i] = (gradC[i] / np.linalg.norm(gradC[i])) * (-0.2) * cfg.h

        # Можно добавить логику ограничения на свободной поверхности:
        # Например, если p_i.detected_as_surface:
        #     обнулить нормальную составляющую shifts[i].
        # Но здесь пропустим эту деталь для краткости.

    # Обновляем позиции всех частиц одновременно
    for i, pi in enumerate(particles):
        pi.x += shifts[i]
