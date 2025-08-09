from SPH.core.particle.particle_dataclass import Particle
from SPH.core.boundary_condition.helpers import _reflective_fold_scalar

from typing import List, Sequence


def reflective_bc(particles: List[Particle],
                  bounds_lo: Sequence[float],
                  bounds_hi: Sequence[float],
                  restitution: Sequence[float] = (1.0, 1.0, 1.0)) -> None:
    """
    Отражающие ГУ по всем трём осям. Модифицирует частицы in-place.
    restitution — коэффициент восстановления по каждой оси (1.0 — упруго).
    Инвертируется только компонент скорости по нормали к стенке.
    """
    for p in particles:
        for a in range(3):
            x_new, v_new = _reflective_fold_scalar(p.x[a], p.v[a], bounds_lo[a], bounds_hi[a], restitution[a])
            p.x[a] = x_new
            p.v[a] = v_new
