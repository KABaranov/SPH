from SPH.core.particle.particle_dataclass import Particle
from SPH.core.boundary_condition.helpers import BCType, _reflective_fold_scalar, _periodic_wrap_scalar

from typing import List, Tuple, Sequence


def apply_bc(particles: List[Particle],
             bounds_lo: Sequence[float],
             bounds_hi: Sequence[float],
             bc_per_axis: Tuple[BCType, BCType, BCType],
             restitution: Sequence[float] = (1.0, 1.0, 1.0)) -> None:
    """
    Универсальная функция: принимает три БК (по x,y,z) и применяет их.
    bc_per_axis: кортеж из строк: "periodic" | "reflective" | "open".
    restitution: коэффициенты восстановления для отражения по осям.
    """
    # Маленький eps, чтобы убрать точные попадания в hi из-за округления (по желанию)
    eps = 0.0

    for p in particles:
        for a, bc in enumerate(bc_per_axis):
            lo, hi = bounds_lo[a], bounds_hi[a]
            if bc == "periodic":
                # Периодика по оси a
                p.x[a] = _periodic_wrap_scalar(p.x[a], lo, hi - eps)
                # скорость не меняется
            elif bc == "reflective":
                # Отражение по оси a, инвертируем только нормальную компоненту
                x_new, v_new = _reflective_fold_scalar(p.x[a], p.v[a], lo, hi - eps, restitution[a])
                p.x[a] = x_new
                p.v[a] = v_new
            elif bc in ["open", "free"]:
                # Ничего не делаем по этой оси (outflow/без стенки)
                continue
            else:
                raise ValueError(f"Unknown BC type '{bc}' for axis {a}")
