from SPH.core.particle.particle_dataclass import Particle
from SPH.core.boundary_condition.helpers import _periodic_wrap_scalar

from typing import List, Sequence


def periodic_bc(particles: List[Particle],
                bounds_lo: Sequence[float],
                bounds_hi: Sequence[float]) -> None:
    """
    Периодические ГУ по всем трём осям. Модифицирует частицы in-place.
    bounds_lo/hi — по 3 элемента (x,y,z).
    """
    for p in particles:
        for a in range(3):
            p.x[a] = _periodic_wrap_scalar(p.x[a], bounds_lo[a], bounds_hi[a])
        # скорость при периодике не меняется
