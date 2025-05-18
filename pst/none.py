from typing import List
from sph.pst.base import IPST
from sph.core.particle import Particle


class PSTNone(IPST):
    """
    Пустая реализация техники сдвига частиц.
    Никаких изменений в скоростях не вносится.
    """
    def apply(self, particles: List[Particle], dt: float) -> None:
        # Нет коррекции скоростей
        pass
