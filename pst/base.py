from abc import ABC, abstractmethod
from typing import List
from sph.core.particle import Particle


class IPST(ABC):
    """
    Интерфейс техники сдвига частиц (ТСЧ = PST).
    Реализации должны определять метод apply,
    который модифицирует скоростное поле частиц.
    """
    @abstractmethod
    def apply(self, particles: List[Particle], dt: float) -> None:
        """
        Применить технику сдвига частиц к списку частиц.

        Args:
            particles: список объектов Particle
            dt: текущий шаг по времени
        """
        pass
