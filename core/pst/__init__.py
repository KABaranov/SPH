"""
Пакет sph.pst: техники сдвига частиц (PST).
"""
from .base import IPST
from .none import PSTNone
from .monaghan import MonaghanXSPH
from .oger import OgerPST

__all__ = [
    "IPST",
    "PSTNone",
    "MonaghanXSPH",
    "OgerPST",
    "get_pst",
]


def get_pst(name: str, **kwargs) -> IPST:
    """
    Фабричный метод для получения PST-модуля по имени.

    Args:
        name: 'none', 'monaghan.yml' или 'oger'
        kwargs: параметры конструктора, например eps для Монaгана или beta и kernel для Огер.

    Returns:
        Экземпляр IPST.

    Raises:
        ValueError: если передано неизвестное имя.
    """
    key = name.strip().lower()
    if key in ("none",):
        # PSTNone не принимает параметров
        return PSTNone()
    if key in ("monaghan.yml", "xsph"):
        return MonaghanXSPH(**kwargs)
    if key in ("oger", "ale"):
        return OgerPST(**kwargs)
    raise ValueError(f"Unknown PST '{name}'; choose from 'none', 'monaghan.yml', 'oger'.")
