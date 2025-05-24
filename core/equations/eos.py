from typing import Optional
import numpy as np


# ------------------------------------------------------------------
# Уравнение состояния
# ------------------------------------------------------------------
def eos(rho: float,
        rho0: float,
        gamma: float,
        p_floor: float,
        b: Optional[float] = None,
        c0: Optional[float] = None) -> float:
    if b is None:
        if c0 is None:
            raise ValueError("Нужно указать B или c0 для EOS")
        b = rho0 * c0 ** 2.0 / gamma
    """Tait EOS: p = B * ((rho / rho0)^gamma − 1)  c отсечкой p ≥ p_floor"""
    p = b * ((rho / rho0) ** gamma - 1.0)
    return max(p, p_floor)
