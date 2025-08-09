from typing import Literal, Tuple


BCType = Literal["periodic", "reflective", "open", "free"]  # open = ничего не делаем по этой оси


def _periodic_wrap_scalar(x: float, lo: float, hi: float) -> float:
    """Периодическое сворачивание в [lo, hi) для любого вылета (мульти-перескоки)."""
    L = hi - lo
    if L <= 0:
        raise ValueError("periodic: hi must be > lo")
    # Возвращает lo <= x' < hi
    return lo + ((x - lo) % L)


def _reflective_fold_scalar(x: float, v: float, lo: float, hi: float, restitution: float = 1.0) -> Tuple[float, float]:
    """
    Зеркальное сворачивание в [lo, hi] с корректной инверсией скорости при нечётном числе отражений.
    Используем свёртку по периоду 2L: [lo, hi] затем [hi, lo] (зеркало).
    """
    L = hi - lo
    if L <= 0:
        raise ValueError("reflective: hi must be > lo")
    twoL = 2.0 * L
    # y в [0, 2L)
    y = (x - lo) % twoL
    if y < L:
        # «Прямой» участок
        x_new = lo + y
        v_new = v
    else:
        # «Зеркальный» участок
        x_new = hi - (y - L)
        v_new = -restitution * v
    return x_new, v_new
