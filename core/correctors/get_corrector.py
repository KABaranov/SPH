from typing import Callable

from SPH.core.correctors.shepard import shepard_filter
from SPH.core.correctors.mls import mls_correction


def get_corrector(name: str) -> Callable:
    corrector_list = {
        "none": None,
        "shepard": shepard_filter,
        "mls": mls_correction
    }
    if name.lower() not in corrector_list.keys():
        raise ValueError(f"Корректора {name} не существует в данной реализации")
    return corrector_list[name]
