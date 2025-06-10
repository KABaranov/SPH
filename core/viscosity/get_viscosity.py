from SPH.core.viscosity.artificial_viscosity import artificial_viscosity
from SPH.core.viscosity.laminar_viscosity import laminar_viscosity
from typing import Callable


def get_viscosity(name: str) -> Callable:
    if name.lower() in ["artificial", "artificial_viscosity", None]:
        return artificial_viscosity
    if name.lower() in ["laminar", "laminar_viscosity"]:
        return laminar_viscosity
    else:
        raise ValueError('Указанный метод поиска соседей не существует в данной реализации')
