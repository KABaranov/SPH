from SPH.scenario.density_test1 import density_test1
from SPH.scenario.eos_test import test_eos_all
from SPH.scenario.density_test2 import density_test2
from SPH.scenario.square_drop import square_drop

from typing import Callable


def get_scenario(name: str) -> Callable:
    scenario_list = {
        "eos_test": test_eos_all,
        "density_test1": density_test1,
        "density_test2": density_test2,
        "square_drop": square_drop
    }
    if name not in scenario_list.keys():
        raise ValueError(f"Сценария {name} не существует в данной реализации")
    return scenario_list[name]
