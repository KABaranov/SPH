from SPH.scenario.density_test1 import density_test1

from typing import Callable


def get_scenario(name: str) -> Callable:
    scenario_list = {
        "density_test1": density_test1
    }
    return scenario_list[name]
