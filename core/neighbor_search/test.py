from __future__ import annotations

from SPH.configs.config_class import Config
from SPH.core.particle.particle_dataclass import Particle

import matplotlib.pyplot as plt
import numpy as np
from typing import Sequence, List, Callable

from SPH.configs.get_config import get_config
from SPH.core.neighbor_search.bruteforce import find_neigh_bruteforce
from SPH.core.neighbor_search.cell_linked import find_neigh_cell_list


def ns_time_test1d(particles: List[Particle], cfg: Config,
                   search_function: Callable,
                   is_periodic: bool = False) -> List[float]:
    results = []
    qmax = 10.0
    search_function(particles, qmax, is_periodic)
    print(results)
    return results


def ns_multitest_1d(cfg: Config, functions: dict,
                    is_periodic: bool = False) -> None:
    result = []
    particles0 = []
    n = cfg.Lx / cfg.dx
    x = [i * cfg.dx for i in range(int(cfg.Lx // cfg.dx) + 2)]
    for xi in x:
        p = Particle(
            id=len(particles0), m=cfg.rho0 * (cfg.dx ** cfg.dim), p=0, x=np.array([xi, 0, 0]),
            drho_dt=0, dv_dt=np.array([0, 0, 0]), state=1, h=cfg.h,
            neigh=[], neigh_w=[], rho=cfg.rho0, v=np.array([0, 0, 0])
        )
        particles0.append(p)
    for name in functions.keys():
        result.append(ns_time_test1d(particles0, cfg=cfg, search_function=functions[name], is_periodic=is_periodic))


if __name__ == "__main__":
    neigh_func = {
        "Brute force": find_neigh_bruteforce,
        "Cell linked": find_neigh_cell_list
    }
    ns_multitest_1d(get_config("common", print_param=True), neigh_func, is_periodic=False)
