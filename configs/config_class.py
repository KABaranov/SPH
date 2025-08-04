import numpy as np

from SPH.core.kernels.get_kernel import get_kernel
from SPH.core.neighbor_search.get_neigbor_search import get_neighbor_search
from SPH.core.correctors.get_corrector import get_corrector


class Config:
    def __init__(
        self,
        start_param: dict,
    ) -> None:
        if "scenario" not in start_param.keys():
            raise ValueError("Нужно указать сценарий (scenario) в параметрах")
        self.scenario_name = start_param["scenario"]
        self.scenario_param = start_param["scenario_param"]
        if "dim" not in self.scenario_param.keys():
            raise ValueError("Нужно указать количество измерений (dim) в параметрах сценария")
        self.dim = self.scenario_param["dim"]
        self.out_plot = bool(self.scenario_param["out_plot"]) if "out_plot" in self.scenario_param.keys() else False

        if "pst" not in start_param.keys():
            raise ValueError("Нужно указать pst (pst) в параметрах")
        self.pst_name = start_param["pst"]
        if self.pst_name.lower() != "none":
            self.pst_param = start_param["pst_param"]

        if "viscosity" not in start_param.keys():
            raise ValueError("Нужно указать тип вязкости (viscosity) в параметрах")
        self.viscosity_name = start_param["viscosity"]
        self.viscosity_param = start_param["viscosity_param"]

        if "kernel" not in start_param.keys():
            raise ValueError("Нужно указать ядро (kernel) в параметрах")
        self.kernel_name = start_param["kernel"]
        self.kernel_param = start_param["kernel_param"]
        self.kernel, self.grad = get_kernel(self.kernel_name)

        if "dx" not in self.scenario_param.keys():
            raise ValueError("Нужно указать dx (dx) в параметрах сценария")
        for param in ["kappa0", "alpha", "beta"]:
            if param not in self.kernel_param.keys():
                raise ValueError(f"Нужно указать {param} ({param}) для рассчёта h в параметрах ядра")
        self.h = self.kernel_param["kappa0"] * self.scenario_param["dx"]**self.kernel_param["alpha"]

        self.rho0 = start_param["rho0"] if "rho0" in start_param.keys() else 1000.0
        self.gamma = start_param["gamma"] if "gamma" in start_param.keys() else 7.0
        if "B" in start_param.keys():
            self.B = start_param["B"]
            self.c0 = np.sqrt(self.B * self.gamma / self.rho0)
        else:
            if "c0" in start_param.keys():
                self.c0 = start_param["c0"]
                self.B = self.rho0 * self.c0 ** 2.0 / self.gamma
            else:
                raise ValueError("Нужно указать B или c0 для EOS")
        self.p_floor = start_param["p_floor"] if "p_floor" in start_param.keys() else 0.0

        if "neighbor_method" not in start_param.keys():
            raise ValueError(f"Нужно метод поиска соседей (neighbor_method) в параметрах")
        self.neighbor_method = start_param["neighbor_method"]
        self.neighbor_search = get_neighbor_search(name=self.neighbor_method, dim=self.dim)
        if "is_periodic" in self.scenario_param.keys() and self.scenario_param["is_periodic"] in [1, "1", True, "True"]:
            self.is_periodic = True
        else:
            self.is_periodic = False
        self.qmax = start_param["qmax"] if "qmax" in start_param.keys() else 10.0

        if "corrector" not in start_param.keys():
            self.corrector_name = "none"
        else:
            self.corrector_name = start_param["corrector"]
        self.corrector = get_corrector(self.corrector_name)
        self.corrector_iter = start_param["corrector_iter"] if "corrector_iter" in start_param.keys() else 1
        self.corrector_period = start_param["corrector_period"] if "corrector_period" in start_param.keys() else 1

        self.total_time = start_param["total_time"] if "total_time" in start_param.keys() else 2.0
        self.dt = start_param["dt"] if "dt" in start_param.keys() else 0.001

        if "xyz_save" in start_param.keys() and start_param["xyz_save"] in [1, "1", True, "True"]:
            self.xyz_save = True
        else:
            self.xyz_save = False
        self.dump_period = start_param["dump_period"] if "dump_period" in start_param.keys() else 1
