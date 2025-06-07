import numpy as np
from SPH.core.kernels.get_kernel import get_kernel
from SPH.core.neighbor_search.get_neigbor_search import get_neighbor_search


class Config:
    def __init__(
        self,
        start_param: dict,
    ) -> None:
        self.scenario_name = start_param["scenario"]
        self.scenario_param = start_param["scenario_param"]

        self.pst_name = start_param["pst"]
        self.pst_param = start_param["pst_param"]

        self.viscosity = start_param["viscosity"]
        self.viscosity_param = start_param["viscosity_param"]

        self.kernel_name = start_param["kernel"]
        self.kernel_param = start_param["kernel_param"]

        self.dx = self.scenario_param["dx"] if "dx" in self.scenario_param.keys() else 0.01
        self.kappa0 = self.kernel_param["kappa0"] if "kappa0" in self.kernel_param.keys() else 1.3
        self.alpha = self.kernel_param["alpha"] if "alpha" in self.kernel_param.keys() else 1.0
        self.beta = self.kernel_param["beta"] if "beta" in self.kernel_param.keys() else 0.0
        self.h = self.kappa0 * self.dx**self.alpha

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
        self.epsilon = start_param["epsilon"] if "epsilon" in start_param.keys() else 0.01
        self.p_floor = start_param["p_floor"] if "p_floor" in start_param.keys() else 0.0
        self.dim = start_param["scenario_param"]["dim"] if "dim" in start_param["scenario_param"].keys() else 2

        self.kernel_name = start_param["kernel"] if "kernel" in start_param.keys() else "gauss"
        self.kernel, self.grad = get_kernel(self.kernel_name)
        self.neighbor_method = get_neighbor_search(start_param["neighbor_method"]) \
            if "neighbor_method" in start_param.keys() else "bruteforce"
