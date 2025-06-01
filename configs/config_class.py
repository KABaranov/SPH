import numpy as np
from SPH.core.kernels.get_kernel import get_kernel
from SPH.core.neighbor_search.get_neigbor_search import get_neighbor_search


class Config:
    def __init__(
        self,
        start_param: dict,
    ) -> None:
        self.dx = start_param["dx"] if "dx" in start_param.keys() else 0.01
        self.kappa0 = start_param["kappa0"] if "kappa0" in start_param.keys() else 1.3
        self.alpha = start_param["alpha"] if "alpha" in start_param.keys() else 1.0
        self.beta = start_param["beta"] if "beta" in start_param.keys() else 0.0
        self.h = self.kappa0 * self.dx**self.alpha

        self.scenario_name = start_param["scenario"]

        self.Lx = start_param["Lx"] if "Lx" in start_param.keys() else self.dx
        self.Ly = start_param["Ly"] if "Ly" in start_param.keys() else self.dx
        self.Lz = start_param["Lz"] if "Lz" in start_param.keys() else self.dx
        self.L = start_param["L"] if "L" in start_param.keys() else (self.Lx, self.Ly, self.Lz)

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
        self.dim = start_param["dim"] if "dim" in start_param.keys() else 2

        self.kernel_name = start_param["kernel"] if "kernel" in start_param.keys() else "gauss"
        self.kernel, self.grad = get_kernel(self.kernel_name)
        self.neighbor_method = get_neighbor_search(start_param["neighbor_method"]) \
            if "neighbor_method" in start_param.keys() else "bruteforce"
