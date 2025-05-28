from .cubic_spline import *
from .gauss import *
from .wendland_c2 import *
from typing import Callable, Tuple


def get_kernel(name: str = 'gauss') -> Tuple[Callable, Callable]:
    if name.lower() in ['gauss', 'gaussian', None]:
        return gaussian_kernel, gaussian_grad
    elif name.lower() in ['wendland_c2', 'wendland_c2', 'wendland_c2']:
        return wendland_c2_kernel, wendland_c2_grad
    elif name.lower() in ['cubic_spline', 'spline']:
        return cubic_spline_kernel, cubic_spline_grad
    else:
        raise ValueError('Указанное ядро не существует в данной реализации')
