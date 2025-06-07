from .bruteforce import bruteforce
from .cell_linked import find_neigh_cell_list
from typing import Callable


def get_neighbor_search(name: str, dim: int) -> Callable:
    if name.lower() in ['bruteforce', 'brute_force', None]:
        return bruteforce(dim)
    elif name.lower() in ['cell_list', 'celllist', 'cell']:
        return find_neigh_cell_list
    else:
        raise ValueError('Указанный метод поиска соседей не существует в данной реализации')
