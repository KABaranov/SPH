from SPH.core.neighbor_search.bruteforce import bruteforce
from SPH.core.neighbor_search.cell_linked import cell_linked
from SPH.core.neighbor_search.kd_tree import kd_tree
from typing import Callable


def get_neighbor_search(name: str, dim: int) -> Callable:
    if name.lower() in ['bruteforce', 'brute_force', None]:
        return bruteforce(dim)
    elif name.lower() in ['cell_list', 'celllist', 'cell', 'cell_linked', 'celllinked']:
        return cell_linked(dim)
    elif name.lower() in ["kd_tree", "tree", "ckd_tree", "kdtree"]:
        return kd_tree(dim)
    else:
        raise ValueError('Указанный метод поиска соседей не существует в данной реализации')
