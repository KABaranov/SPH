from SPH.core.pst.oger import apply_pst_oger
from SPH.core.pst.monaghan import apply_pst_monaghan


def get_pst(name: str):
    if name.lower() in ["xpst", "monaghan", "monaghan_pst"]:
        return apply_pst_monaghan
    elif name.lower() in ["oger", "oger_pst"]:
        return apply_pst_oger
    elif name.lower() in ["none", None]:
        return None
    else:
        raise ValueError("Данной техники сдвига частиц (PST) не существует в данной реализации")
