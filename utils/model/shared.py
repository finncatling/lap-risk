from typing import Dict, List


def flatten_model_var_dict(model_vars: Dict) -> List[str]:
    """Flattens model variable name dict into single list."""
    return list(model_vars["cat"]) + list(model_vars["cont"]) + [model_vars["target"]]
