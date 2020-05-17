from typing import Dict, List


def flatten_model_var_dict(model_vars: Dict) -> List[str]:
    """Flattens model variable name dict into single list. Function
        placed in this file to avoid cyclical dependencies."""
    return (list(model_vars['cat']) + list(model_vars['cont']) +
            [model_vars['target']])



