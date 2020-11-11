from typing import Dict, List

import numpy as np


def flatten_model_var_dict(model_vars: Dict) -> List[str]:
    """Flattens model variable name dict into single list."""
    return (
        list(model_vars["cat"]) +
        list(model_vars["cont"]) +
        [model_vars["target"]]
    )


class LogOddsTransformer:
    """Used to inverse transform novel model PDPs from log odds space into
        probability space. Implemented as a class just to match the relevant
        parts of the sklearn QuantileTransformer API."""

    def __init__(self):
        pass

    @staticmethod
    def inverse_transform(log_odds: np.ndarray) -> np.ndarray:
        odds = np.exp(log_odds)
        return odds / (1 + odds)
