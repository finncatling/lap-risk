from typing import Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Table1Variable:
    name: str
    pretty_name: str
    in_novel_model: bool
    var_type: str  # {'continuous', 'binary', 'ordinal_multicat', 'multicat'}
    raw_data: bool = False  # If True, get from raw data, not preprocessed
    decimal_places: int = 0


def generate_demographic_table(
    specification: Tuple[Table1Variable, ...],
    preprocessed_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    output_filepath: str
):
    table = pd.DataFrame(
        data=np.zeros((len(specification), 6)),
        columns=(
            'Variable',
            f'All cases (n={preprocessed_df.shape[0]})'
        ),
        dtype=str
    )
