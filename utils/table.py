from typing import Tuple
from collections import OrderedDict
from dataclasses import dataclass
import numpy as np
import pandas as pd

from utils.split import TrainTestSplitter


@dataclass
class DemographicTableVariable:
    name: str
    pretty_name: str
    in_novel_model: bool
    var_type: str  # {'continuous', 'binary', 'ordinal_multicat', 'multicat'}
    decimal_places: int = 0


def generate_demographic_table(
    variables: Tuple[DemographicTableVariable, ...],
    df: pd.DataFrame,
    modified_tts: TrainTestSplitter,
    output_filepath: str
):
    dfs = OrderedDict()
    dfs['all'] = df
    dfs['train'] = df.loc[modified_tts.train_i[0]].copy().reset_index(drop=True)
    dfs['test'] = df.loc[modified_tts.test_i[0]].copy().reset_index(drop=True)

    table = pd.DataFrame(
        data=np.zeros((len(variables), 6)),
        columns=(
            'Variable',
            f'All cases (n={dfs["all"].shape[0]})',
            'Missing values (%)',
            f'Development cases (n={dfs["train"].shape[0]})',
            f'Evalulation cases (n={dfs["test"].shape[0]})',
            'In novel model'
        ),
        dtype=str
    )

    for var_i, var in enumerate(variables):
        if var.var_type == 'continuous':
            table.loc[var_i, 'Variable'] = f'{var.pretty_name}: median (IQR)'
            for df_i, df in enumerate(dfs.values()):
                quantiles = df[var.name].quantile([0.25, 0.5, 0.75]).values
                if var.decimal_places > 0:
                    quantiles = np.round(quantiles, var.decimal_places)
                else:
                    quantiles = np.round(quantiles).astype(int).astype(str)
                table.iloc[var_i, df_i + 2] = (
                    f'{quantiles[1]} ({quantiles[0]} - {quantiles[2]})')

        elif var.var_type == 'binary':
            pass

        elif var.var_type == 'ordinal_multicat':
            pass

        elif var.var_type == 'multicat':
            pass

    return table
