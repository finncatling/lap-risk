from typing import Tuple, Union
from collections import OrderedDict
from dataclasses import dataclass
import numpy as np
import pandas as pd

from utils.split import TrainTestSplitter
from utils.wrangling import percent_missing


@dataclass
class DemographicTableVariable:
    name: str
    pretty_name: str
    in_novel_model: bool
    var_type: str  # {'continuous', 'binary', 'ordinal_multicat', 'multicat'}
    decimal_places: int = 0
    category_labels: Union[None, Tuple[str, ...]] = None


def generate_demographic_table(
    variables: Tuple[DemographicTableVariable, ...],
    this_df: pd.DataFrame,
    modified_tts: TrainTestSplitter,
    output_filepath: str
):
    dfs = OrderedDict()
    dfs['all'] = this_df
    dfs['train'] = this_df.loc[
        modified_tts.train_i[0]].copy().reset_index(drop=True)
    dfs['test'] = this_df.loc[
        modified_tts.test_i[0]].copy().reset_index(drop=True)

    # Initialise demographic table
    table = pd.DataFrame(
        data=np.zeros((len(variables), 6)),
        columns=(
            'Variable',
            f'All cases (n={dfs["all"].shape[0]})',
            f'Development cases (n={dfs["train"].shape[0]})',
            f'Evalulation cases (n={dfs["test"].shape[0]})',
            'Missing values',
            'In novel model'
        ),
        dtype=str
    )

    for var_i, var in enumerate(variables):
        # Calculate summary statistics
        if var.var_type in ('continuous', 'ordinal_multicat'):
            table.loc[var_i, 'Variable'] = f'{var.pretty_name}: median (IQR)'
            for df_i, this_df in enumerate(dfs.values()):
                quantiles = this_df[var.name].quantile([0.25, 0.5, 0.75]).values

                if var.var_type == 'continuous':
                    if var.decimal_places > 0:
                        quantiles = np.round(quantiles, var.decimal_places)
                    else:
                        quantiles = np.round(quantiles).astype(int).astype(str)
                    table.iloc[var_i, df_i + 1] = (
                        f'{quantiles[1]} ({quantiles[0]} - {quantiles[2]})')

                elif var.var_type == 'ordinal_multicat':
                    quantiles = np.round(quantiles, var.decimal_places)
                    labelled_quantiles = [
                        var.category_labels[label_i] for label_i in quantiles]
                    table.iloc[var_i, df_i + 1] = (
                        f'{labelled_quantiles[1]} ({labelled_quantiles[0]} - '
                        f'{labelled_quantiles[2]})')

        elif var.var_type == 'binary':
            table.loc[var_i, 'Variable'] = (
                f'{var.pretty_name}: n (% of non-missing)')
            for df_i, this_df in enumerate(dfs.values()):
                n_nonnull = this_df.loc[this_df[var.name].notnull()].shape[0]
                n_positive = this_df.loc[this_df[var.name] == 1].shape[0]
                perc_positive = np.round(n_positive / n_nonnull * 100, 1)
                table.iloc[var_i, df_i + 1] = f'{n_positive} ({perc_positive}%)'

        elif var.var_type == 'multicat':
            pass

        # Calculate missingness
        n_missing = dfs['all'].loc[dfs['all'][var.name].isnull()].shape[0]
        perc_missing = np.round(percent_missing(dfs['all'], var.name), 1)
        table.loc[var_i, 'Missing values'] = (
            f'{n_missing} ({perc_missing}%)')

        # Add indicator for whether variable in novel model
        if var.in_novel_model:
            table.loc[var_i, 'In novel model'] = 'Yes'
        else:
            table.loc[var_i, 'In novel model'] = 'No'

    return table
