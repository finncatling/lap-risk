import os

import numpy as np
import pandas as pd

from utils.helpers import load_object


def drop_incomplete_cases(df: pd.DataFrame) -> (pd.DataFrame, int, int):
    """Drops incomplete rows in input DataFrame, printing pre- and
        post-drop summary stats."""
    total_n = df.shape[0]
    df = df.dropna()
    complete_n = df.shape[0]

    print(f'{total_n} cases in input DataFrame')
    print(f'Dropped {total_n - complete_n} incomplete cases',
          f'({np.round(100 * (1 - complete_n / total_n), 3)}%)')
    print(f'{complete_n} complete cases in returned Dataframe.')

    return df, total_n, complete_n


def split_into_folds(
        df: pd.DataFrame,
        tts_filepath: str = os.path.join('data', 'train_test_split.pkl')
) -> (pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray):
    """Splits supplied DataFrame into train and test folds, such that the test
        fold is the cases from the test trusts which are complete for the
        variables
        in the current NELA risk model, and the train fold is all the available
        cases from the train trusts. Two things to note:

        1) The train fold will be different between models as for the current
        NELA
        model it will only contain complete cases, whereas for our model it will
        also contain the cases that were incomplete prior to imputation.

        2) The test fold will be the same for all models, i.e. the current-NELA-
        model incomplete cases from the test fold trusts will not be used in
        training or testing of any of the models."""
    train_test_split = load_object(tts_filepath)
    split = {}

    for fold in ('train', 'test'):
        if fold == 'train':
            train_total = train_test_split[f'{fold}_i'].shape[0]
            train_test_split[f'{fold}_i'] = np.array([i for i in
                                                      train_test_split[
                                                          f'{fold}_i'] if
                                                      i in df.index])
            train_intersection = train_test_split[f'{fold}_i'].shape[0]
            percent_unavailable = 100 * (1 - train_intersection / train_total)
            print(f'{train_total} cases in unabridged train fold.')
            print(f'Excluded {train_total - train_intersection} cases',
                  f'({np.round(percent_unavailable, 3)}%)',
                  'not available in input DataFrame.')
            print(f'{train_intersection} cases in returned train Dataframe.')

        split[fold] = {'X_df': df.loc[
            train_test_split[f'{fold}_i']].copy().reset_index(drop=True)}
        split[fold]['y'] = split[fold]['X_df']['Target'].values
        split[fold]['X_df'] = split[fold]['X_df'].drop('Target', axis=1)

    assert (split['test']['X_df'].shape[0] == train_test_split['test_i'].shape[
        0])

    return (split['train']['X_df'], split['train']['y'],
            split['test']['X_df'], split['test']['y'])