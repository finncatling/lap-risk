import os

import pandas as pd

from utils.constants import DATA_DIR, RANDOM_SEED
from utils.current_nela_model import (preprocess_df, SplitterTrainerPredictor,
                                      flatten_nela_var_dict, WINSOR_THRESHOLDS,
                                      CURRENT_NELA_MODEL_VARS, CENTRES)
from utils.split_data import drop_incomplete_cases
from utils.io import load_object, save_object
from utils.report import Reporter


reporter = Reporter()


reporter.first('Loading manually-wrangled NELA data')
df = pd.read_pickle(
    os.path.join(DATA_DIR, 'df_after_univariate_wrangling.pkl'))


reporter.report('Removing unused variables')
df = df[flatten_nela_var_dict(CURRENT_NELA_MODEL_VARS)]


reporter.report("Drop cases which are incomplete for the models' variables")
df, _ = drop_incomplete_cases(df)


reporter.report('Preparing list of variables for binarization')
"""
GCS is exempt from binarisation as it is binned in a separate function. ASA is
exempt as it is only used in a later interaction term.
"""
binarize_vars = list(CURRENT_NELA_MODEL_VARS['cat'])
binarize_vars.remove('S03GlasgowComaScore')
binarize_vars.remove('S03ASAScore')


reporter.report('Preparing list of variables for quadratic transformation')
"""
Sodium is exempt as it undergoes a customised transformation. We apply the
quadratic transformation to creatinine and urea after they are log-transformed.
"""
quadratic_vars = list(CURRENT_NELA_MODEL_VARS['cont'])
quadratic_vars.remove('S03Sodium')
for original, logged in (('S03SerumCreatinine', 'logcreat'),
                         ('S03Urea', 'logurea')):
    quadratic_vars.remove(original)
    quadratic_vars.append(logged)


reporter.report('Preprocessing data')
"""
Our preprocessing code was originally designed to allow preprocessing after
splitting the data. However, the preprocessing steps do not 'leak' any
information from the test-fold cases, e.g. they don't use any summary statistics
derived from the data for use in transforming the variables. This means that
we can speed up our code by running the preprocessing prior to the loop wherein
the data are repeatedly split and the model retrained.  
"""
preprocessed_df, _ = preprocess_df(
    df,
    current_nela_model_vars=CURRENT_NELA_MODEL_VARS,
    quadratic_vars=quadratic_vars,
    winsor_threholds=WINSOR_THRESHOLDS,
    centres=CENTRES,
    binarize_vars=binarize_vars,
    label_binarizers=None)


reporter.report('Loading data needed for train-test splitting')
tt_splitter = load_object(os.path.join('outputs', 'train_test_splitter.pkl'))


reporter.report('Beginning train-test splitting and model fitting')
stp = SplitterTrainerPredictor(
    df,
    test_train_splitter=tt_splitter,
    target_variable_name=CURRENT_NELA_MODEL_VARS['target'],
    random_seed=RANDOM_SEED)
stp.split_train_predict()


reporter.report('Saving SplitterTrainerPredictor for use in model evaluation')
save_object(tt_splitter, os.path.join('outputs', 'train_test_splitter.pkl'))


# TODO: Evaluate model predictions
