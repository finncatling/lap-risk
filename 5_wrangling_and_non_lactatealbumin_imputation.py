import copy
import os
from typing import Tuple, Dict

import pandas as pd

from utils.constants import (DATA_DIR, STATS_OUTPUT_DIR, INDICATION_PREFIX,
                             MISSING_IND_CATEGORY)
from utils.model.shared import flatten_model_var_dict
from utils.io import make_directory, load_object
from utils.model.novel import (NOVEL_MODEL_VARS, MULTI_CATEGORY_LEVELS,
                               LACTATE_VAR_NAME, ALBUMIN_VAR_NAME,
                               LACTATE_ALBUMIN_VARS,
                               preprocess_novel_pre_split)
from utils.impute import ImputationInfo, SplitterWinsorMICE
from utils.report import Reporter


INDICATION_VAR_NAME = 'Indication'


reporter = Reporter()
reporter.title('Wrangle NELA data in preparation for later input to the '
               'novel model. Multiply impute missing values for all variables '
               'apart from lactate and albumin')


reporter.report("Creating output dirs (if they don't already exist)")
make_directory(STATS_OUTPUT_DIR)


reporter.report('Loading manually-wrangled NELA data')
df = pd.read_pickle(os.path.join(
    DATA_DIR, 'df_after_univariate_wrangling_new_indications.pkl'))


reporter.report('Finding names of indication variables')
indications = [c for c in df.columns if INDICATION_PREFIX in c]


reporter.report('Removing variables not used in the novel model')
df = df[flatten_model_var_dict(NOVEL_MODEL_VARS) + indications]


reporter.report('Preparing details of discrete variables')
multi_category_levels: Dict[str, Tuple] = copy.deepcopy(MULTI_CATEGORY_LEVELS)
multi_category_levels[INDICATION_VAR_NAME] = tuple(indications)
binary_vars = list(set(NOVEL_MODEL_VARS['cat']) -
                   set(multi_category_levels.keys()))


reporter.report('Doing pre-train test split data preprocessing')
df = preprocess_novel_pre_split(
    df,
    category_mapping={'S03ECG': {1.0: 0.0, 4.0: 1.0, 8.0: 1.0}},
    add_missingness_indicators_for=[LACTATE_VAR_NAME, ALBUMIN_VAR_NAME],
    indication_variable_name=INDICATION_VAR_NAME,
    indications=indications,
    missing_indication_value=MISSING_IND_CATEGORY,
    multi_category_levels=multi_category_levels)


reporter.report('Checking that there are no cases where all features are '
                'missing (these cases would be dropped by statsmodels MICEData,'
                ' which could create problems with the post-imputation data '
                'reconstruction)')
assert df.shape[0] == df.dropna(axis=0, how='all').shape[0]


reporter.report('Making DataFrame and variable list for use in MICE')
mice_df = df.drop(list(multi_category_levels.keys()) +
                  list(LACTATE_ALBUMIN_VARS) +
                  [NOVEL_MODEL_VARS['target']], axis=1).copy()
mice_cont_vars = list(NOVEL_MODEL_VARS['cont'])
mice_cont_vars.remove(LACTATE_VAR_NAME)
mice_cont_vars.remove(ALBUMIN_VAR_NAME)


reporter.report('Define stages of imputation, and the number of imputations '
                'needed at each stage')
imputation_stages = ImputationInfo()
imputation_stages.add_stage(
    description=('MICE for continuous variables (except lactate and albumin) '
                 'and non-binary discrete variables'),
    df=mice_df,
    variables_to_impute=list(mice_df.columns))
imputation_stages.add_stage(
    description='Non-binary discrete variables',
    df=df.drop(list(LACTATE_ALBUMIN_VARS) + [NOVEL_MODEL_VARS['target']],
               axis=1),
    variables_to_impute=list(multi_category_levels.keys()))
imputation_stages.add_stage(
    description='Lactate and albumin',
    df=df.drop(NOVEL_MODEL_VARS['target'], axis=1),
    variables_to_impute=[LACTATE_VAR_NAME, ALBUMIN_VAR_NAME])
# TODO: Save imputation_stages for later use


print(imputation_stages.__dict__)


# reporter.report('Loading data needed for train-test splitting')
# tt_splitter = load_object(os.path.join('outputs', 'train_test_splitter.pkl'))
#
#
# reporter.report('Running MICE')
# swm = SplitterWinsorMICE(df=mice_df,
#                          test_train_splitter=tt_splitter,
#                          target_variable_name=NOVEL_MODEL_VARS['target'],
#                          winsor_variables=mice_cont_vars,
#                          winsor_quantiles=(0.001, 0.999),
#                          winsor_include={'S01AgeOnArrival': (False, True),
#                                          'S03GlasgowComaScore': (False, False)},
#                          n_imputations=imputation_stages.n_imputations[0],
#                          binary_variables=binary_vars,
#                          n_burn_in=10,
#                          n_skip=3)


# TODO: Perform winsorization for lactate and albumin

# TODO: Use fit imputation models from train set for test set. This may not be
#   practical for MICE
