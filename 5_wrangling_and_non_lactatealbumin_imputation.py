import copy
import os
from typing import Tuple, Dict

import pandas as pd

from utils.constants import (DATA_DIR, STATS_OUTPUT_DIR, INDICATION_PREFIX,
                             MISSING_IND_CATEGORY)
from utils.model.shared import flatten_model_var_dict
from utils.io import make_directory
from utils.model.novel import (NOVEL_MODEL_VARS, MULTI_CATEGORY_LEVELS,
                               LACTATE_VAR_NAME, ALBUMIN_VAR_NAME,
                               LACTATE_ALBUMIN_VARS,
                               preprocess_novel_pre_split)
from utils.impute import ImputationInfo
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


reporter.report('Check that there are no cases where all features are missing '
                '(these cases would be dropped by statsmodels MICEData, which '
                'could create problems with the post-imputation data '
                'reconstruction)')
assert df.shape[0] == df.dropna(axis=0, how='all').shape[0]


reporter.report('Define stages of imputation, and the number of imputations '
                'needed at each stage')
mice_df = df.drop(list(multi_category_levels.keys()) +
                  list(LACTATE_ALBUMIN_VARS) +
                  [NOVEL_MODEL_VARS['target']], axis=1)
imputation_stages = (
    ImputationInfo(description=('MICE for continuous variables (except lactate '
                                'and albumin) and non-binary discrete '
                                'variables'),
                   df=mice_df,
                   variables_to_impute=tuple(mice_df.columns)),
)

print(imputation_stages[0].__dict__)

# n_imputations = {
#     'mice': determine_n_imputations(df.drop(
#         list(VARS_FOR_MISSINGNESS_INDICATORS) + list(multi_category_levels.keys()), axis=1)),
#     'no_lactate_albumin': determine_n_imputations(df.drop(
#         list(VARS_FOR_MISSINGNESS_INDICATORS), axis=1)),
#     'all_variables': determine_n_imputations(df)}





# TODO: Class to handle preprocessing loop for each train-test split

# TODO: Use fit imputation models from train set for test set. For MICE, will
#    need to run the burn-in stages again

# TODO: Keep track of per-case-per-variable missing data indices, so that
