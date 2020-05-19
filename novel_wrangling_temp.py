import copy
import os
from typing import Tuple, Dict

import pandas as pd

from utils.constants import (DATA_DIR, NOVEL_MODEL_OUTPUT_DIR,
                             INDICATION_PREFIX,
                             MISSING_IND_CATEGORY)
from utils.io import make_directory, save_object
from utils.model.novel import (NOVEL_MODEL_VARS, MULTI_CATEGORY_LEVELS,
                               LACTATE_VAR_NAME, ALBUMIN_VAR_NAME,
                               preprocess_novel_pre_split)
from utils.model.shared import flatten_model_var_dict
from utils.report import Reporter

INDICATION_VAR_NAME = 'Indication'

reporter = Reporter()
reporter.title('Wrangle NELA data in preparation for later input to the '
               'novel model. Perform MICE for continuous variables apart from '
               'lactate and albumin, and for binary discrete variables')

reporter.report("Creating output dirs (if they don't already exist)")
make_directory(NOVEL_MODEL_OUTPUT_DIR)

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

reporter.report('Doing pre-train-test-split data preprocessing')
df = preprocess_novel_pre_split(
    df,
    category_mapping={'S03ECG': {1.0: 0.0, 4.0: 1.0, 8.0: 1.0}},
    add_missingness_indicators_for=[LACTATE_VAR_NAME, ALBUMIN_VAR_NAME],
    indication_variable_name=INDICATION_VAR_NAME,
    indications=indications,
    missing_indication_value=MISSING_IND_CATEGORY,
    multi_category_levels=multi_category_levels)

reporter.report('Saving preprocessed data for later use')
df.to_pickle(os.path.join(DATA_DIR, 'df_preprocessed_for_novel_pre_split.pkl'))

reporter.report('Saving levels of categorical variables (with indications '
                'added) for later use')
save_object(multi_category_levels, os.path.join(
    NOVEL_MODEL_OUTPUT_DIR, 'multi_category_levels_with_indications.pkl'))

reporter.last('Done.')
