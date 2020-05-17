import copy
import os
from typing import Tuple, Dict

import pandas as pd

from utils.constants import DATA_DIR, STATS_OUTPUT_DIR, INDICATION_PREFIX
from utils.helpers import flatten_model_var_dict
from utils.io import make_directory
from utils.novel_model import NOVEL_MODEL_VARS, MULTI_CATEGORY_LEVELS
from utils.report import Reporter


reporter = Reporter()
reporter.title('Wrangle NELA data in preparation for later input into the '
               'novel model. The file includes multiple imputation of missing '
               'values for all variables apart from lactate and albumin')


reporter.report("Creating output dirs (if they don't already exist)")
make_directory(STATS_OUTPUT_DIR)


reporter.report('Loading manually-wrangled NELA data')
df = pd.read_pickle(os.path.join(
    DATA_DIR, 'df_after_univariate_wrangling_new_indications.pkl'))


reporter.report('Finding names of indication variables')
indications = [c for c in df.columns if INDICATION_PREFIX in c]


reporter.report('Removing variables not used in the novel model')
df = df[flatten_model_var_dict(NOVEL_MODEL_VARS) + indications]


reporter.report('Prepare details of discrete variables')
multi_category_levels: Dict[str, Tuple] = copy.deepcopy(MULTI_CATEGORY_LEVELS)
multi_category_levels['Indication'] = tuple(indications)
binary_vars = list(set(NOVEL_MODEL_VARS['cat']) -
                   set(multi_category_levels.keys()))
