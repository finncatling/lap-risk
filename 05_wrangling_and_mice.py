import copy
import os
from typing import Tuple, Dict

import pandas as pd

from utils.constants import (
    DATA_DIR,
    INTERNAL_OUTPUT_DIR,
    NOVEL_MODEL_OUTPUT_DIR,
    RANDOM_SEED
)
from utils.model.shared import flatten_model_var_dict
from utils.io import load_object, save_object
from utils.model.novel import (
    NOVEL_MODEL_VARS,
    MULTI_CATEGORY_LEVELS,
    LACTATE_VAR_NAME,
    ALBUMIN_VAR_NAME,
    LACTATE_ALBUMIN_VARS,
    preprocess_novel_pre_split,
    WINSOR_QUANTILES,
)
from utils.indications import (
    INDICATION_VAR_NAME,
    MISSING_IND_CATEGORY,
    get_indication_variable_names
)
from utils.impute import ImputationInfo, SplitterWinsorMICE
from utils.split import TrainTestSplitter
from utils.report import Reporter

reporter = Reporter()
reporter.title(
    "Wrangle NELA data in preparation for later input to the "
    "novel model. Perform MICE for continuous variables apart from "
    "lactate and albumin, and for binary variables"
)


reporter.report("Loading NELA data output from 04_consolidate_indications.py")
df: pd.DataFrame = pd.read_pickle(
    os.path.join(DATA_DIR, "04_output_df.pkl")
)


reporter.report("Finding names of indication variables")
indications = get_indication_variable_names(df.columns)


reporter.report("Removing variables not used in the novel model")
df = df[flatten_model_var_dict(NOVEL_MODEL_VARS) + indications]


reporter.report("Preparing dictionary of non-binary categorical variables")
multi_category_levels: Dict[str, Tuple] = copy.deepcopy(MULTI_CATEGORY_LEVELS)
multi_category_levels[INDICATION_VAR_NAME] = tuple(indications)
save_object(
    multi_category_levels,
    os.path.join(
        NOVEL_MODEL_OUTPUT_DIR,
        "05_multi_category_levels_with_indications.pkl"
    )
)


reporter.report("Doing pre-train-test-split data preprocessing")
df = preprocess_novel_pre_split(
    df,
    category_mapping={"S03ECG": {1.0: 0.0, 4.0: 1.0, 8.0: 1.0}},
    add_missingness_indicators_for=[LACTATE_VAR_NAME, ALBUMIN_VAR_NAME],
    indication_variable_name=INDICATION_VAR_NAME,
    indications=indications,
    missing_indication_value=MISSING_IND_CATEGORY,
    multi_category_levels=multi_category_levels,
)


reporter.report("Saving preprocessed data for later use")
df.to_pickle(os.path.join(DATA_DIR, "05_preprocessed_df.pkl"))


reporter.report(
    "Define stages of imputation, and the number of imputations needed at each "
    "stage"
)
imputation_stages = ImputationInfo()
imputation_stages.add_stage(
    description=(
        "MICE for continuous variables (except lactate and albumin) "	
        "and binary variables, then separate imputation of non-binary discrete "
        "variables"
    ),
    df=df.drop(
        list(LACTATE_ALBUMIN_VARS) + [NOVEL_MODEL_VARS["target"]], axis=1
    )
)
imputation_stages.add_stage(
    description="Lactate and albumin",
    df=df[[LACTATE_VAR_NAME, ALBUMIN_VAR_NAME]]
)


reporter.report("Saving imputation stage information for later use")
save_object(
    imputation_stages,
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "05_imputation_stages.pkl")
)


reporter.report(
    "Making DataFrame for use in MICE, and checking that there are no "
    "cases where all features are missing (these cases would be dropped by "
    "statsmodels MICEData, which could create problems with the "
    "post-imputation data reconstruction)"
)
mice_df = df.drop(
    list(multi_category_levels.keys()) + list(LACTATE_ALBUMIN_VARS), axis=1
).copy()
assert mice_df.shape[0] == mice_df.dropna(axis=0, how="all").shape[0]


reporter.report("Loading data needed for train-test splitting")
tt_splitter: TrainTestSplitter = load_object(
    os.path.join(INTERNAL_OUTPUT_DIR, "01_train_test_splitter.pkl")
)


reporter.report("Making list of binary variables for use in MICE")
binary_vars = list(
    set(NOVEL_MODEL_VARS["cat"]) -
    set(multi_category_levels.keys())
)


reporter.report("Making list of continuous variables for use in MICE")
mice_cont_vars = list(NOVEL_MODEL_VARS["cont"])
mice_cont_vars.remove(LACTATE_VAR_NAME)
mice_cont_vars.remove(ALBUMIN_VAR_NAME)


reporter.report("Running MICE")
swm = SplitterWinsorMICE(
    df=mice_df,
    train_test_splitter=tt_splitter,
    target_variable_name=NOVEL_MODEL_VARS["target"],
    cont_variables=mice_cont_vars,
    binary_variables=binary_vars,
    winsor_quantiles=WINSOR_QUANTILES,
    winsor_include={
        "S01AgeOnArrival": (False, True),
        "S03GlasgowComaScore": (False, False),
    },
    n_mice_imputations=imputation_stages.n_imputations[0],
    n_mice_burn_in=10,
    n_mice_skip=3,
    random_seed=RANDOM_SEED
)
swm.split_winsorize_mice()


reporter.report("Saving MICE imputations for later use")
save_object(
    swm,
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "05_splitter_winsor_mice.pkl")
)


reporter.last("Done.")
