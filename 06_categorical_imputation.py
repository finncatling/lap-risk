import os
from typing import Dict, Tuple

import pandas as pd

from utils.constants import (
    DATA_DIR,
    NOVEL_MODEL_OUTPUT_DIR,
    RANDOM_SEED,
)
from utils.impute import SplitterWinsorMICE, CategoricalImputer
from utils.model.novel import LACTATE_ALBUMIN_VARS
from utils.io import load_object, save_object
from utils.report import Reporter


reporter = Reporter()
reporter.title("Impute non-binary discrete variables")


reporter.report("Loading previous analysis outputs needed for imputation")
df = pd.read_pickle(
    os.path.join(DATA_DIR, "05_preprocessed_df.pkl")
)
swm: SplitterWinsorMICE = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "05_splitter_winsor_mice.pkl")
)
multi_category_levels: Dict[str, Tuple] = load_object(
    os.path.join(
        NOVEL_MODEL_OUTPUT_DIR,
        "05_multi_category_levels_with_indications.pkl"
    )
)


reporter.report("Fitting imputers for non-binary categorical variables")
cat_imputer = CategoricalImputer(
    df=df.drop(list(LACTATE_ALBUMIN_VARS), axis=1),
    splitter_winsor_mice=swm,
    cat_vars=list(multi_category_levels.keys()),
    random_seed=RANDOM_SEED,
)
cat_imputer.impute()


reporter.report("Saving categorical imputer for later use")
save_object(
    cat_imputer,
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "06_categorical_imputer.pkl")
)


reporter.last("Done.")
