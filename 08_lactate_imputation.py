# TODO: Finish this script
raise NotImplementedError

import os
from typing import Dict, Tuple

import pandas as pd

from utils.constants import DATA_DIR, NOVEL_MODEL_OUTPUT_DIR, RANDOM_SEED
from utils.impute import CategoricalImputer, LactateAlbuminImputer
from utils.indications import INDICATION_VAR_NAME
from utils.io import save_object, load_object
from utils.model.novel import (
    LACTATE_VAR_NAME,
    NOVEL_MODEL_VARS,
    WINSOR_QUANTILES,
)
from utils.report import Reporter


reporter = Reporter()
reporter.title("Fit lactate imputation models")


reporter.report("Loading previous analysis outputs needed for imputation")
df = pd.read_pickle(os.path.join(DATA_DIR, "05_preprocessed_df.pkl"))
cat_imputer: CategoricalImputer = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "06_categorical_imputer.pkl")
)
multi_category_levels: Dict[str, Tuple] = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR,
                 "05_multi_category_levels_with_indications.pkl")
)


reporter.report("Fitting imputers for lactate")
lac_imputer = LactateAlbuminImputer(
    df=df.loc[:, [LACTATE_VAR_NAME, NOVEL_MODEL_VARS["target"]]],
    categorical_imputer=cat_imputer,
    lacalb_variable_name=LACTATE_VAR_NAME,
    # TODO: Make lactate_model_factory
    imputation_model_factory=lactate_model_factory,
    winsor_quantiles=WINSOR_QUANTILES,
    multi_cat_vars=multi_category_levels,
    indication_var_name=INDICATION_VAR_NAME,
    random_seed=RANDOM_SEED
)
lac_imputer.tts.n_splits = 2  # TODO: Remove this testing line
lac_imputer.fit()
print(lac_imputer.__dict__)


reporter.report("Saving draft lactate imputer for later use")
save_object(
    lac_imputer,
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "draft_lactate_imputer.pkl")
)


reporter.last("Done.")

# TODO: Save summary stats (including those from MICE) for external use
