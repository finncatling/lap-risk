from typing import Dict, Tuple
import os
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

from utils.report import Reporter
from utils.constants import DATA_DIR, NOVEL_MODEL_OUTPUT_DIR, RANDOM_SEED
from utils.io import make_directory, save_object, load_object
from utils.impute import CategoricalImputer, LactateAlbuminImputer
from utils.model.novel import (LACTATE_VAR_NAME, NOVEL_MODEL_VARS,
                               WINSOR_QUANTILES, INDICATION_VAR_NAME)
from utils.model.albumin import albumin_model_factory


reporter = Reporter()
reporter.title('Fit lactate imputation models')


reporter.report("Creating output dirs (if they don't already exist)")
make_directory(NOVEL_MODEL_OUTPUT_DIR)


reporter.report('Loading previous analysis outputs needed for imputation')
df = pd.read_pickle(
    os.path.join(DATA_DIR, 'df_preprocessed_for_novel_pre_split.pkl'))
cat_imputer: CategoricalImputer = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, 'categorical_imputer.pkl'))
multi_category_levels: Dict[str, Tuple] = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR,
                 'multi_category_levels_with_indications.pkl'))


reporter.report('Fitting imputers for lactate')
lac_imputer = LactateAlbuminImputer(
    df=df.loc[:, [LACTATE_VAR_NAME, NOVEL_MODEL_VARS['target']]],
    categorical_imputer=cat_imputer,
    imputation_target=LACTATE_VAR_NAME,
    # TODO: Make lactate_model_factory
    imputation_model_factory=lactate_model_factory,
    winsor_quantiles=WINSOR_QUANTILES,
    transformer=QuantileTransformer,
    transformer_args={'output_distribution': 'normal',
                      'random_state': RANDOM_SEED},
    multi_cat_vars=multi_category_levels,
    indication_var_name=INDICATION_VAR_NAME)
lac_imputer.tts.n_splits = 2  # TODO: Remove this testing line
lac_imputer.fit()
print(lac_imputer.__dict__)


reporter.report('Saving draft lactate imputer for later use')
save_object(lac_imputer, os.path.join(NOVEL_MODEL_OUTPUT_DIR,
                                      'draft_lactate_imputer.pkl'))


reporter.last('Done.')

# TODO: Save summary stats (including those from MICE) for external use
