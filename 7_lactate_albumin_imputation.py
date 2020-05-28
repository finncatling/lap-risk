from typing import Dict, Tuple
import os
import pandas as pd

from utils.report import Reporter
from utils.constants import DATA_DIR, NOVEL_MODEL_OUTPUT_DIR
from utils.io import make_directory, save_object, load_object
from utils.impute import (ImputationInfo, CategoricalImputer,
                          LactateAlbuminImputer)
from utils.model.novel import (ALBUMIN_VAR_NAME, NOVEL_MODEL_VARS,
                               WINSOR_QUANTILES, INDICATION_VAR_NAME)
from utils.model.albumin import albumin_model_factory, GammaTransformer


reporter = Reporter()
reporter.title('Fit albumin imputation models')
# reporter.title('Impute lactate and albumin')


reporter.report("Creating output dirs (if they don't already exist)")
make_directory(NOVEL_MODEL_OUTPUT_DIR)


reporter.report('Loading previous analysis outputs needed for imputation')
df = pd.read_pickle(
    os.path.join(DATA_DIR, 'df_preprocessed_for_novel_pre_split.pkl'))
imp_stages: ImputationInfo = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, 'imputation_stages.pkl'))
cat_imputer: CategoricalImputer = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, 'categorical_imputer.pkl'))
multi_category_levels: Dict[str, Tuple] = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR,
                 'multi_category_levels_with_indications.pkl'))


reporter.report('Fitting imputers for albumin')
alb_imputer = LactateAlbuminImputer(
    df=df.loc[:, [ALBUMIN_VAR_NAME, NOVEL_MODEL_VARS['target']]],
    categorical_imputer=cat_imputer,
    imputation_target=ALBUMIN_VAR_NAME,
    imputation_model_factory=albumin_model_factory,
    winsor_quantiles=WINSOR_QUANTILES,
    transformer=GammaTransformer,
    transformer_args={},
    multi_cat_vars=multi_category_levels,
    indication_var_name=INDICATION_VAR_NAME)
alb_imputer.tts.n_splits = 2  # TODO: Remove this testing line
alb_imputer.fit()
print(alb_imputer.__dict__)


reporter.report('Saving draft albumin imputer for later use')
save_object(alb_imputer, os.path.join(NOVEL_MODEL_OUTPUT_DIR,
                                      'draft_albumin_imputer.pkl'))


reporter.last('Done.')

# TODO: Save summary stats (including those from MICE) for external use
# TODO: Finish this script
