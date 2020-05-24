import os
from typing import Dict, Tuple

import pandas as pd

from utils.constants import (DATA_DIR, NOVEL_MODEL_OUTPUT_DIR, STATS_OUTPUT_DIR,
                             RANDOM_SEED)
from utils.impute import ImputationInfo, SplitterWinsorMICE, CategoricalImputer
from utils.io import make_directory, load_object, save_object
from utils.report import Reporter


reporter = Reporter()
reporter.title('Impute non-binary discrete variables')


reporter.report("Creating output dirs (if they don't already exist)")
make_directory(STATS_OUTPUT_DIR)
make_directory(NOVEL_MODEL_OUTPUT_DIR)


reporter.report('Loading previous analysis outputs needed for imputation')
df = pd.read_pickle(
    os.path.join(DATA_DIR, 'df_preprocessed_for_novel_pre_split.pkl'))
imp_stages: ImputationInfo = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, 'imputation_stages.pkl'))
swm: SplitterWinsorMICE = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, 'splitter_winsor_mice.pkl'))
multi_category_levels: Dict[str, Tuple] = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR,
                 'multi_category_levels_with_indications.pkl'))


reporter.report('Fitting imputers for non-binary categorical variables')
cat_imputer = CategoricalImputer(
    df=df,
    splitter_winsor_mice=swm,
    cat_vars=list(multi_category_levels.keys()),
    n_imputations_per_mice=imp_stages.multiple_of_previous_n_imputations[1],
    random_seed=RANDOM_SEED)
cat_imputer.tts.n_splits = 2  # TODO: remove this testing line later
cat_imputer.fit()


reporter.report('Imputing a dataframe')
cat_imp_df_0 = cat_imputer.impute_X_df('train', 1, 3, 0)


reporter.report('Imputing another dataframe')
cat_imp_df_1 = cat_imputer.impute_X_df('train', 1, 3, 1)


reporter.report('Inspecting')
print(cat_imp_df_0.columns)
print('Shapes:', cat_imp_df_0.shape, cat_imp_df_1.shape)
print('Dropna shape:'. cat_imp_df.dropna(how='any').shape)
print(cat_imp_df_0.head())
print(cat_imp_df_1.head())
print(cat_imp_df_0.equals(cat_imp_df_1))


reporter.report('Saving temporary cat imputer')
save_object(cat_imputer, os.path.join(NOVEL_MODEL_OUTPUT_DIR,
                                      'temp_cat_imputer.pkl'))


# TODO: Save summary stats (including those from MICE) for external use
# TODO: Perform winsorization for lactate and albumin


reporter.last('Done.')
