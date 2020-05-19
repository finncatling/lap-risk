import os

from utils.io import make_directory, load_object
from utils.constants import (INTERNAL_OUTPUT_DIR, NOVEL_MODEL_OUTPUT_DIR,
                             STATS_OUTPUT_DIR)
from utils.report import Reporter


reporter = Reporter()
reporter.title('Impute non-binary discrete variables')


reporter.report("Creating output dirs (if they don't already exist)")
make_directory(STATS_OUTPUT_DIR)
make_directory(NOVEL_MODEL_OUTPUT_DIR)


reporter.report('Loading previous analysis outputs needed for imputation')
imputation_stages = load_object(os.path.join(NOVEL_MODEL_OUTPUT_DIR,
                                             'imputation_stages.pkl'))
tt_splitter = load_object(os.path.join(INTERNAL_OUTPUT_DIR,
                                       'train_test_splitter.pkl'))
swm = load_object(os.path.join(NOVEL_MODEL_OUTPUT_DIR,
                               'splitter_winsor_mice.pkl'))



# TODO: Non-binary discrete variable imputation. Use fit imputation models from
#  train set for test set.

# TODO: Save summary stats (including those from MICE) for external use

# TODO: Perform winsorization for lactate and albumin
