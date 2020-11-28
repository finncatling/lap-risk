import os
from progressbar import progressbar as pb

from utils.constants import INTERNAL_OUTPUT_DIR, CURRENT_MODEL_FEATHER_DIR
from utils.data_check import load_nela_data_and_sanity_check
from utils.io import load_object, make_directory
from utils.model.current import (
    CENTRES,
    CURRENT_MODEL_VARS,
    WINSOR_THRESHOLDS,
    preprocess_current)
from utils.model.shared import flatten_model_var_dict
from utils.report import Reporter
from utils.split import TrainTestSplitter, drop_incomplete_cases, Splitter


reporter = Reporter()
reporter.title(
    "Preprocess data ready for fitting the current NELA model, "
    "then export it to .feather ready for import to R in a later "
    "script."
)


reporter.report("Creating output directories (if they don't already exist)")
make_directory(CURRENT_MODEL_FEATHER_DIR)
make_directory(os.path.join(CURRENT_MODEL_FEATHER_DIR, 'train'))
make_directory(os.path.join(CURRENT_MODEL_FEATHER_DIR, 'test'))
make_directory(os.path.join(CURRENT_MODEL_FEATHER_DIR, 'y_pred'))


reporter.report("Loading manually-wrangled NELA data")
df = load_nela_data_and_sanity_check()


reporter.report("Removing unused variables")
df = df[flatten_model_var_dict(CURRENT_MODEL_VARS)]


reporter.report("Dropping cases which are incomplete for the models' variables")
df, _ = drop_incomplete_cases(df)


reporter.report("Preparing list of variables for binarization")
"""
GCS is exempt from binarization as it is binned in a separate function. ASA is
exempt as it is only used in a later interaction term.
"""
binarize_vars = list(CURRENT_MODEL_VARS["cat"])
binarize_vars.remove("S03GlasgowComaScore")
binarize_vars.remove("S03ASAScore")


reporter.report("Preparing list of variables for quadratic transformation")
"""
Sodium is exempt as it undergoes a customised transformation. We apply the
quadratic transformation to creatinine and urea after they are log-transformed.
"""
quadratic_vars = list(CURRENT_MODEL_VARS["cont"])
quadratic_vars.remove("S03Sodium")
for original, logged in (
    ("S03SerumCreatinine", "logcreat"),
    ("S03Urea", "logurea")
):
    quadratic_vars.remove(original)
    quadratic_vars.append(logged)


reporter.report("Preprocessing data")
"""
Our preprocessing code was originally designed to allow preprocessing after
splitting the data. However, the preprocessing steps do not 'leak' any
information from the test-fold cases, e.g. they don't use any summary statistics
derived from the data for use in transforming the variables. This means that
we can speed up our code by running the preprocessing prior to the loop wherein
the data are repeatedly split and the model retrained.  
"""
preprocessed_df, _ = preprocess_current(
    df,
    quadratic_vars=quadratic_vars,
    winsor_thresholds=WINSOR_THRESHOLDS,
    centres=CENTRES,
    binarize_vars=binarize_vars,
    label_binarizers=None,
)


reporter.report("Loading data needed for train-test splitting")
tt_splitter: TrainTestSplitter = load_object(
    os.path.join(INTERNAL_OUTPUT_DIR, "01_train_test_splitter.pkl")
)


reporter.report("Saving each train-test split as a .feather")
splitter = Splitter(
    preprocessed_df,
    train_test_splitter=tt_splitter,
    target_variable_name=CURRENT_MODEL_VARS["target"]
)
for i in pb(range(splitter.tts.n_splits), prefix="Split iteration"):
    X_train, y_train, X_test, y_test = splitter._split(i)



reporter.last("Done.")
