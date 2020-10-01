import os
from datetime import datetime

from utils.constants import (
    RANDOM_SEED,
    STATS_OUTPUT_DIR,
    INTERNAL_OUTPUT_DIR
)
from utils.data_check import load_nela_data_and_sanity_check
from utils.io import make_directory
from utils.io import save_object
from utils.model.current import CURRENT_MODEL_VARS
from utils.model.shared import flatten_model_var_dict
from utils.report import Reporter
from utils.split import TrainTestSplitter


reporter = Reporter()
reporter.title(
    "Derive case indices needed to repeatedly split NELA data "
    "into train and test folds"
)


reporter.report("Creating external outputs dir (if it doesn't already exist)")
make_directory(os.path.join(STATS_OUTPUT_DIR))


reporter.report("Loading manually-wrangled NELA data")
df = load_nela_data_and_sanity_check()


reporter.report("Performing train-test split")
tt_splitter_args = {
    "split_variable_name": "HospitalId.anon",
    "test_fraction": 0.2,
    "n_splits": 120,
}
tt_splitter = TrainTestSplitter(
    df=df,
    current_nela_model_vars=flatten_model_var_dict(CURRENT_MODEL_VARS),
    random_seed=RANDOM_SEED,
    **tt_splitter_args
)
tt_splitter.split()


reporter.report("Saving TrainTestSplitter for use later in analysis")
save_object(
    tt_splitter,
    os.path.join(INTERNAL_OUTPUT_DIR, "train_test_splitter.pkl")
)


reporter.report("Saving summary statistics for external use")
tt_split_stats = {
    "start_datetime": datetime.fromtimestamp(reporter.timer.start_time),
    "n_institutions": tt_splitter.n_institutions,
    "n_train_institutions": tt_splitter.n_train_institutions,
    "n_test_institutions": tt_splitter.n_test_institutions,
    "drop_stats": tt_splitter.drop_stats,
    "split_stats": tt_splitter.split_stats,
    **tt_splitter_args,
}
save_object(
    tt_split_stats,
    os.path.join(STATS_OUTPUT_DIR, "1_train_test_split_stats.pkl")
)


reporter.last("Done.")
