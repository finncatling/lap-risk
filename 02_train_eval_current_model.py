import os
from datetime import datetime

from utils.constants import (
    RANDOM_SEED,
    INTERNAL_OUTPUT_DIR,
    STATS_OUTPUT_DIR,
    CURRENT_MODEL_OUTPUT_DIR,
    CALIB_GAM_N_SPLINES,
    CALIB_GAM_LAM_CANDIDATES,
)
from utils.data_check import load_nela_data_and_sanity_check
from utils.evaluate import LogisticScorer, score_logistic_predictions
from utils.io import load_object, save_object
from utils.model.current import (
    preprocess_current,
    CurrentModel,
    WINSOR_THRESHOLDS,
    CURRENT_MODEL_VARS,
    CENTRES,
)
from utils.model.shared import flatten_model_var_dict
from utils.report import Reporter
from utils.split import TrainTestSplitter, drop_incomplete_cases


reporter = Reporter()
reporter.title(
    "Re-fit current NELA emergency laparotomy mortality risk "
    "model on the different train folds, and evaluate the models "
    "obtained on the corresponding test folds"
)


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


reporter.report("Beginning train-test splitting and model fitting")
current_model = CurrentModel(
    df,
    train_test_splitter=tt_splitter,
    target_variable_name=CURRENT_MODEL_VARS["target"],
    random_seed=RANDOM_SEED,
)
current_model.split_train_predict()


reporter.report("Saving CurrentModel for later use")
save_object(
    current_model,
    os.path.join(CURRENT_MODEL_OUTPUT_DIR, "02_current_model.pkl")
)


reporter.report("Scoring model performance")
scorer = LogisticScorer(
    y_true=current_model.y_test,
    y_pred=current_model.y_pred,
    scorer_function=score_logistic_predictions,
    calibration_n_splines=CALIB_GAM_N_SPLINES,
    calibration_lam_candidates=CALIB_GAM_LAM_CANDIDATES,
)
scorer.calculate_scores()
print("")
scorer.print_scores(dec_places=3)


reporter.first("Saving ModelScorer for later use")
save_object(
    scorer,
    os.path.join(CURRENT_MODEL_OUTPUT_DIR, "02_current_model_scorer.pkl")
)


reporter.report("Saving summary statistics for external use")
current_model_stats = {
    "start_datetime": datetime.fromtimestamp(reporter.timer.start_time),
    "train_fold_stats": current_model.split_stats,
    "model_features": current_model.features,
    "model_coefficients": current_model.coefficients,
    "scores": scorer.scores,
    "calib_p": scorer.p,
    "calib_curves": scorer.calib_curves,
    "calib_n_splines": scorer.calib_n_splines,
    "calib_lam_candidates": scorer.calib_lam_candidates,
    "calib_best_lams": scorer.calib_lams,
}
save_object(
    current_model_stats,
    os.path.join(STATS_OUTPUT_DIR, "02_current_model_stats.pkl"),
)


reporter.last("Done.")
