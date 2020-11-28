import os

from utils.constants import (
    CALIB_GAM_LAM_CANDIDATES,
    CALIB_GAM_N_SPLINES,
    CURRENT_MODEL_OUTPUT_DIR,
    STATS_OUTPUT_DIR
)
from utils.evaluate import LogisticScorer, score_logistic_predictions
from utils.io import load_object, save_object
from utils.model.current import CurrentModelDataIO
from utils.report import Reporter


reporter = Reporter()
reporter.title(
    "Evaluate predicted risks from the re-fit current NELA emergency "
    "laparotomy mortality risk model."
)


reporter.report("Loading current model data IO")
current_model_io: CurrentModelDataIO = load_object(
    os.path.join(CURRENT_MODEL_OUTPUT_DIR, "02_alt_current_model_data_io.pkl")
)


reporter.report("Scoring model performance")
scorer = LogisticScorer(
    y_true=current_model_io.y_test,
    y_pred=current_model_io.y_pred,
    scorer_function=score_logistic_predictions,
    n_splits=current_model_io.tts.n_splits,
    calibration_n_splines=CALIB_GAM_N_SPLINES,
    calibration_lam_candidates=CALIB_GAM_LAM_CANDIDATES,
)
scorer.calculate_scores()
reporter.first("Scores with median as point estimate:")
scorer.print_scores(dec_places=3, point_estimate='median')
reporter.first("Scores with fold 0 as point estimate:")
scorer.print_scores(dec_places=3, point_estimate='split0')


reporter.first("Saving model scorer for later use")
save_object(
    scorer,
    os.path.join(CURRENT_MODEL_OUTPUT_DIR, "02_alt_current_model_scorer.pkl")
)


reporter.report("Saving summary statistics for external use")
current_model_stats = {
    "scores": scorer.scores,
    "calib_p": scorer.p,
    "calib_curves": scorer.calib_curves,
    "calib_n_splines": scorer.calib_n_splines,
    "calib_lam_candidates": scorer.calib_lam_candidates,
    "calib_best_lams": scorer.calib_lams,
}
save_object(
    current_model_stats,
    os.path.join(STATS_OUTPUT_DIR, "02_alt_current_model_score_stats.pkl"),
)


reporter.last("Done.")
