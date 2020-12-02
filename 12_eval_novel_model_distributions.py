import os
import numpy as np
from progressbar import progressbar as pb

from utils.constants import (
    CALIB_GAM_N_SPLINES,
    NOVEL_MODEL_OUTPUT_DIR
)
from utils.evaluate import LogisticScorer, score_logistic_predictions
from utils.io import load_object, save_object
from utils.model.novel import NovelModel
from utils.report import Reporter

reporter = Reporter()
reporter.title(
    "Using pre-fitted novel model, generate predicted risk *distributions* and "
    "evaluate these."
)


reporter.report(f"Loading pretrained novel model")
novel_model: NovelModel = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "08_novel_model.pkl"))


reporter.report(f"Generating predicted risk distributions.")
y_obs, y_preds = [], []
for split_i in pb(
    range(novel_model.cat_imputer.tts.n_splits),
    prefix="Split iteration"
):
    y_ob, y_pred = novel_model.get_observed_and_predicted(
        fold_name='test',
        split_i=split_i,
        n_samples_per_imp_i=5
    )
    # NB. Shape of y_ob is (n_patients,)
    # NB. Shape of y_pred is (n_predicted_probabilities, n_patients)
    y_obs.append(np.repeat(y_ob, y_pred.shape[0]))
    y_preds.append(y_pred.flatten(order='F'))


reporter.report(f"Scoring novel model performance.")
scorer = LogisticScorer(
    y_true=y_obs,
    y_pred=y_preds,
    scorer_function=score_logistic_predictions,
    n_splits=novel_model.cat_imputer.tts.n_splits,
    calibration_n_splines=CALIB_GAM_N_SPLINES,
    calibration_lam_candidates=np.array(0.00132571, 0.00175751))
scorer.calculate_scores()
reporter.first("Scores with median as point estimate:")
scorer.print_scores(dec_places=3, point_estimate='median')
reporter.first("Scores with split 0 as point estimate:")
scorer.print_scores(dec_places=3, point_estimate='split0')


reporter.first("Saving model scorer for later use")
save_object(
    scorer,
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "12_novel_model_samples_scorer.pkl"))


reporter.last("Done.")
