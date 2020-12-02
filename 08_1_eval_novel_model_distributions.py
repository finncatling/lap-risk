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


reporter.report("Load calibration GAM best lams from point-prediction novel "
                "model scorer")
point_prediction_novel_model_scorer: LogisticScorer = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "08_novel_model_scorer.pkl"))
best_lams = np.array([nested_lam[0][0] for nested_lam in
                      point_prediction_novel_model_scorer.calib_lams])


reporter.report("Restricting calibration GAM fitting for risk distributions to "
                "the 3 most-frequently-chosen lams during previous novel model "
                "fiting, as calibration GAM fitting is slow with such large "
                "datasets.")
unique_lams, counts_per_unique_lam = np.unique(best_lams, return_counts=True)
top_3_lams = unique_lams[counts_per_unique_lam.argsort()][-3:]
reporter.report(f"Top 3 lams are f{top_3_lams}")


reporter.report("Generating predicted risk distributions.")
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


reporter.report(f"{y_pred.shape[0]} samples form each risk distribution")


reporter.report(f"Scoring novel model performance.")
scorer = LogisticScorer(
    y_true=y_obs,
    y_pred=y_preds,
    scorer_function=score_logistic_predictions,
    n_splits=novel_model.cat_imputer.tts.n_splits,
    calibration_n_splines=CALIB_GAM_N_SPLINES,
    calibration_lam_candidates=top_3_lams
)
scorer.calculate_scores()
reporter.first("Scores with median as point estimate:")
scorer.print_scores(dec_places=3, point_estimate='median')
reporter.first("Scores with split 0 as point estimate:")
scorer.print_scores(dec_places=3, point_estimate='split0')


reporter.first("Saving model scorer for later use")
save_object(
    scorer,
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "08_1_novel_model_samples_scorer.pkl"))


reporter.last("Done.")
