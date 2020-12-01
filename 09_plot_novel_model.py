import os

import numpy as np

from utils.constants import FIGURES_OUTPUT_DIR, NOVEL_MODEL_OUTPUT_DIR
from utils.evaluate import LogisticScorer
from utils.io import load_object
from utils.model.novel import NovelModel
from utils.plot.evaluate import (
    plot_calibration,
    plot_stratified_risk_distributions,
    plot_example_risk_distributions
)
from utils.plot.helpers import plot_saver
from utils.report import Reporter


reporter = Reporter()
reporter.title(
    "Make plots related to the novel emergency laparotomy "
    "mortality risk model"
)


reporter.report("Loading results of model scoring")
scorer: LogisticScorer = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "08_novel_model_scorer.pkl")
)


reporter.report("Plotting calibration curves")
plot_saver(
    plot_calibration,
    p=scorer.p,
    calib_curves=scorer.calib_curves,
    curve_transparency=0.15,
    output_dir=FIGURES_OUTPUT_DIR,
    output_filename="09_novel_model_calibration",
)


reporter.report("Plotting risk distributions")
plot_saver(
    plot_stratified_risk_distributions,
    y_true=np.hstack(scorer.y_true),
    y_pred=np.hstack(scorer.y_pred),
    output_dir=FIGURES_OUTPUT_DIR,
    output_filename="09_novel_model_risk_distributions",
)


reporter.report("Loading trained novel model")
novel_model: NovelModel = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "08_novel_model.pkl"))


reporter.report("Predicting risk distributions on train fold 0")
_, y_pred_samples = novel_model.get_observed_and_predicted('test', 0, 50)


reporter.report("Plotting predicted risk distributions for example patients")
plot_saver(
    plot_example_risk_distributions,
    y_pred_samples=y_pred_samples,
    patient_indices=(9942, 3094),
    kde_bandwidths=(0.008, 0.04),
    output_dir=FIGURES_OUTPUT_DIR,
    output_filename="09_novel_model_2_example_risk_distributions",
)
plot_saver(
    plot_example_risk_distributions,
    y_pred_samples=y_pred_samples,
    patient_indices=(9942, 6530, 3094),
    kde_bandwidths=(0.008, 0.012, 0.04),
    output_dir=FIGURES_OUTPUT_DIR,
    output_filename="09_novel_model_3_example_risk_distributions",
)



reporter.last("Done.")
