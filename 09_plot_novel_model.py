import os

import numpy as np

from utils.constants import FIGURES_OUTPUT_DIR, NOVEL_MODEL_OUTPUT_DIR
from utils.evaluate import LogisticScorer
from utils.io import load_object
from utils.plot.evaluate import (
    plot_calibration,
    plot_stratified_risk_distributions
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


reporter.last("Done.")
