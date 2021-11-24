import os
import numpy as np
from utils.plot.evaluate import (
    plot_calibration,
    plot_stratified_risk_distributions
)
from utils.plot.helpers import plot_saver
from utils.report import Reporter
from utils.constants import FIGURES_OUTPUT_DIR, CURRENT_MODEL_OUTPUT_DIR
from utils.io import load_object
from utils.evaluate import LogisticScorer


reporter = Reporter()
reporter.title(
    "Make plots related to the current NELA emergency laparotomy "
    "mortality risk model"
)


reporter.report("Loading results of model scoring")
scorer: LogisticScorer = load_object(
    os.path.join(CURRENT_MODEL_OUTPUT_DIR, "02_current_model_scorer.pkl")
)


reporter.report("Plotting calibration curves")
for hist_switch, hist_suffix in ((False, ''), (True, '_histograms')):
    plot_saver(
        plot_calibration,
        p=scorer.p,
        calib_curves=scorer.calib_curves,
        curve_transparency=0.15,
        plot_histograms=hist_switch,
        y_pred=np.concatenate(scorer.y_pred),
        output_dir=FIGURES_OUTPUT_DIR,
        output_filename=f"03_current_model_calibration{hist_suffix}",
    )


reporter.report("Plotting risk distributions")
plot_saver(
    plot_stratified_risk_distributions,
    y_true=np.hstack(scorer.y_true),
    y_pred=np.hstack(scorer.y_pred),
    output_dir=FIGURES_OUTPUT_DIR,
    output_filename="03_current_model_risk_distributions",
)


reporter.last("Done.")
