import os

import numpy as np

from utils.constants import (
    FIGURES_OUTPUT_DIR,
    CURRENT_MODEL_OUTPUT_DIR,
    NOVEL_MODEL_OUTPUT_DIR
)
from utils.evaluate import LogisticScorer
from utils.io import load_object
from utils.model.novel import NovelModel
from utils.plot.evaluate import (
    plot_calibration,
    plot_calibration_subplots,
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


reporter.report("Loading results of current model scoring")
current_scorer: LogisticScorer = load_object(
    os.path.join(CURRENT_MODEL_OUTPUT_DIR, "02_current_model_scorer.pkl")
)


for scorer_name in ('novel_model', '1_novel_model_samples'):
    reporter.report(f"Loading results of {scorer_name} scoring")
    scorer: LogisticScorer = load_object(
        os.path.join(NOVEL_MODEL_OUTPUT_DIR, f"08_{scorer_name}_scorer.pkl")
    )

    for hist_switch, hist_suffix in ((False, ''), (True, '_histograms')):
        reporter.report("Plotting individual calibration curves")
        plot_saver(
            plot_calibration,
            p=scorer.p,
            calib_curves=scorer.calib_curves,
            curve_transparency=0.15,
            plot_histograms=hist_switch,
            y_pred=np.concatenate(scorer.y_pred),
            output_dir=FIGURES_OUTPUT_DIR,
            output_filename=f"09_{scorer_name}_calibration{hist_suffix}",
        )

        reporter.report("Plotting calibration subplots figure")
        plot_saver(
            plot_calibration_subplots,
            p=scorer.p,
            calib_curves=(current_scorer.calib_curves, scorer.calib_curves),
            model_names=('Re-fitted NELA calculator', 'Novel model'),
            curve_transparency=0.15,
            plot_histograms=hist_switch,
            y_preds=(
                np.concatenate(current_scorer.y_pred),
                np.concatenate(scorer.y_pred)
            ),
            output_dir=FIGURES_OUTPUT_DIR,
            output_filename=(
                f"09_current_vs_{scorer_name}_calibration{hist_suffix}"
            ),
        )

    reporter.report("Plotting risk distributions")
    plot_saver(
        plot_stratified_risk_distributions,
        y_true=np.hstack(scorer.y_true),
        y_pred=np.hstack(scorer.y_pred),
        output_dir=FIGURES_OUTPUT_DIR,
        output_filename=f"09_{scorer_name}_risk_distributions",
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
    output_filename="09_novel_model_2_example_risk_distributions_simpler",
)
plot_saver(
    plot_example_risk_distributions,
    y_pred_samples=y_pred_samples,
    patient_indices=(9942, 6530, 3094),
    kde_bandwidths=(0.008, 0.012, 0.04),
    output_dir=FIGURES_OUTPUT_DIR,
    output_filename="09_novel_model_3_example_risk_distributions_simpler",
)


reporter.last("Done.")
