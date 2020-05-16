import os
import numpy as np
from utils.plot import (plot_calibration, plot_stratified_risk_distributions,
                        plot_saver)
from utils.report import Reporter
from utils.constants import FIGURES_OUTPUT_DIR, CURRENT_MODEL_OUTPUT_DIR
from utils.io import load_object, make_directory


reporter = Reporter()
reporter.title('Make plots related to the current NELA emergency laparotomy '
               'mortality risk model')


reporter.report("Creating output dirs (if they don't already exist)")
make_directory(os.path.join(FIGURES_OUTPUT_DIR))


reporter.report('Loading results of model scoring')
scorer = load_object(os.path.join(CURRENT_MODEL_OUTPUT_DIR, 'scorer.pkl'))


reporter.report('Plotting calibration curves')
plot_saver(plot_calibration,
           p=scorer.p,
           calib_curves=scorer.calib_curves,
           curve_transparency=0.15,
           output_dir=FIGURES_OUTPUT_DIR,
           output_filename='current_model_calibration')


reporter.report('Plotting risk distributions')
plot_saver(plot_stratified_risk_distributions,
           y_true=np.array(scorer.y_true).flatten(),
           y_pred=np.array(scorer.y_pred).flatten(),
           output_dir=FIGURES_OUTPUT_DIR,
           output_filename='current_model_risk_distributions')


reporter.last('Done.')
