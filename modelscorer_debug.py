import os
from utils.io import load_object, make_directory
from utils.constants import (FIGURES_OUTPUT_DIR, CALIB_GAM_N_SPLINES,
                             CALIB_GAM_LAM_CANDIDATES)
from utils.report import Reporter
from utils.evaluate import ModelScorer, plot_calibration

reporter = Reporter()


reporter.first("Creating external outputs dir (if it doesn't already exist)")
make_directory(os.path.join(FIGURES_OUTPUT_DIR))


reporter.report('Loading SplitterTrainerPredictor for use in model evaluation')
stp = load_object(os.path.join('outputs', 'splitter_trainer_predictor.pkl'))


reporter.report('Scoring model performance')
scorer = ModelScorer(y_true=stp.y_test,
                     y_pred=stp.y_pred,
                     calibration_n_splines=CALIB_GAM_N_SPLINES,
                     calibration_lam_candidates=CALIB_GAM_LAM_CANDIDATES)
scorer.calculate_scores()
scorer.print_scores(dec_places=3)


reporter.report('Plotting calibration')
plot_calibration(p=scorer.p,
                 calib_curves=scorer.calib_curves,
                 curve_transparency=0.1,
                 output_dir=FIGURES_OUTPUT_DIR,
                 output_filename='current_model_calibration')


reporter.last('Done.')
