import os

import pandas as pd

from utils.constants import (DATA_DIR, RANDOM_SEED, FIGURES_OUTPUT_DIR,
                             STATS_OUTPUT_DIR, CALIB_GAM_N_SPLINES,
                             CALIB_GAM_LAM_CANDIDATES)
from utils.current_nela_model import (preprocess_df, SplitterTrainerPredictor,
                                      WINSOR_THRESHOLDS,
                                      CURRENT_NELA_MODEL_VARS, CENTRES)
from utils.io import make_directory
from utils.helpers import flatten_nela_var_dict
from utils.split_data import drop_incomplete_cases
from utils.evaluate import ModelScorer, plot_calibration
from utils.io import load_object, save_object
from utils.report import Reporter


reporter = Reporter()
reporter.first('***** Re-fit current NELA emergency laparotomy mortality risk'
               "model on the different train folds, and evaluate the models'"
               'performance *****')


reporter.report("Creating external output dirs (if it doesn't already exist)")
make_directory(os.path.join(FIGURES_OUTPUT_DIR))
make_directory(os.path.join(STATS_OUTPUT_DIR))


reporter.report('Loading manually-wrangled NELA data')
df = pd.read_pickle(
    os.path.join(DATA_DIR, 'df_after_univariate_wrangling.pkl'))


reporter.report('Removing unused variables')
df = df[flatten_nela_var_dict(CURRENT_NELA_MODEL_VARS)]


reporter.report("Dropping cases which are incomplete for the models' variables")
df, _ = drop_incomplete_cases(df)


reporter.report('Preparing list of variables for binarization')
"""
GCS is exempt from binarisation as it is binned in a separate function. ASA is
exempt as it is only used in a later interaction term.
"""
binarize_vars = list(CURRENT_NELA_MODEL_VARS['cat'])
binarize_vars.remove('S03GlasgowComaScore')
binarize_vars.remove('S03ASAScore')


reporter.report('Preparing list of variables for quadratic transformation')
"""
Sodium is exempt as it undergoes a customised transformation. We apply the
quadratic transformation to creatinine and urea after they are log-transformed.
"""
quadratic_vars = list(CURRENT_NELA_MODEL_VARS['cont'])
quadratic_vars.remove('S03Sodium')
for original, logged in (('S03SerumCreatinine', 'logcreat'),
                         ('S03Urea', 'logurea')):
    quadratic_vars.remove(original)
    quadratic_vars.append(logged)


reporter.report('Preprocessing data')
"""
Our preprocessing code was originally designed to allow preprocessing after
splitting the data. However, the preprocessing steps do not 'leak' any
information from the test-fold cases, e.g. they don't use any summary statistics
derived from the data for use in transforming the variables. This means that
we can speed up our code by running the preprocessing prior to the loop wherein
the data are repeatedly split and the model retrained.  
"""
preprocessed_df, _ = preprocess_df(df,
                                   quadratic_vars=quadratic_vars,
                                   winsor_threholds=WINSOR_THRESHOLDS,
                                   centres=CENTRES,
                                   binarize_vars=binarize_vars,
                                   label_binarizers=None)


reporter.report('Loading data needed for train-test splitting')
tt_splitter = load_object(os.path.join('outputs', 'train_test_splitter.pkl'))


reporter.report('Beginning train-test splitting and model fitting')
stp = SplitterTrainerPredictor(
    df,
    test_train_splitter=tt_splitter,
    target_variable_name=CURRENT_NELA_MODEL_VARS['target'],
    random_seed=RANDOM_SEED)
stp.split_train_predict()


reporter.report('Saving SplitterTrainerPredictor for later use')
save_object(stp, os.path.join('outputs', 'splitter_trainer_predictor.pkl'))


reporter.report('Scoring model performance')
scorer = ModelScorer(y_true=stp.y_test,
                     y_pred=stp.y_pred,
                     calibration_n_splines=CALIB_GAM_N_SPLINES,
                     calibration_lam_candidates=CALIB_GAM_LAM_CANDIDATES)
scorer.calculate_scores()
print('')
scorer.print_scores(dec_places=3)


reporter.first('Saving ModelScorer for later use')
save_object(stp, os.path.join('outputs', 'splitter_trainer_predictor.pkl'))


reporter.report('Saving plot of calibration curves')
plot_calibration(p=scorer.p,
                 calib_curves=scorer.calib_curves,
                 curve_transparency=0.15,
                 output_dir=FIGURES_OUTPUT_DIR,
                 output_filename='current_model_calibration')


reporter.report('Saving summary statistics for external use')
del scorer.scores['per_iter']
current_model_stats = {'train_fold_stats': stp.split_stats,
                       'model_features': stp.features,
                       'model_coefficients': stp.coefficients,
                       'scores': scorer.scores,
                       'calib_p': scorer.p,
                       'calib_curves': scorer.calib_curves,
                       'calib_n_splines': scorer.calib_n_splines,
                       'calib_lam_candidates': scorer.calib_lam_candidates,
                       'calib_best_lams': scorer.calib_lams}
save_object(current_model_stats,
            os.path.join(STATS_OUTPUT_DIR,
                         '2_train_eval_current_model_stats.pkl'))


reporter.last('Done.')
