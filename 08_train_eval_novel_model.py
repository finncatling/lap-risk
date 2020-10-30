import os
from typing import Tuple, Dict

import pandas as pd

from utils.constants import (
    NOVEL_MODEL_OUTPUT_DIR,
    RANDOM_SEED,
    FIGURES_OUTPUT_DIR,
    CALIB_GAM_N_SPLINES,
    CALIB_GAM_LAM_CANDIDATES
)
from utils.impute import ImputationInfo
from utils.indications import INDICATION_VAR_NAME, IndicationNameProcessor
from utils.io import load_object, save_object
from utils.model.novel import (
    CategoricalImputer,
    LactateAlbuminImputer,
    NovelModel,
    novel_model_factory,
    LogOddsTransformer
)
from utils.plot.helpers import plot_saver
from utils.plot.pdp import PDPTerm, PDPFigure
from utils.evaluate import LogisticScorer, score_logistic_predictions
from utils.report import Reporter


reporter = Reporter()
reporter.title(
    "Fit novel emergency laparotomy mortality risk model "
    "on the different train folds, and evaluate the models "
    "obtained on the corresponding test folds"
)


reporter.report("Loading previous analysis outputs needed for novel model")
multi_category_levels: Dict[str, Tuple] = load_object(
    os.path.join(
        NOVEL_MODEL_OUTPUT_DIR,
        "05_multi_category_levels_with_indications.pkl"
    )
)
cat_imputer: CategoricalImputer = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "06_categorical_imputer.pkl")
)
albumin_imputer: LactateAlbuminImputer = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "07_albumin_imputer.pkl")
)
lactate_imputer: LactateAlbuminImputer = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "07_lactate_imputer.pkl")
)
imputation_stages: ImputationInfo = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "05_imputation_stages.pkl")
)


reporter.report("Specifying properties of GAM partial dependence plot")
indication_names = IndicationNameProcessor(
    multi_category_levels=multi_category_levels,
    remove_missing_category=True
)
pdp_terms = [
    PDPTerm("S01AgeOnArrival", "Age (years)", (0, 0)),
    PDPTerm("S03SystolicBloodPressure", "Systolic pressure (mmHg)", (0, 1)),
    PDPTerm(
        "S03Pulse",
        "Heart rate (BPM)",
        (0, 2),
        None,
        ["Sinus", "Arrhythmia"],
        "best",
    ),
    PDPTerm(
        "S03WhiteCellCount",
        r"White cell count ($\times$10${^9}$/L)",
        (1, 0)
    ),
    PDPTerm("S03Sodium", "Sodium (mmol/L)", (1, 1)),
    PDPTerm("S03Potassium", "Potassium (mmol/L)", (1, 2)),
    PDPTerm("S03PreOpArterialBloodLactate", "Lactate (mmol/L)", (2, 0)),
    PDPTerm("S03PreOpLowestAlbumin", "Albumin (g/L)", (2, 1)),
    PDPTerm("S03GlasgowComaScore", "Glasgow Coma Score", (2, 2)),
    PDPTerm("S03ASAScore", "ASA physical status", (3, 0), list(range(1, 6))),
    PDPTerm(
        "S03PreOpArterialBloodLactate_missing",
        "Lactate missing",
        (3, 1),
        ["No", "Yes"]
    ),
    PDPTerm(
        "S03PreOpLowestAlbumin_missing",
        "Albumin missing",
        (3, 2),
        ["No", "Yes"]
    ),
    PDPTerm(
        "S03DiagnosedMalignancy",
        "Malignancy",
        (4, 0),
        ["None", "Primary\nonly", "Nodal\nmets.", "Distant\nmets."],
        ["No CT", "CT"],
        "best",
    ),
    PDPTerm(
        "S03Pred_Peritsoil",
        "Peritoneal soiling",
        (4, 1),
        ["None", "Serous", "Local\npus", "Free pus /\nblood / faeces"],
        ["No CT", "CT"],
        "best",
    ),
    PDPTerm(
        ("S03CardiacSigns", "S03RespiratorySigns"),
        ("Cardiovascular", "Respiratory"),
        (4, 2),
        (None, None),
        None,
        None,
        (40, 205)
    ),
    PDPTerm(
        ("S03SerumCreatinine", "S03Urea"),
        ("Creatinine (mmol/L)", "Urea (mmol/L)"),
        (5, 2),
        (None, None),
        None,
        None,
        (40, 205)
    ),
    PDPTerm(
        INDICATION_VAR_NAME,
        "Indication",
        (slice(5, 7), slice(0, 2)),
        indication_names.sanitized,
        ["No CT", "CT"],
        "best",
    ),
]


reporter.report("Saving partial dependence plot specification")
save_object(
    pdp_terms,
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "08_pd_plot_specification.pkl")
)


reporter.report("Beginning train-test splitting and model fitting")
novel_model = NovelModel(
    categorical_imputer=cat_imputer,
    albumin_imputer=albumin_imputer,
    lactate_imputer=lactate_imputer,
    model_factory=novel_model_factory,
    n_lacalb_imputations_per_mice_imp=(
        imputation_stages.multiple_of_previous_n_imputations[1]),
    random_seed=RANDOM_SEED
)
novel_model.fit()


reporter.report(f"Saving novel model for later use")
save_object(
    novel_model,
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "08_novel_model.pkl"))


# # TODO: Remove this development code
# reporter.report(f"Loading pretrained novel model")
# novel_model: NovelModel = load_object(
#     os.path.join(NOVEL_MODEL_OUTPUT_DIR, "08_novel_model.pkl"))


reporter.report(f"Scoring novel model performance.")
# TODO: Method to get observed and predicted values
# y_obs, y_preds = novel_model.get_all_observed_and_predicted()
# scorer = LogisticScorer(
#     y_true=y_obs,
#     y_pred=y_preds,
#     scorer_function=score_logistic_predictions,
#     calibration_n_splines=CALIB_GAM_N_SPLINES,
#     calibration_lam_candidates=CALIB_GAM_LAM_CANDIDATES)
# scorer.calculate_scores()
# reporter.first("Scores with median as point estimate:")
# scorer.print_scores(dec_places=3, point_estimate='median')
# reporter.first("Scores with fold 0 as point estimate:")
# scorer.print_scores(dec_places=3, point_estimate='fold0')
#
#
# reporter.first("Saving model scorer for later use")
# save_object(
#     scorer,
#     os.path.join(NOVEL_MODEL_OUTPUT_DIR, f"08_novel_model_scorer.pkl"))


reporter.report('Preparing data for PDP histograms')
pdp_hist_data = pd.concat(
    objs=(
        novel_model.get_features_and_labels('train', 0, 0, 0)[0],
        novel_model.get_features_and_labels('test', 0, 0, 0)[0]
    ),
    axis=0,
    ignore_index=True)

reporter.first("Plotting novel model partial dependence plots")
for hist_switch, hist_text in ((False, ''), (True, '_with_histograms')):
    for space, kwargs in (
        ('log_odds', {}),
        ('probability', {'transformer': LogOddsTransformer()})
    ):
        pdp_generator = PDPFigure(
            gam=novel_model.models[0],
            pdp_terms=pdp_terms,
            plot_hists=hist_switch,
            hist_data=pdp_hist_data,
            **kwargs)
        plot_saver(
            pdp_generator.plot,
            output_dir=FIGURES_OUTPUT_DIR,
            output_filename=f"08_novel_model_{space}_pd_plot{hist_text}")


reporter.last("Done.")
