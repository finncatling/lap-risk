import os

import pandas as pd
from progressbar import progressbar

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
    LactateAlbuminImputer,
    NovelModel,
    novel_model_factory
)
from utils.model.shared import LogOddsTransformer
from utils.plot.helpers import plot_saver
from utils.plot.pdp import PDPTerm, PDPFigure
from utils.evaluate import LogisticScorer, score_logistic_predictions
from utils.report import Reporter


reporter = Reporter()
reporter.title(
    "Fit novel emergency laparotomy mortality risk model "
    "on the different train folds, and evaluate the models "
    "obtained on the corresponding test folds."
)
reporter.title(
    "NB. If all model training is attempted in a single run of this script, "
    "it may crash in the later stages of model training. If this "
    "occurs, please see commented code in the script for loading your "
    "partially-trained model and resuming training."
)


reporter.report("Loading previous analysis outputs needed for novel model")
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
    multi_category_levels=albumin_imputer.multi_cat_vars,
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
        r"White cell count ($\times$10$^9$ L$^{-1}$)",
        (1, 0)
    ),
    PDPTerm("S03Sodium", r"Sodium (mmol L$^{-1}$)", (1, 1)),
    PDPTerm("S03Potassium", r"Potassium (mmol L$^{-1}$)", (1, 2)),
    PDPTerm("S03PreOpArterialBloodLactate", r"Lactate (mmol L$^{-1}$)", (2, 0)),
    PDPTerm("S03PreOpLowestAlbumin", r"Albumin (g L$^{-1}$)", (2, 1)),
    PDPTerm("S03GlasgowComaScore", "Glasgow Coma Score", (2, 2)),
    PDPTerm("S03ASAScore", "ASA physical status", (3, 0), list(range(1, 6))),
    PDPTerm(
        "S03DiagnosedMalignancy",
        "Malignancy",
        (3, 1),
        ["None", "Primary\nonly", "Nodal\nmets.", "Distant\nmets."],
        ["No CT", "CT"],
        "best",
    ),
    PDPTerm(
        "S03Pred_Peritsoil",
        "Peritoneal soiling",
        (3, 2),
        ["None", "Serous", "Local\npus", "Free pus /\nblood / faeces"],
        ["No CT", "CT"],
        "best",
    ),
    PDPTerm(
        ("S03CardiacSigns", "S03RespiratorySigns"),
        ("Cardiovascular", "Respiratory"),
        (4, 0),
        (None, None),
        None,
        None,
        (40, 205)
    ),
    PDPTerm(
        ("S03SerumCreatinine", "S03Urea"),
        (r"Creatinine (mmol L$^{-1}$)", r"Urea (mmol L$^{-1}$)"),
        (4, 1),
        (None, None),
        None,
        None,
        (40, 205)
    ),
    PDPTerm(
        INDICATION_VAR_NAME,
        "Indication",
        (slice(5, 7), slice(0, 3)),
        indication_names.sanitized,
        ["No CT", "CT"],
        "best",
    )
]


reporter.report("Saving partial dependence plot specification")
save_object(
    pdp_terms,
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "08_pd_plot_specification.pkl")
)


"""
BEGIN code for either instantiating novel model (if no previous training) or 
loading a partially-trained instance of novel model. Comment as appropriate.
"""
reporter.report("Making new instance of novel model")
novel_model = NovelModel(
    categorical_imputer=albumin_imputer.cat_imputer,
    albumin_imputer=albumin_imputer,
    lactate_imputer=lactate_imputer,
    model_factory=novel_model_factory,
    n_lacalb_imputations_per_mice_imp=(
        imputation_stages.multiple_of_previous_n_imputations[1]),
    random_seed=RANDOM_SEED
)

# reporter.report(f"Loading pretrained novel model")
# novel_model: NovelModel = load_object(
#     os.path.join(NOVEL_MODEL_OUTPUT_DIR, "08_novel_model.pkl"))
"""END code for instantiating / loading novel model."""


reporter.report("Beginning train-test splitting and model fitting. Training "
                "resumes where it left off if the model is partially trained, "
                "and is skipped entirely if the model is fully trained.")
for split_i in progressbar(
    range(len(novel_model.models), novel_model.cat_imputer.tts.n_splits),
    prefix="Split iteration"
):
    novel_model._single_train_test_split(split_i)
    save_object(
        novel_model, os.path.join(NOVEL_MODEL_OUTPUT_DIR, "08_novel_model.pkl"))


reporter.report(f"Scoring novel model performance.")
y_obs, y_preds = novel_model.get_all_observed_and_median_predicted(
    fold_name='test',
    n_samples_per_imp_i=5
)
scorer = LogisticScorer(
    y_true=y_obs,
    y_pred=y_preds,
    scorer_function=score_logistic_predictions,
    n_splits=novel_model.cat_imputer.tts.n_splits,
    calibration_n_splines=CALIB_GAM_N_SPLINES,
    calibration_lam_candidates=CALIB_GAM_LAM_CANDIDATES)
scorer.calculate_scores()
reporter.first("Scores with median as point estimate:")
scorer.print_scores(dec_places=3, point_estimate='median')
reporter.first("Scores with split 0 as point estimate:")
scorer.print_scores(dec_places=3, point_estimate='split0')


reporter.first("Saving model scorer for later use")
save_object(
    scorer,
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "08_novel_model_scorer.pkl"))


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
    for space_name, pretty_space_name, kwargs in (
        ('log_odds', 'Log-odds of mortality', {}),
        ('relative_risk',
         'Relative mortality risk',
         {'transformer': LogOddsTransformer()})
    ):
        pdp_generator = PDPFigure(
            gam=novel_model.models[0],
            pdp_terms=pdp_terms,
            ylabel=pretty_space_name,
            plot_hists=hist_switch,
            hist_data=pdp_hist_data, **kwargs)
        plot_saver(
            pdp_generator.plot,
            output_dir=FIGURES_OUTPUT_DIR,
            output_filename=(
                f"08_novel_model_{space_name}_pd_plot{hist_text}_nomissinds"))


reporter.last("Done.")
