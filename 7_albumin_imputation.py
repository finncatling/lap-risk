from typing import Dict, Tuple
import os
import pandas as pd

from utils.report import Reporter
from utils.constants import DATA_DIR, NOVEL_MODEL_OUTPUT_DIR, FIGURES_OUTPUT_DIR
from utils.io import make_directory, save_object, load_object
from utils.impute import CategoricalImputer, LactateAlbuminImputer
from utils.model.novel import (
    ALBUMIN_VAR_NAME,
    NOVEL_MODEL_VARS,
    WINSOR_QUANTILES,
    INDICATION_VAR_NAME,
)
from utils.model.albumin import albumin_model_factory, GammaTransformer
from utils.plot.pdp import PDPTerm, plot_partial_dependence
from utils.plot.helpers import sanitize_indication, plot_saver


reporter = Reporter()
reporter.title("Fit albumin imputation models")


reporter.report("Creating output dirs (if they don't already exist)")
make_directory(NOVEL_MODEL_OUTPUT_DIR)
make_directory(FIGURES_OUTPUT_DIR)


reporter.report("Loading previous analysis outputs needed for imputation")
df = pd.read_pickle(os.path.join(DATA_DIR, "df_preprocessed_for_novel_pre_split.pkl"))
cat_imputer: CategoricalImputer = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "categorical_imputer.pkl")
)
multi_category_levels: Dict[str, Tuple] = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "multi_category_levels_with_indications.pkl")
)


reporter.report("Fitting imputers for albumin")
alb_imputer = LactateAlbuminImputer(
    df=df.loc[:, [ALBUMIN_VAR_NAME, NOVEL_MODEL_VARS["target"]]],
    categorical_imputer=cat_imputer,
    imputation_target=ALBUMIN_VAR_NAME,
    imputation_model_factory=albumin_model_factory,
    winsor_quantiles=WINSOR_QUANTILES,
    transformer=GammaTransformer,
    transformer_args={},
    multi_cat_vars=multi_category_levels,
    indication_var_name=INDICATION_VAR_NAME,
)
alb_imputer.tts.n_splits = 1  # TODO: Remove this testing line
alb_imputer.fit()


reporter.report("Saving draft albumin imputer for later use")
save_object(
    alb_imputer, os.path.join(NOVEL_MODEL_OUTPUT_DIR, "draft_albumin_imputer.pkl")
)


reporter.report("Specifying properties of albumin GAM partial dependence plot")
alb_pdp_terms = [
    PDPTerm("S01AgeOnArrival", "Age (years)", (0, 0)),
    PDPTerm("S03Sodium", "Sodium (mmol/L)", (1, 1)),
    PDPTerm("S03Potassium", "Potassium (mmol/L)", (1, 2)),
    PDPTerm("S03Urea", "Urea (mmol/L)", (2, 0)),
    PDPTerm("S03WhiteCellCount", r"White cell count ($\times$10${^9}$/L)", (1, 0)),
    PDPTerm("S03SystolicBloodPressure", "Systolic pressure (mmHg)", (0, 1)),
    PDPTerm("S03ASAScore", "ASA physical status", (2, 1), list(range(1, 6))),
    PDPTerm(
        "S03DiagnosedMalignancy",
        "Malignancy",
        (2, 2),
        ["None", "Primary\nonly", "Nodal\nmets.", "Distant\nmets."],
        ["No CT", "CT"],
        "upper left",
    ),
    PDPTerm(
        "S03Pred_Peritsoil",
        "Peritoneal soiling",
        (3, 0),
        ["None", "Serous", "Local\npus", "Free pus /\nblood / faeces"],
        ["No CT", "CT"],
        "upper left",
    ),
    PDPTerm(
        INDICATION_VAR_NAME,
        "Indication",
        (slice(3, 5), slice(1, 3)),
        [sanitize_indication(s) for s in multi_category_levels[INDICATION_VAR_NAME]],
        ["No CT", "CT"],
        "upper left",
    ),
    PDPTerm(
        "S03Pulse",
        "Heart rate (BPM)",
        (0, 2),
        None,
        ["Sinus", "Arrhythmia"],
        "lower right",
    ),
]


reporter.report("Saving albumin PDP specification")
save_object(
    alb_pdp_terms, os.path.join(NOVEL_MODEL_OUTPUT_DIR, "alb_pdp_specification.pkl")
)


# TODO: Variables don't look winsorized on the plots - explore why
# TODO: Indications should be more regularised
# TODO: Flip axes given that albumin is transformed?


reporter.report("Plotting albumin imputer partial dependence plot")
plot_saver(
    plot_partial_dependence,
    gam=alb_imputer._imputers[0],
    pdp_terms=alb_pdp_terms,
    output_dir=FIGURES_OUTPUT_DIR,
    output_filename="alb_imputer_pdp",
)


reporter.last("Done.")

# TODO: Save summary stats (including those from MICE) for external use
