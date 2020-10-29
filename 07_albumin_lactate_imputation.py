import os
from typing import Dict, Tuple

import pandas as pd

from utils.constants import (
    DATA_DIR,
    NOVEL_MODEL_OUTPUT_DIR,
    FIGURES_OUTPUT_DIR,
    RANDOM_SEED
)
from utils.indications import INDICATION_VAR_NAME
from utils.io import save_object, load_object
from utils.model.albumin import albumin_model_factory
from utils.model.lactate import lactate_model_factory
from utils.model.novel import (
    ALBUMIN_VAR_NAME,
    LACTATE_VAR_NAME,
    NOVEL_MODEL_VARS,
    WINSOR_QUANTILES,
    CategoricalImputer,
    LactateAlbuminImputer
)
from utils.plot.helpers import sanitize_indication, plot_saver
from utils.plot.pdp import PDPTerm, PDPFigure
from utils.evaluate import Scorer, score_linear_predictions
from utils.report import Reporter


reporter = Reporter()
reporter.title("Fit albumin and lactate imputation models")


reporter.report("Loading previous analysis outputs needed for imputation")
df = pd.read_pickle(
    os.path.join(DATA_DIR, "05_preprocessed_df.pkl")
)
cat_imputer: CategoricalImputer = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "06_categorical_imputer.pkl")
)
multi_category_levels: Dict[str, Tuple] = load_object(
    os.path.join(
        NOVEL_MODEL_OUTPUT_DIR,
        "05_multi_category_levels_with_indications.pkl"
    )
)


reporter.report("Specifying properties of GAM partial dependence plot")
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
    PDPTerm("S03GlasgowComaScore", "Glasgow Coma Score", (2, 0)),
    PDPTerm("S03ASAScore", "ASA physical status", (2, 1), list(range(1, 6))),
    PDPTerm(
        "S03DiagnosedMalignancy",
        "Malignancy",
        (2, 2),
        ["None", "Primary\nonly", "Nodal\nmets.", "Distant\nmets."],
        ["No CT", "CT"],
        "best",
    ),
    PDPTerm(
        "S03Pred_Peritsoil",
        "Peritoneal soiling",
        (3, 0),
        ["None", "Serous", "Local\npus", "Free pus /\nblood / faeces"],
        ["No CT", "CT"],
        "best",
    ),
    PDPTerm(
        ("S03CardiacSigns", "S03RespiratorySigns"),
        ("Cardiovascular", "Respiratory"),
        (3, 1),
        (None, None),
        None,
        None,
        (30, 25)
    ),
    PDPTerm(
        ("S03SerumCreatinine", "S03Urea"),
        ("Creatinine (mmol/L)", "Urea (mmol/L)"),
        (3, 2),
        (None, None),
        None,
        None,
        (30, 115)
    ),
    PDPTerm(
        INDICATION_VAR_NAME,
        "Indication",
        (slice(4, 6), slice(0, 3)),
        [sanitize_indication(s) for s in
         multi_category_levels[INDICATION_VAR_NAME]],
        ["No CT", "CT"],
        "best",
    ),
]


reporter.report("Saving partial dependence plot specification")
save_object(
    pdp_terms,
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "07_pd_plot_specification.pkl")
)


for pretty_name, variable_name, model_factory in (
    ('albumin', ALBUMIN_VAR_NAME, albumin_model_factory),
    ('lactate', LACTATE_VAR_NAME, lactate_model_factory),
):
    reporter.report(f"Fitting imputers for {pretty_name}")
    imputer = LactateAlbuminImputer(
        df=df.loc[:, [variable_name, NOVEL_MODEL_VARS["target"]]],
        categorical_imputer=cat_imputer,
        lacalb_variable_name=variable_name,
        imputation_model_factory=model_factory,
        winsor_quantiles=WINSOR_QUANTILES,
        multi_cat_vars=multi_category_levels,
        indication_var_name=INDICATION_VAR_NAME,
        random_seed=RANDOM_SEED)
    imputer.fit()


    reporter.report(f"Saving {pretty_name} imputer for later use")
    save_object(
        imputer,
        os.path.join(NOVEL_MODEL_OUTPUT_DIR, f"07_{pretty_name}_imputer.pkl"))


    reporter.report(f"Scoring {pretty_name} imputation model performance.")
    y_obs, y_preds = imputer.get_all_observed_and_predicted(
        fold_name='test',
        probabilistic=False,
        lac_alb_imp_i=None)
    scorer = Scorer(
        y_true=y_obs,
        y_pred=y_preds,
        scorer_function=score_linear_predictions)
    scorer.calculate_scores()
    reporter.first("Scores with median as point estimate:")
    scorer.print_scores(dec_places=3, point_estimate='median')
    reporter.first("Scores with fold 0 as point estimate:")
    scorer.print_scores(dec_places=3, point_estimate='fold0')


    reporter.first("Saving model scorer for later use")
    save_object(
        scorer,
        os.path.join(
            NOVEL_MODEL_OUTPUT_DIR, f"07_{pretty_name}_imputer_scorer.pkl"))


    reporter.first(f"Plotting {pretty_name} imputer partial dependence plots")
    for space, kwargs in (
        ('gaussian', {}),
        ('inv_trans', {'transformer': imputer.transformers[0]})
    ):
        pdp_generator = PDPFigure(
            gam=imputer.imputers[0],
            pdp_terms=pdp_terms,
            **kwargs)
        plot_saver(
            pdp_generator.plot,
            output_dir=FIGURES_OUTPUT_DIR,
            output_filename=f"07_{pretty_name}_imputer_{space}_pd_plot")


reporter.last("Done.")
