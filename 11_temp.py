import os

from utils.constants import (FIGURES_OUTPUT_DIR, NOVEL_MODEL_OUTPUT_DIR)
from utils.io import load_object
from utils.model.novel import (
    NovelModel
)
from utils.model.shared import LogOddsTransformer
from utils.plot.helpers import plot_saver
from utils.plot.pdp import (
    compare_pdps_from_different_gams_plot
)
from utils.report import Reporter

reporter = Reporter()
reporter.title("Check sensitivity of albumin and lactate features in novel "
               "model to use of mortality labels as a feature in the albumin "
               "and lactate imputation models. We refit the novel model on the "
               "zeroth train-test split only, as we are just conducting a "
               "visual comparison of the novel model partial dependence plots.")


reporter.report("Loading pre-refitted novel model")
refit_novel_model: NovelModel = load_object(os.path.join(
    NOVEL_MODEL_OUTPUT_DIR,
    "11_novel_model_lacalb_sensitivity.pkl"))


reporter.report("Loading 'original' novel model trained in script 08")
original_novel_model: NovelModel = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "08_novel_model.pkl"))


reporter.report("Plotting comparison of the PDPs for albumin and lactate in "
                "the original and refit novel models")
for space, transformer in (
    ('log_odds', None),
    ('relative_risk', LogOddsTransformer)
):
    plot_saver(
        compare_pdps_from_different_gams_plot,
        gams=(original_novel_model.models[0], refit_novel_model.models[0]),
        gam_names=('Original', 'Re-fitted'),
        term_indices=(6, 7),
        term_names=("Lactate (mmol/L)", "Albumin (g/L)"),
        column_indices=(19, 17),
        transformer=transformer,
        output_dir=FIGURES_OUTPUT_DIR,
        output_filename=f"11_novel_model_vs_refit_lacalb_{space}_pd_plot"
    )


reporter.last("Done.")
