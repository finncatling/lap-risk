import os
from typing import Dict, Tuple, List

import pandas as pd

from utils.constants import (
    DATA_DIR,
    NOVEL_MODEL_OUTPUT_DIR,
    FIGURES_OUTPUT_DIR,
    RANDOM_SEED,
)
from utils.indications import INDICATION_VAR_NAME, IndicationNameProcessor
from utils.io import save_object, load_object
from utils.model.albumin import albumin_model_factory
from utils.model.lactate import lactate_model_factory
from utils.model.novel import (
    ALBUMIN_VAR_NAME,
    LACTATE_VAR_NAME,
    NOVEL_MODEL_VARS,
    WINSOR_QUANTILES,
    CategoricalImputer,
    LactateAlbuminImputer,
    NovelModel,
    novel_model_factory
)
from utils.model.shared import LogOddsTransformer
from utils.impute import ImputationInfo
from utils.plot.helpers import plot_saver
from utils.plot.pdp import (
    PDPTerm,
    PDPFigure,
    compare_pdps_from_different_gams_plot
)
from utils.evaluate import Scorer, score_linear_predictions
from utils.report import Reporter


reporter = Reporter()
reporter.title("Check sensitivity of albumin and lactate features in novel "
               "model to use of mortality labels as a feature in the albumin "
               "and lactate imputation models. We refit the novel model on the "
               "zeroth train-test split only, as we are just conducting a "
               "visual comparison of the novel model partial dependence plots.")


reporter.report("Loading previous outputs needed for analysis")
df = pd.read_pickle(
    os.path.join(DATA_DIR, "05_preprocessed_df.pkl"))
cat_imputer: CategoricalImputer = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "06_categorical_imputer.pkl"))
multi_category_levels: Dict[str, Tuple] = load_object(
    os.path.join(
        NOVEL_MODEL_OUTPUT_DIR,
        "05_multi_category_levels_with_indications.pkl"))
lacalb_pdp_terms: List[PDPTerm] = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "07_pd_plot_specification.pkl"))
imputation_stages: ImputationInfo = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "05_imputation_stages.pkl"))
novel_pdp_terms: List[PDPTerm] = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "08_pd_plot_specification.pkl"))


reporter.report("Adding mortality to albumin and lactate partial dependence "
                "plots")
indication_names = IndicationNameProcessor(
    multi_category_levels=multi_category_levels,
    remove_missing_category=True
)
lacalb_pdp_terms[-1] = PDPTerm(
    INDICATION_VAR_NAME,
    "Indication",
    (slice(4, 6), slice(0, 2)),
    indication_names.sanitized,
    ["No CT", "CT"],
    "best")
lacalb_pdp_terms.append(
    PDPTerm(
        NOVEL_MODEL_VARS["target"],
        "Mortality",
        (4, 2),
        ["Lived", "Died"]))


imputers = {}
for pretty_name, variable_name, model_factory in (
    ('albumin', ALBUMIN_VAR_NAME, albumin_model_factory),
    ('lactate', LACTATE_VAR_NAME, lactate_model_factory)
):
    reporter.report(f"Fitting imputers for {pretty_name}")
    imputers[pretty_name] = LactateAlbuminImputer(
        df=df.loc[:, [variable_name, NOVEL_MODEL_VARS["target"]]],
        categorical_imputer=cat_imputer,
        lacalb_variable_name=variable_name,
        imputation_model_factory=model_factory,
        winsor_quantiles=WINSOR_QUANTILES,
        multi_cat_vars=multi_category_levels,
        indication_var_name=INDICATION_VAR_NAME,
        mortality_as_feature=True,
        random_seed=RANDOM_SEED)
    imputers[pretty_name].fit()


    reporter.report(f"Saving {pretty_name} imputer for later use")
    save_object(
        imputers[pretty_name],
        os.path.join(NOVEL_MODEL_OUTPUT_DIR,
                     f"11_{pretty_name}_imputer_with_mortality_feature.pkl"))


    reporter.report(f"Scoring {pretty_name} imputation model performance.")
    y_obs, y_preds = imputers[pretty_name].get_all_observed_and_predicted(
        fold_name='test',
        probabilistic=False,
        lac_alb_imp_i=None)
    scorer = Scorer(
        y_true=y_obs,
        y_pred=y_preds,
        scorer_function=score_linear_predictions,
        n_splits=imputers[pretty_name].tts.n_splits)
    scorer.calculate_scores()
    reporter.first("Scores with median as point estimate:")
    scorer.print_scores(dec_places=3, point_estimate='median')
    reporter.first("Scores with split 0 as point estimate:")
    scorer.print_scores(dec_places=3, point_estimate='split0')


    reporter.first("Saving model scorer for later use")
    save_object(
        scorer,
        os.path.join(
            NOVEL_MODEL_OUTPUT_DIR,
            f"11_{pretty_name}_imputer_with_mortality_feature_scorer.pkl"))


    reporter.report('Preparing data for PDP histograms')
    pdp_hist_data = pd.concat(
        objs=(
            cat_imputer.get_imputed_df('train', 0, 0),
            cat_imputer.get_imputed_df('test', 0, 0),
        ),
        axis=0,
        ignore_index=True)


    reporter.report(f"Plotting {pretty_name} imputer partial dependence plots")
    for hist_switch, hist_text in ((False, ''), (True, '_with_histograms')):
        for space, ylabel, kwargs in (
            ('gaussian', None, {}),
            ('inv_trans',
             pretty_name,
             {
                'transformer': imputers[pretty_name].transformers[0],
                'plot_just_outer_ci': True
             })
        ):
            pdp_generator = PDPFigure(
                gam=imputers[pretty_name].imputers[0],
                pdp_terms=lacalb_pdp_terms,
                ylabel=ylabel,
                plot_hists=hist_switch,
                hist_data=pdp_hist_data, **kwargs)
            plot_saver(
                pdp_generator.plot,
                output_dir=FIGURES_OUTPUT_DIR,
                output_filename=(
                    f"11_{pretty_name}_imputer_with_mortality_feature_"
                    f"{space}_pd_plot{hist_text}"))


# # TODO: Remove this development code
# reporter.report(f"Loading pretrained imputation models")
# imputers = {}
# for imputer_name in ('albumin', 'lactate'):
#     imputers[imputer_name] = load_object(os.path.join(
#         NOVEL_MODEL_OUTPUT_DIR,
#         f"11_{imputer_name}_imputer_with_mortality_feature.pkl"))


reporter.report("Restricting novel model refitting to zeroth train-test split")
cat_imputer.tts.n_splits = 1


reporter.report("Refitting novel model")
refit_novel_model = NovelModel(
    categorical_imputer=cat_imputer,
    albumin_imputer=imputers['albumin'],
    lactate_imputer=imputers['lactate'],
    model_factory=novel_model_factory,
    n_lacalb_imputations_per_mice_imp=(
        imputation_stages.multiple_of_previous_n_imputations[1]),
    random_seed=RANDOM_SEED)
refit_novel_model.fit()


reporter.report("Saving refitted novel model")
save_object(
    refit_novel_model,
    os.path.join(
        NOVEL_MODEL_OUTPUT_DIR,
        "11_novel_model_lacalb_sensitivity.pkl"))


reporter.report('Preparing data for PDP histograms')
pdp_hist_data = pd.concat(
    objs=(
        refit_novel_model.get_features_and_labels('train', 0, 0, 0)[0],
        refit_novel_model.get_features_and_labels('test', 0, 0, 0)[0]
    ),
    axis=0,
    ignore_index=True)


reporter.report("Plotting novel model partial dependence plots")
for hist_switch, hist_text in ((False, ''), (True, '_with_histograms')):
    for space, kwargs in (
        ('log_odds', {}),
        ('relative_risk', {'transformer': LogOddsTransformer()})
    ):
        pdp_generator = PDPFigure(gam=refit_novel_model.models[0],
                                  pdp_terms=novel_pdp_terms, ylabel='',
                                  plot_hists=hist_switch,
                                  hist_data=pdp_hist_data, **kwargs)
        plot_saver(
            pdp_generator.plot,
            output_dir=FIGURES_OUTPUT_DIR,
            output_filename=(
                f"11_novel_model_lacalb_sensitivity_{space}_pd_plot"
                f"{hist_text}"))


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
        gam_names=('Original', 'Re-fit'),
        term_indices=(6, 7),
        term_names=("Lactate (mmol/L)", "Albumin (g/L)"),
        column_indices=(19, 17),
        transformer=transformer,
        output_dir=FIGURES_OUTPUT_DIR,
        output_filename=f"11_novel_model_vs_refit_lacalb_{space}_pd_plot"
    )


reporter.last("Done.")
