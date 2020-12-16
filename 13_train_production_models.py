import copy
import os
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
from progressbar import progressbar as pb
from pprint import PrettyPrinter

from utils.constants import (
    DATA_DIR,
    INTERNAL_OUTPUT_DIR,
    NOVEL_MODEL_OUTPUT_DIR,
    FIGURES_OUTPUT_DIR,
    CALIB_GAM_N_SPLINES,
    RANDOM_SEED
)
from utils.data_check import load_nela_data_and_sanity_check
from utils.model.shared import flatten_model_var_dict
from utils.io import load_object, save_object
from utils.model.current import CURRENT_MODEL_VARS
from utils.model.novel import (
    NOVEL_MODEL_VARS,
    LACTATE_VAR_NAME,
    ALBUMIN_VAR_NAME,
    WINSOR_QUANTILES,
    SplitterWinsorMICE,
    CategoricalImputer,
    LactateAlbuminImputer,
    NovelModel,
    novel_model_factory
)
from utils.indications import INDICATION_VAR_NAME
from utils.model.albumin import albumin_model_factory
from utils.model.lactate import lactate_model_factory
from utils.model.shared import LogOddsTransformer
from utils.evaluate import (
    Scorer,
    LogisticScorer,
    score_linear_predictions,
    score_logistic_predictions
)
from utils.impute import ImputationInfo
from utils.split import TrainTestSplitter
from utils.plot.helpers import plot_saver
from utils.plot.pdp import PDPTerm, PDPFigure
from utils.plot.evaluate import (
    plot_calibration,
    plot_stratified_risk_distributions
)
from utils.filter import StratifiedDispersionQuantifier
from utils.report import Reporter


reporter = Reporter()
reporter.title(
    "Retrain models (lactate imputer, albumin imputer and novel model) on "
    "whole dataset for use in production. Test each on whole dataset (for "
    "purposes on sanity checking rather than proper validation)."
)


reporter.report("Loading previous analysis outputs")
df = pd.read_pickle(
    os.path.join(DATA_DIR, "05_preprocessed_df.pkl")
)
multi_category_levels: Dict[str, Tuple] = load_object(
    os.path.join(
        NOVEL_MODEL_OUTPUT_DIR,
        "05_multi_category_levels_with_indications.pkl"
    )
)
imputation_stages: ImputationInfo = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "05_imputation_stages.pkl")
)
lacalb_pdp_terms: List[PDPTerm] = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "07_pd_plot_specification.pkl")
)
novel_pdp_terms: List[PDPTerm] = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "08_pd_plot_specification.pkl")
)
point_prediction_novel_model_scorer: LogisticScorer = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "08_novel_model_scorer.pkl")
)


reporter.report("Loading manually-wrangled NELA data")
df_for_tt_splitter = load_nela_data_and_sanity_check()


reporter.title(
    "Making new train-test splitter containing a single split, in which both "
    "the train and test folds contain all cases in the dataset."
)
tt_splitter_production = TrainTestSplitter(
    df=df_for_tt_splitter,
    current_nela_model_vars=flatten_model_var_dict(CURRENT_MODEL_VARS),
    random_seed=RANDOM_SEED,
    split_variable_name="HospitalId.anon",
    test_fraction=0.0,
    n_splits=1
)
tt_splitter_production.split()
tt_splitter_production.test_i = copy.deepcopy(tt_splitter_production.train_i)
tt_splitter_production.test_institution_ids = copy.deepcopy(
    tt_splitter_production.train_institution_ids
)
tt_splitter_production.drop_stats = None
tt_splitter_production.split_stats = None
tt_splitter_production.complete_case_df = None
tt_splitter_production.test_fraction = None


reporter.report("Saving new train-test splitter")
save_object(
    tt_splitter_production,
    os.path.join(INTERNAL_OUTPUT_DIR, "13_train_test_splitter_production.pkl")
)


reporter.title("Re-running MICE for production models")


reporter.report("Making DataFrame for use in MICE")
mice_df = df.drop(
    list(multi_category_levels.keys()) + [LACTATE_VAR_NAME, ALBUMIN_VAR_NAME],
    axis=1
).copy()


reporter.report("Making list of binary variables for use in MICE")
binary_vars = list(
    set(NOVEL_MODEL_VARS["cat"]) - set(multi_category_levels.keys())
)


reporter.report("Making list of continuous variables for use in MICE")
mice_cont_vars = list(NOVEL_MODEL_VARS["cont"])
mice_cont_vars.remove(LACTATE_VAR_NAME)
mice_cont_vars.remove(ALBUMIN_VAR_NAME)


reporter.report("Running MICE")
swm = SplitterWinsorMICE(
    df=mice_df,
    train_test_splitter=tt_splitter_production,
    target_variable_name=NOVEL_MODEL_VARS["target"],
    cont_variables=mice_cont_vars,
    binary_variables=binary_vars,
    winsor_quantiles=WINSOR_QUANTILES,
    winsor_include={
        "S01AgeOnArrival": (False, True),
        "S03GlasgowComaScore": (False, False),
    },
    n_mice_imputations=imputation_stages.n_imputations[0],
    n_mice_burn_in=10,
    n_mice_skip=3,
    random_seed=RANDOM_SEED
)
swm.split_winsorize_mice()


reporter.report("Saving MICE imputations")
save_object(
    swm,
    os.path.join(
        NOVEL_MODEL_OUTPUT_DIR,
        "13_splitter_winsor_mice_production.pkl"
    )
)


reporter.title("Re-running categorical imputation for production models")
cat_imputer = CategoricalImputer(
    df=df.drop([LACTATE_VAR_NAME, ALBUMIN_VAR_NAME], axis=1).copy(),
    splitter_winsor_mice=swm,
    cat_vars=list(multi_category_levels.keys()),
    random_seed=RANDOM_SEED
)
cat_imputer.impute()


reporter.report("Saving non-binary categorical imputer")
save_object(
    cat_imputer,
    os.path.join(
        NOVEL_MODEL_OUTPUT_DIR,
        "13_categorical_imputer_production.pkl"
    )
)


reporter.title(
    "Training & sanity checking production albumin and lactate models"
)


imputers = {}
for name, pretty_name, variable_name, model_factory in (
    ('albumin', 'Albumin (g/L)', ALBUMIN_VAR_NAME, albumin_model_factory),
    ('lactate', 'Lactate (mmol/L)', LACTATE_VAR_NAME, lactate_model_factory)
):
    reporter.report(f"Fitting imputers for {name}")
    imputers[name] = LactateAlbuminImputer(
        df=df.loc[:, [variable_name, NOVEL_MODEL_VARS["target"]]].copy(),
        categorical_imputer=cat_imputer,
        lacalb_variable_name=variable_name,
        imputation_model_factory=model_factory,
        winsor_quantiles=WINSOR_QUANTILES,
        multi_cat_vars=multi_category_levels,
        indication_var_name=INDICATION_VAR_NAME,
        mortality_as_feature=False,
        random_seed=RANDOM_SEED)
    imputers[name].fit()


    reporter.report(f"Saving {name} imputer for later use")
    save_object(
        imputers[name],
        os.path.join(
            NOVEL_MODEL_OUTPUT_DIR,
            f"13_{name}_imputer_production.pkl"
        )
    )


    reporter.report(f"Scoring {name} imputation model performance.")
    y_obs, y_preds = imputers[name].get_all_observed_and_predicted(
        fold_name='test',
        probabilistic=False,
        lac_alb_imp_i=None)
    lacalb_scorer = Scorer(
        y_true=y_obs,
        y_pred=y_preds,
        scorer_function=score_linear_predictions,
        n_splits=imputers[name].tts.n_splits)
    lacalb_scorer.calculate_scores()
    lacalb_scorer.print_scores(dec_places=3, point_estimate='median')


    reporter.first("Saving model scorer for later use")
    save_object(
        lacalb_scorer,
        os.path.join(
            NOVEL_MODEL_OUTPUT_DIR,
            f"13_{name}_imputer_scorer_production.pkl"
        )
    )


    reporter.report('Preparing data for PDP histograms')
    pdp_hist_data = pd.concat(
        objs=(
            cat_imputer.get_imputed_df('train', 0, 0).drop(
                cat_imputer.target_variable_name, axis=1
            ),
            cat_imputer.get_imputed_df('test', 0, 0).drop(
                cat_imputer.target_variable_name, axis=1
            ),
        ),
        axis=0,
        ignore_index=True
    )


    reporter.report(f"Plotting {name} imputer partial dependence plots")
    for hist_switch, hist_text in ((False, ''), (True, '_with_histograms')):
        for space, ylabel, kwargs in (
            ('gaussian', None, {}),
            ('inv_trans',
             pretty_name,
             {
                 'transformer': imputers[name].transformers[0],
                 'plot_just_outer_ci': True
             })
        ):
            pdp_generator = PDPFigure(
                gam=imputers[name].imputers[0],
                pdp_terms=lacalb_pdp_terms,
                ylabel=ylabel,
                plot_hists=hist_switch,
                hist_data=pdp_hist_data,
                **kwargs
            )
            plot_saver(
                pdp_generator.plot,
                output_dir=FIGURES_OUTPUT_DIR,
                output_filename=(
                    f"13_{name}_imputer_{space}_pd_plot{hist_text}_production"
                )
            )


reporter.title("Training & sanity checking production novel model")


reporter.report("Training novel model")
novel_model = NovelModel(
    categorical_imputer=cat_imputer,
    albumin_imputer=imputers['albumin'],
    lactate_imputer=imputers['lactate'],
    model_factory=novel_model_factory,
    n_lacalb_imputations_per_mice_imp=(
        imputation_stages.multiple_of_previous_n_imputations[1]),
    random_seed=RANDOM_SEED
)
novel_model.fit()


reporter.report("Saving novel model")
save_object(
    novel_model,
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "13_novel_model_production.pkl")
)


reporter.report('Preparing data for PDP histograms')
pdp_hist_data = pd.concat(
    objs=(
        novel_model.get_features_and_labels('train', 0, 0, 0)[0],
        novel_model.get_features_and_labels('test', 0, 0, 0)[0]
    ),
    axis=0,
    ignore_index=True
)


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
            pdp_terms=novel_pdp_terms,
            ylabel=pretty_space_name,
            plot_hists=hist_switch,
            hist_data=pdp_hist_data, **kwargs)
        plot_saver(
            pdp_generator.plot,
            output_dir=FIGURES_OUTPUT_DIR,
            output_filename=(
                f"13_novel_model_{space_name}_pd_plot{hist_text}_production"
            )
        )


reporter.report(
    "Find calibration GAM best lams from point-prediction novel model scorer"
)
best_lams = np.array([
    nested_lam[0][0] for nested_lam in
    point_prediction_novel_model_scorer.calib_lams
])


reporter.report(
    "Restricting calibration GAM fitting for risk distributions to the 3 "
    "most-frequently-chosen lams during previous novel model fitting, as "
    "calibration GAM fitting is slow with such large datasets."
)
unique_lams, counts_per_unique_lam = np.unique(best_lams, return_counts=True)
top_3_lams = unique_lams[counts_per_unique_lam.argsort()][-3:]
reporter.report(f"Top 3 lams are f{top_3_lams}")


reporter.report("Generating predicted risk distributions.")
y_preds, y_obs_repeated, y_preds_flat = [], [], []
for split_i in pb(
    range(novel_model.cat_imputer.tts.n_splits),
    prefix="Split iteration"
):
    y_ob, y_pred = novel_model.get_observed_and_predicted(
        fold_name='test',
        split_i=split_i,
        n_samples_per_imp_i=5
    )
    # NB. Shape of y_ob is (n_patients,)
    # NB. Shape of y_pred is (n_predicted_probabilities, n_patients)
    y_preds.append(y_pred)
    y_obs_repeated.append(np.repeat(y_ob, y_pred.shape[0]))
    y_preds_flat.append(y_pred.flatten(order='F'))


reporter.report(f"{y_preds[0].shape[0]} samples form each risk distribution")


reporter.report(
    "Quantifying dispersion of predicted risk distributions, stratified by "
    "imputation method"
)
dispersion_quantifier = StratifiedDispersionQuantifier(
    y_pred_samples=y_preds,
    novel_model=novel_model,
    fold_name='test'
)
dispersion_quantifier.calculate_dispersion_and_stratify()


reporter.first('Printing stratified dispersion')
PrettyPrinter().pprint(dispersion_quantifier.per_split_median_95ci)


reporter.first("Saving dispersion quantifier for later use")
save_object(
    dispersion_quantifier,
    os.path.join(
        NOVEL_MODEL_OUTPUT_DIR,
        "13_dispersion_quantifier_production.pkl"
    )
)


reporter.report("Scoring novel model performance.")
scorer = LogisticScorer(
    y_true=y_obs_repeated,
    y_pred=y_preds_flat,
    scorer_function=score_logistic_predictions,
    n_splits=novel_model.cat_imputer.tts.n_splits,
    calibration_n_splines=CALIB_GAM_N_SPLINES,
    calibration_lam_candidates=top_3_lams
)
scorer.calculate_scores()
scorer.print_scores(dec_places=3, point_estimate='median')


reporter.first("Saving model scorer for later use")
save_object(
    scorer,
    os.path.join(
        NOVEL_MODEL_OUTPUT_DIR,
        "13_novel_model_samples_scorer_production.pkl"
    )
)


reporter.title("Generating some more plots for the production novel model")
reporter.report("Plotting calibration curves")
plot_saver(
    plot_calibration,
    p=scorer.p,
    calib_curves=scorer.calib_curves,
    curve_transparency=0.15,
    output_dir=FIGURES_OUTPUT_DIR,
    output_filename=f"13_novel_model_calibration_production",
)

reporter.report("Plotting risk distributions")
plot_saver(
    plot_stratified_risk_distributions,
    y_true=np.hstack(scorer.y_true),
    y_pred=np.hstack(scorer.y_pred),
    output_dir=FIGURES_OUTPUT_DIR,
    output_filename=f"13_novel_model_risk_distributions_production",
)


reporter.last("Done.")
