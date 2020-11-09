import os

from utils.constants import CURRENT_MODEL_OUTPUT_DIR, NOVEL_MODEL_OUTPUT_DIR
from utils.evaluate import LogisticScorer, ScoreComparer
from utils.io import load_object, save_object
from utils.filter import (
    get_indices_of_case_imputed_using_target,
    filter_y_and_rescore
)
from utils.model.novel import SplitterWinsorMICE, CategoricalImputer
from utils.report import Reporter


reporter = Reporter()
reporter.title(
    "Compare the performance of the current and novel emergency laparotomy "
    "mortality risk models."
)


reporter.report("Loading previous analysis outputs")
current_scorer: LogisticScorer = load_object(
    os.path.join(CURRENT_MODEL_OUTPUT_DIR, "02_current_model_scorer.pkl")
)
novel_scorer: LogisticScorer = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "08_novel_model_scorer.pkl")
)
swm: SplitterWinsorMICE = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "05_splitter_winsor_mice.pkl")
)
cat_imputer: CategoricalImputer = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "06_categorical_imputer.pkl")
)


reporter.report("Calculating difference between scores for each model on each "
                "train-test split")
score_comparer = ScoreComparer(
    scorers=(current_scorer, novel_scorer),
    scorer_names=('current', 'novel')
)
score_comparer.compare_scores()
reporter.first("Difference in scores with median as difference point estimate:")
score_comparer.print_scores(dec_places=4, point_estimate='median')
reporter.first("Difference with split 0 as difference point estimate:")
score_comparer.print_scores(dec_places=4, point_estimate='split0')


reporter.first("Saving score comparer for later use")
save_object(
    score_comparer,
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "10_score_comparer.pkl"))


reporter.report("Recalculating scores for both model when we exclude test fold "
                "cases where the features are imputed using the mortality "
                "labels.")
imputed_using_target_indices = get_indices_of_case_imputed_using_target(
    fold_name='test',
    splitter_winsor_mice=swm,
    categorical_imputer=cat_imputer
)
non_target_imputation_scorers = {
    'current': filter_y_and_rescore(
        scorer=current_scorer,
        indices=imputed_using_target_indices,
        invert_index=True),
    'novel': filter_y_and_rescore(
        scorer=novel_scorer,
        indices=imputed_using_target_indices,
        invert_index=True)
}
non_target_imputation_scorers['comparer'] = ScoreComparer(
    scorers=(
        non_target_imputation_scorers['current'],
        non_target_imputation_scorers['novel']),
    scorer_names=('current', 'novel')
)
reporter.first("Difference in scores with median as difference point estimate:")
non_target_imputation_scorers['comparer'].print_scores(
    dec_places=4,
    point_estimate='median')
reporter.first("Difference with split 0 as difference point estimate:")
non_target_imputation_scorers['comparer'].print_scores(
    dec_places=4,
    point_estimate='split0')


reporter.first("Saving recalculated scores for later use")
save_object(
    score_comparer,
    os.path.join(
        NOVEL_MODEL_OUTPUT_DIR,
        "10_non_target_imputation_scorers.pkl"))


reporter.last("Done.")
