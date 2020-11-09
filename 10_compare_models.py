import os

from utils.constants import CURRENT_MODEL_OUTPUT_DIR, NOVEL_MODEL_OUTPUT_DIR
from utils.evaluate import LogisticScorer, ScoreComparer
from utils.io import load_object, save_object
from utils.report import Reporter


reporter = Reporter()
reporter.title(
    "Compare the performance of the current and novel emergency laparotomy "
    "mortality risk models."
)


reporter.report("Loading results of model scoring")
current_scorer: LogisticScorer = load_object(
    os.path.join(CURRENT_MODEL_OUTPUT_DIR, "02_current_model_scorer.pkl")
)
novel_scorer: LogisticScorer = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "08_novel_model_scorer.pkl")
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


reporter.first("Saving model scorer for later use")
save_object(
    score_comparer,
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "10_score_comparer.pkl"))


reporter.last("Done.")
