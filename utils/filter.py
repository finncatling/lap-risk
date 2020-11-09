from typing import Set, Dict, Union
import copy
import numpy as np

from utils.evaluate import Scorer, LogisticScorer
from utils.model.novel import SplitterWinsorMICE, CategoricalImputer


def get_indices_of_case_imputed_using_target(
    fold_name: str,
    splitter_winsor_mice: SplitterWinsorMICE,
    categorical_imputer: CategoricalImputer
) -> Dict[int, Set[int]]:
    """Find indices of cases where some features are imputed by
        SplitterWinsorMICE and/or CategoricalImputer. These are the cases where
        mortality labels are used as a feature in the imputation.

    Args:
        fold_name: In {'train', 'test'}. NB. the test folds will contain far
            fewer imputed cases as all test fold cases are
            current-model-variable complete
        splitter_winsor_mice: Fitted SplitterWinsorMICE instance
        categorical_imputer: Fitted CategoricalImputer instance

    Returns:
        Format is {0: {4, 7, ...}, 1: {3, 12, ..), ...} with the dict's integer
            keys enumerating the train-test splits. The integers in the sets
            index the cases from the specified fold in each split where some
            features are imputed by SplitterWinsorMICE and/or
            CategoricalImputer
    """
    all_missing_i = {}
    for i in range(splitter_winsor_mice.tts.n_splits):
        all_missing_i[i] = set(np.hstack(
            list(splitter_winsor_mice.missing_i[fold_name][i].values()) +
            list(categorical_imputer.missing_i[fold_name][i].values())
        ))
    return all_missing_i


def filter_y_and_rescore(
    scorer: Union[Scorer, LogisticScorer],
    indices: Dict[int, Set[int]],
    invert_index: bool
) -> Scorer:
    """Filters the y_true labels and y_pred predictions in each split of a
        Scorer or LogisticScorer, and recalculates model scores. This allows
        calculation of model performance on a subset of patients.

    Args:
        scorer: The model scorer. Can have had .calculate_scores() run
            previously. This will be rerun within this function.
        indices: Format is {0: {4, 7, ...}, 1: {3, 12, ..), ...} with the
            dict's integer keys enumerating the train-test splits. The integers
            in sets index the cases that should be retained if invert_index is
            False, or that should be filtered out if invert_index is True
        invert_index: see above

    Returns:
        The model scorer with y_true and y_pred, and recalculated scores.
    """
    scorer = copy.deepcopy(scorer)
    for i in range(len(scorer.y_true)):
        mask = np.in1d(range(scorer.y_true[i].shape[0]), list(indices))
        if invert_index:
            mask = ~mask
        scorer.y_true[i] = scorer.y_true[i][mask]
        scorer.y_pred[i] = scorer.y_pred[i][mask]
    scorer.calculate_scores()
    return scorer
