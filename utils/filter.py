from typing import Set, Dict, Union, List
import copy
import numpy as np
from progressbar import progressbar as pb

from utils.evaluate import Scorer, LogisticScorer
from utils.model.novel import (
    SplitterWinsorMICE,
    CategoricalImputer,
    NovelModel
)


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


class StratifiedDispersionQuantifier:
    """Calculate 95% CI range (range between the 2.5th and 97.5th percentiles)
        for every predicted risk distribution in the specified fold for every
        train-test split. Stratified the 95% CI ranges by the different types
        of imputation that has been done on those cases."""
    def __init__(
        self,
        y_pred_samples: List[np.ndarray],
        novel_model: NovelModel,
        fold_name: str
    ):
        """Each element of y_pred_samples had shape
            (n_predicted_probabilities, n_patients)"""
        self.y_pred_samples = y_pred_samples
        self.swm = novel_model.cat_imputer.swm
        self.cat_imputer = novel_model.cat_imputer
        self.alb_imputer = novel_model.alb_imputer
        self.lac_imputer = novel_model.lac_imputer
        self.n_splits = novel_model.cat_imputer.tts.n_splits
        self.fold_name = fold_name
        self.per_case_ranges: List[np.ndarray] = []
        self.per_split_95ci = self._init_per_split_95ci()
        self.mice_cat_imputed_case_i = get_indices_of_case_imputed_using_target(
            fold_name,
            novel_model.cat_imputer.swm,
            novel_model.cat_imputer
        )

    def calculate_dispersion_and_stratify(self):
        """Calculate 95% CI range for every train-test split. Stratifies
            these."""
        for split_i in pb(range(self.n_splits), prefix="Split iteration"):
            self._calculate_y_pred_95ci_range(split_i)
            self._stratify(split_i)

    def _init_per_split_95ci(self) -> Dict[str, np.ndarray]:
        """Each value in dict is of shape (n_splits, 2)"""
        per_split_range_95ci = {}
        for description in (
            'no_imputation',
            'just_mice_cat_imputation',
            'just_alb_imputation',
            'just_lac_imputation',
            'just_lac_alb_imputation'
        ):
            per_split_range_95ci[description] = np.zeros((self.n_splits, 2))
        return per_split_range_95ci

    def _calculate_y_pred_95ci_range(self, split_i: int):
        """Returned array is of shape (n_patients,). Each element is the range
            between the 2.5th and 97.5th percentiles of the predicted risk
            distribution for each patient."""
        y_pred_percentiles = np.percentile(
            self.y_pred_samples[split_i], (2.5, 97.5), axis=0)
        self.per_case_ranges.append(
            y_pred_percentiles[1, :] - y_pred_percentiles[0, :])

    def _stratify(self, split_i: int):
        all_case_i = set(range(self.y_pred_samples[split_i].shape[0]))
        non_mice_cat_imputed_case_i = (
            all_case_i - self.mice_cat_imputed_case_i[split_i]
        )
        lac_imputed_case_i = set(
            self.lac_imputer.missing_i[self.fold_name][split_i][
                'S03PreOpArterialBloodLactate']
        )
        alb_imputed_case_i = set(
            self.alb_imputer.missing_i[self.fold_name][split_i][
                'S03PreOpLowestAlbumin']
        )
        no_imputation_case_i = (
            non_mice_cat_imputed_case_i -
            lac_imputed_case_i
        ) - alb_imputed_case_i
        just_mice_cat_imputation_case_i = (
            self.mice_cat_imputed_case_i[split_i] - alb_imputed_case_i
        ) - lac_imputed_case_i
        just_alb_imputation_case_i = (
            alb_imputed_case_i - lac_imputed_case_i
        ) - self.mice_cat_imputed_case_i[split_i]
        just_lac_imputation_case_i = (
            lac_imputed_case_i - alb_imputed_case_i
        ) - self.mice_cat_imputed_case_i[split_i]
        just_lac_alb_imputation_case_i = (
            lac_imputed_case_i.intersection(
                alb_imputed_case_i)
        ) - self.mice_cat_imputed_case_i[split_i]
        self.per_split_95ci['no_imputation'][split_i, :] = (
            self._calculate_per_split_95ci(split_i, no_imputation_case_i))
        self.per_split_95ci['just_mice_cat_imputation'][split_i, :] = (
            self._calculate_per_split_95ci(
                split_i, just_mice_cat_imputation_case_i))
        self.per_split_95ci['just_alb_imputation'][split_i, :] = (
            self._calculate_per_split_95ci(
                split_i, just_alb_imputation_case_i))
        self.per_split_95ci['just_lac_imputation'][split_i, :] = (
            self._calculate_per_split_95ci(
                split_i, just_lac_imputation_case_i))
        self.per_split_95ci['just_lac_alb_imputation'][split_i, :] = (
            self._calculate_per_split_95ci(
                split_i, just_lac_alb_imputation_case_i))

    def _calculate_per_split_95ci(
        self,
        split_i: int,
        case_subset_indices: Set[int]
    ) -> np.array:
        mask = np.in1d(
            range(self.per_case_ranges[split_i].size),
            list(case_subset_indices))
        return np.percentile(self.per_case_ranges[split_i][mask], (2.5, 97.5))


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
