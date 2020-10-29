from typing import Tuple, Dict
import os
import pandas as pd

from utils.report import Reporter
from utils.constants import NOVEL_MODEL_OUTPUT_DIR, RANDOM_SEED
from utils.io import load_object, save_object
from utils.model.novel import (
    CategoricalImputer,
    LactateAlbuminImputer,
    NovelModel,
    novel_model_factory
)
from utils.impute import ImputationInfo


reporter = Reporter()
reporter.title(
    "Fit novel emergency laparotomy mortality risk model "
    "on the different train folds, and evaluate the models "
    "obtained on the corresponding test folds"
)


reporter.report("Loading previous analysis outputs needed for novel model")
cat_imputer: CategoricalImputer = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "06_categorical_imputer.pkl")
)
albumin_imputer: LactateAlbuminImputer = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "07_albumin_imputer.pkl")
)
lactate_imputer: LactateAlbuminImputer = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "07_lactate_imputer.pkl")
)
imputation_stages: ImputationInfo = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "05_imputation_stages.pkl")
)


reporter.report("Beginning train-test splitting and model fitting")
novel_model = NovelModel(
    categorical_imputer=cat_imputer,
    albumin_imputer=albumin_imputer,
    lactate_imputer=lactate_imputer,
    model_factory=novel_model_factory,
    n_lacalb_imputations_per_mice_imp=(
        imputation_stages.multiple_of_previous_n_imputations[1]),
    random_seed=RANDOM_SEED
)
novel_model.cat_imputer.tts.n_splits = 1  # TODO: Remove this testing line
novel_model.fit()


reporter.report(f"Saving draft novel model for later use")
save_object(
    novel_model,
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, f"08_draft_novel_model.pkl"))


# TODO: Finish this script
