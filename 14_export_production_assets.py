import copy
import os

import numpy as np

from utils.constants import NOVEL_MODEL_OUTPUT_DIR, PRODUCTION_OUTPUT_DIR
from utils.io import load_object, save_object
from utils.model.novel import NovelModel
from utils.report import Reporter


reporter = Reporter()
reporter.title(
    "Export (non-PHI) assets needed to deploy the novel model, plus the "
    "lactate and albumin imputation models, in production."
)


reporter.report("Loading pretrained production novel model")
novel_model: NovelModel = load_object(
    os.path.join(
        NOVEL_MODEL_OUTPUT_DIR,
        '13_novel_model_production.pkl'
    )
)


reporter.report("Storing production models")
assets = {
    'albumin': {'model': copy.deepcopy(novel_model.alb_imputer.imputers[0])},
    'lactate': {'model': copy.deepcopy(novel_model.lac_imputer.imputers[0])},
    'mortality': {'model': novel_model.models[0]}
}


reporter.report("Consolidating and storing Winsorization thresholds")
assets['winsor_thresholds'] = copy.deepcopy(
    novel_model.cat_imputer.swm.winsor_thresholds[0]
)
assets['winsor_thresholds'][
    novel_model.alb_imputer.lacalb_variable_name
] = copy.deepcopy(novel_model.alb_imputer.winsor_thresholds[0])
assets['winsor_thresholds'][
    novel_model.lac_imputer.lacalb_variable_name
] = copy.deepcopy(novel_model.lac_imputer.winsor_thresholds[0])
assets['winsor_thresholds']['S01AgeOnArrival'][0] = 18.0
del assets['winsor_thresholds']['S03GlasgowComaScore']


reporter.report(
    "Storing format of the input data for each model (not the data itself)"
)
model_input_data = {  # This dictionary is NOT for export with other assets
    'albumin': novel_model.alb_imputer._get_features_where_lacalb_missing(
        fold_name='train',
        split_i=0,
        mice_imp_i=0
    ),
    'lactate': novel_model.lac_imputer._get_features_where_lacalb_missing(
        fold_name='train',
        split_i=0,
        mice_imp_i=0
    ),
    'mortality': novel_model.get_features_and_labels(
        fold_name='train',
        split_i=0,
        mice_imp_i=0,
        lac_alb_imp_i=0
    )[0]
}

for model_name, features in model_input_data.items():
    assets[model_name]['input_data'] = {
        'dtypes': copy.deepcopy(features.dtypes),
        'describe': features.describe(),
        'unique_categories': {}
    }
    for c in features[
        features.columns.difference(list(assets['winsor_thresholds'].keys()))
    ].columns:
        assets[model_name]['input_data']['unique_categories'][c] = np.sort(
            features[c].unique()
        )


reporter.report("Storing transformer for both imputation models")
for model_name in ('albumin', 'lactate'):
    assets[model_name]['transformer'] = novel_model.alb_imputer.transformers[0]


reporter.report("Storing label encoding for non-binary categorical labels")
assets['label_encoding'] = load_object(
    os.path.join(
        NOVEL_MODEL_OUTPUT_DIR,
        "05_multi_category_levels_with_indications.pkl"
    )
)


reporter.report("Exporting all stored assets")
save_object(
    assets,
    os.path.join(PRODUCTION_OUTPUT_DIR, 'production_assets.pkl')
)


reporter.last('Done.')
