from utils.model import shared
from utils.model.current import CURRENT_MODEL_VARS


def test_flatten_model_var_dict():
    column_name_list = shared.flatten_model_var_dict(CURRENT_MODEL_VARS)
    assert isinstance(column_name_list, list)
    assert all(isinstance(col_name, str) for col_name in column_name_list)
