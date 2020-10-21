from typing import Dict, Tuple

import pandas as pd
from pygam import LinearGAM, s, f, te


def albumin_model_factory(
    columns: pd.Index,
    multi_cat_levels: Dict[str, Tuple],
    indication_var_name: str
) -> LinearGAM:
    # TODO: White et al recommends using all analysis model variables in the
    #  imputation model
    # TODO: Check creatinine features in DataFrame - why is it causing
    #  divergence? Should it interact with urea?
    # TODO: Should GCS splines have lower order?
    # TODO: Consider edge knots
    # TODO: Consider reducing n_splines for most continuous variables
    return LinearGAM(
        s(columns.get_loc("S01AgeOnArrival"), lam=500)
        + s(columns.get_loc("S03Sodium"), lam=400)
        + s(columns.get_loc("S03Potassium"), lam=300)
        + s(columns.get_loc("S03Urea"), lam=400)
        + s(columns.get_loc("S03WhiteCellCount"), lam=400)
        + s(columns.get_loc("S03SystolicBloodPressure"), lam=400)
        + f(columns.get_loc("S03ASAScore"), coding="dummy")
        + te(
            columns.get_loc("S03DiagnosedMalignancy"),
            columns.get_loc("S02PreOpCTPerformed"),
            lam=(2, 1),
            n_splines=(len(multi_cat_levels["S03DiagnosedMalignancy"]), 2),
            spline_order=(0, 0),
            dtype=("categorical", "categorical"),
        )
        + te(
            columns.get_loc("S03Pred_Peritsoil"),
            columns.get_loc("S02PreOpCTPerformed"),
            lam=(2, 1),
            n_splines=(len(multi_cat_levels["S03Pred_Peritsoil"]), 2),
            spline_order=(0, 0),
            dtype=("categorical", "categorical"),
        )
        + te(
            columns.get_loc(indication_var_name),
            columns.get_loc("S02PreOpCTPerformed"),
            lam=(2, 1),
            n_splines=(len(multi_cat_levels[indication_var_name]), 2),
            spline_order=(0, 0),
            dtype=("categorical", "categorical"),
        )
        + te(
            columns.get_loc("S03Pulse"),
            columns.get_loc("S03ECG"),
            lam=(400, 2),
            n_splines=(20, 2),
            spline_order=(3, 0),
            dtype=("numerical", "categorical"),
        )
    )
