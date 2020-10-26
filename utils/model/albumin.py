from typing import Dict, Tuple

import pandas as pd
from pygam import LinearGAM, s, f, te


def albumin_model_factory(
    columns: pd.Index,
    multi_cat_levels: Dict[str, Tuple],
    indication_var_name: str
) -> LinearGAM:
    return LinearGAM(
        s(columns.get_loc("S01AgeOnArrival"), lam=300)
        + s(columns.get_loc("S03SystolicBloodPressure"), lam=300)
        + te(
            columns.get_loc("S03Pulse"),
            columns.get_loc("S03ECG"),
            lam=(200, 5),
            n_splines=(20, 2),
            spline_order=(3, 0),
            dtype=("numerical", "categorical"),
        )
        + s(columns.get_loc("S03WhiteCellCount"), lam=300)
        + s(columns.get_loc("S03Sodium"), lam=300)
        + s(columns.get_loc("S03Potassium"), lam=300)
        + s(
            columns.get_loc("S03GlasgowComaScore"),
            n_splines=13,
            spline_order=0,
            lam=200
        )
        + f(columns.get_loc("S03ASAScore"), coding="dummy", lam=25)
        + te(
            columns.get_loc("S03DiagnosedMalignancy"),
            columns.get_loc("S02PreOpCTPerformed"),
            lam=(100, 5),
            n_splines=(len(multi_cat_levels["S03DiagnosedMalignancy"]), 2),
            spline_order=(0, 0),
            dtype=("categorical", "categorical"),
        )
        + te(
            columns.get_loc("S03Pred_Peritsoil"),
            columns.get_loc("S02PreOpCTPerformed"),
            lam=(50, 10),
            n_splines=(len(multi_cat_levels["S03Pred_Peritsoil"]), 2),
            spline_order=(0, 0),
            dtype=("categorical", "categorical"),
        )
        + te(
            columns.get_loc("S03CardiacSigns"),
            columns.get_loc("S03RespiratorySigns"),
            lam=30,
            n_splines=(
                len(multi_cat_levels["S03CardiacSigns"]),
                len(multi_cat_levels["S03RespiratorySigns"])
            ),
            spline_order=(0, 0),
            dtype=("categorical", "categorical"),
        )
        + te(
            columns.get_loc("S03SerumCreatinine"),
            columns.get_loc("S03Urea"),
            lam=40,
            dtype=("numerical", "numerical"),
        )
        + te(
            columns.get_loc(indication_var_name),
            columns.get_loc("S02PreOpCTPerformed"),
            lam=(60, 2),
            n_splines=(len(multi_cat_levels[indication_var_name]), 2),
            spline_order=(0, 0),
            dtype=("categorical", "categorical"),
        )
    )
