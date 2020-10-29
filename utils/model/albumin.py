from typing import Dict, Tuple

import pandas as pd
from pygam import LinearGAM, s, f, te


def albumin_model_factory(
    columns: pd.Index,
    multi_cat_levels: Dict[str, Tuple],
    indication_var_name: str
) -> LinearGAM:
    return LinearGAM(
        s(
            columns.get_loc("S01AgeOnArrival"),
            spline_order=2,
            n_splines=10,
            lam=75
        )
        + s(
            columns.get_loc("S03SystolicBloodPressure"),
            spline_order=2,
            n_splines=10,
            lam=50
        )
        + te(
            columns.get_loc("S03Pulse"),
            columns.get_loc("S03ECG"),
            lam=(250, 20),
            n_splines=(20, 2),
            spline_order=(3, 0),
            dtype=("numerical", "categorical"),
        )
        + s(
            columns.get_loc("S03WhiteCellCount"),
            spline_order=2,
            n_splines=10,
            lam=25
        )
        + s(
            columns.get_loc("S03Sodium"),
            spline_order=2,
            n_splines=10,
            lam=60
        )
        + s(
            columns.get_loc("S03Potassium"),
            spline_order=2,
            n_splines=10,
            lam=25
        )
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
            lam=(70, 20),
            n_splines=(len(multi_cat_levels["S03Pred_Peritsoil"]), 2),
            spline_order=(0, 0),
            dtype=("categorical", "categorical"),
        )
        + te(
            columns.get_loc("S03CardiacSigns"),
            columns.get_loc("S03RespiratorySigns"),
            lam=80,
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
            spline_order=2,
            lam=90,
            dtype=("numerical", "numerical"),
        )
        + te(
            columns.get_loc(indication_var_name),
            columns.get_loc("S02PreOpCTPerformed"),
            lam=(80, 5),
            n_splines=(len(multi_cat_levels[indication_var_name]), 2),
            spline_order=(0, 0),
            dtype=("categorical", "categorical"),
        )
    )
