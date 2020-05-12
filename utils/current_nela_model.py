from typing import Dict, List

# -*- coding: utf-8 -*-
"""NB. Web form for current NELA risk model includes Hb, but only uses this to
calculate P-POSSUM."""
CURRENT_NELA_MODEL_VARS = {
    'cat': ('S01Sex',
            'S03ASAScore',
            'S03NCEPODUrgency',
            'S03ECG',
            'S03NumberOfOperativeProcedures',
            'S03CardiacSigns',
            'S03RespiratorySigns',
            'S03Pred_Peritsoil',
            'S03Pred_TBL',
            'S03DiagnosedMalignancy',
            'S03WhatIsTheOperativeSeverity',
            'S03GlasgowComaScore'),
    'cont': ('S01AgeOnArrival',
             'S03SerumCreatinine',
             'S03Sodium',
             'S03Potassium',
             'S03Urea',
             'S03WhiteCellCount',
             'S03Pulse',
             'S03SystolicBloodPressure'),
    'target': 'Target'}


def flatten_nela_var_dict(nela_vars: Dict) -> List[str]:
    """Flattens current NELA model variable name dict into single list."""
    return (list(nela_vars['cat']) + list(nela_vars['cont']) +
            [nela_vars['target']])
