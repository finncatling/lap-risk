# -*- coding: utf-8 -*-

RANDOM_SEED = 1

"""NB. NB. Web form for NELA risk model includes Hb,
    but only uses this to calculate P-POSSUM."""
CURRENT_NELA_RISK_MODEL_VARS = {
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

# For use when plotting gam output
N_GAM_CONFIDENCE_INTERVALS = 5
GAM_OUTER_CONFIDENCE_INTERVALS = (0.025, 0.975)
