import os

import numpy as np
import pandas as pd

from utils.constants import (
    RAW_NELA_DATA_FILEPATH,
    DATA_DIR,
    NELA_DATA_FILEPATH,
    STATS_OUTPUT_DIR,
    CURRENT_MODEL_OUTPUT_DIR,
    NOVEL_MODEL_OUTPUT_DIR,
    FIGURES_OUTPUT_DIR,
    TABLES_OUTPUT_DIR
)
from utils.report import Reporter
from utils.wrangling import (
    drop_values_under_threshold,
    remap_categories,
    remove_non_whole_numbers
)
from utils.io import make_directory


reporter = Reporter()
reporter.title("Initial data wrangling")


reporter.report("Creating output directories (if they don't already exist)")
make_directory(STATS_OUTPUT_DIR)
make_directory(FIGURES_OUTPUT_DIR)
make_directory(TABLES_OUTPUT_DIR)
make_directory(NOVEL_MODEL_OUTPUT_DIR)
make_directory(CURRENT_MODEL_OUTPUT_DIR)


reporter.report('Loading raw data')
df = pd.read_csv(RAW_NELA_DATA_FILEPATH)
print(f'Dataset contains {df.shape[0]} cases')


reporter.first('Processing S01AgeOnArrival')
reporter.report('Excluding patients < 18 years old')
df = df.loc[df['S01AgeOnArrival'] > 17]
print(df.shape[0], 'cases remain')
reporter.report('Excluding patients >= 110')
df = df.loc[df['S01AgeOnArrival'] < 110]
print(df.shape[0], 'cases remain')


reporter.first('Preprocessing HospitalId.anon')
# Remove prefix and convert to integers
df['HospitalId.anon'] = df['HospitalId.anon'].str[4:].astype(int)


reporter.report('Processing S01Sex')
# 1 = male, 2 = female.
# We convert to a {0, 1} encoding
df = remap_categories(df, 'S01Sex', [(1, 0), (2, 1)])


reporter.report('Processing S02PreOpCTPerformed')
# - 1 = yes
# - 0 = no
# - 9 = unknown
# We convert the unknowns to missing values
df.loc[
    df['S02PreOpCTPerformed'] == 9,
    'S02PreOpCTPerformed'
] = np.nan


reporter.first('Processing urea and creatinine')
# Looks like creatinine and urea values are accidentally swapped in some cases
# In other cases, urea and creatinine have exactly the same value
CREATININE_LOWER_THRESHOLD = 8.8
UREA_UPPER_THRESHOLD = 72
JOINT_CREATININE_UREA_REDACTION_RATIO = 2.0
JOINT_CREATININE_UREA_REDACTION_THRESHOLD = 30
CREATININE_UREA_VARIABLE_NAMES = ['S03SerumCreatinine', 'S03Urea']
redact = df[CREATININE_UREA_VARIABLE_NAMES].copy()
for name in CREATININE_UREA_VARIABLE_NAMES:
    reporter.report(
        f'{df.loc[df[name].notnull()].shape[0]} cases have non-null '
        f'{name} prior to processing'
    )


reporter.report(f'Redacting creatinines below {CREATININE_LOWER_THRESHOLD}')
n_before = redact.loc[redact['S03SerumCreatinine'].notnull()].shape[0]
redact.loc[
    redact['S03SerumCreatinine'] < CREATININE_LOWER_THRESHOLD,
    'S03SerumCreatinine'
] = np.nan
n_after = redact.loc[redact['S03SerumCreatinine'].notnull()].shape[0]
reporter.report(f'{n_before - n_after} creatinines removed')


reporter.report(f'Redacting ureas above {UREA_UPPER_THRESHOLD}')
n_before = redact.loc[redact['S03Urea'].notnull()].shape[0]
redact.loc[
    redact['S03Urea'] > UREA_UPPER_THRESHOLD,
    'S03Urea'
] = np.nan
n_after = redact.loc[redact['S03Urea'].notnull()].shape[0]
reporter.report(f'{n_before - n_after} ureas removed')


reporter.report(
    'Redacting both creatinine and urea in cases where they are the same'
)
n_before = redact.loc[redact['S03SerumCreatinine'].notnull()].shape[0]
redact.loc[
    (redact['S03SerumCreatinine'] - redact['S03Urea']) < 0.1,
    CREATININE_UREA_VARIABLE_NAMES
] = np.nan
n_after = redact.loc[redact['S03SerumCreatinine'].notnull()].shape[0]
reporter.report(f'Urea & creatinine removed in {n_before - n_after} cases')


reporter.report(
    'Redacting both creatinine and urea in cases where urea > '
    f'{JOINT_CREATININE_UREA_REDACTION_THRESHOLD}, and '
    f'{JOINT_CREATININE_UREA_REDACTION_RATIO} * urea > creatinine'
)
n_before = redact.loc[redact['S03SerumCreatinine'].notnull()].shape[0]
redact.loc[
    ((
        JOINT_CREATININE_UREA_REDACTION_RATIO * redact['S03Urea'] >
        redact['S03SerumCreatinine']
    ) & (
            redact['S03Urea'] > JOINT_CREATININE_UREA_REDACTION_THRESHOLD
    )),
    CREATININE_UREA_VARIABLE_NAMES
] = np.nan
n_after = redact.loc[redact['S03SerumCreatinine'].notnull()].shape[0]
reporter.report(f'Urea & creatinine removed in {n_before - n_after} cases')


reporter.report('Updating main data with creatinine and urea redactions')
df.loc[:, CREATININE_UREA_VARIABLE_NAMES] = redact.loc[
    :, CREATININE_UREA_VARIABLE_NAMES]


reporter.first('Processing S03PreOpArterialBloodLactate')
drop_values_under_threshold(df, 'S03PreOpArterialBloodLactate', 0.001)


reporter.first('Processing S03PreOpLowestAlbumin')
drop_values_under_threshold(df, 'S03PreOpLowestAlbumin', 0.001)


reporter.first('Processing S03SystolicBloodPressure')
drop_values_under_threshold(df, 'S03SystolicBloodPressure', 60.0)


reporter.first('Processing S03Pulse')
drop_values_under_threshold(df, 'S03Pulse', 30.0)


reporter.first('Processing S03GlasgowComaScore')
df = remove_non_whole_numbers(df, 'S03GlasgowComaScore')


reporter.report('Processing S03WhatIsTheOperativeSeverity')
# - 8 is 'major+'
# - 4 is 'major'
# We convert to a {0, 1} encoding
df = remap_categories(df, 'S03WhatIsTheOperativeSeverity', [(8, 1), (4, 0)])


reporter.report('Processing S03NCEPODUrgency')
# - 1 = 3 expedited (>18 hours)
# - 2 = 2B urgent (6-18 hours)
# - 3 = 2A urgent (2-6 hours)
# - 4 = Emergency, resuscitation of >2 hours possible (no longer available)
# - 8 = 1 immediate (<2 hours)
# We consolidate category 4 to 3
df.loc[df['S03NCEPODUrgency'] == 4., 'S03NCEPODUrgency'] = 3.


reporter.report('Processing indications')
indications = [c for c in df.columns if "S05Ind_" in c]
# - 1 = case has this indication
# - 2 = case doesn't have this indication#
# There are some NaNs. Unclear why this is.
# We convert to a {0, 1} encoding, eliminating NaNs
for c in indications:
    df.loc[df[c] != 1, c] = 0
    df.loc[:, c] = df[c].astype(int).values


reporter.first('Processing S07Status_Disch')
# - 0 - Dead
# - 1 - Alive
# - 60 - still in hospital at 60 days
reporter.report('Summarising mortality in raw data (as frequencies)')
print(df['S07Status_Disch'].value_counts())
reporter.report('Summarising mortality in raw data (as proportions)')
print(df['S07Status_Disch'].value_counts(normalize=True))


reporter.first(
    'Reassigning patients still alive in hospital at 60 days post-op as alive'
)
df = remap_categories(df, 'S07Status_Disch', [
    (60, 1),
    (1, 2),  # make temporary category 2 so that (0, 1) works properly below
    (0, 1),
    (2, 0)  # reassign from temporary category
])
reporter.report('Summarising mortality in raw data (as frequencies)')
print(df['S07Status_Disch'].value_counts())
reporter.report('Summarising mortality in raw data (as proportions)')
print(df['S07Status_Disch'].value_counts(normalize=True))
df['Target'] = df['S07Status_Disch'].copy()
df.drop('S07Status_Disch', axis=1)


reporter.report('Resetting DataFrame index')
df = df.reset_index(drop=True)


reporter.report('Saving wrangled data')
df.to_pickle(os.path.join(
    DATA_DIR,
    "lap_risk_df_after_univariate_wrangling_all_variables.pkl"
))



reporter.first('Dropping variables unused in downstream analysis')
lap_risk_vars = [
    "HospitalId.anon",
    "S01Sex",
    "S03ASAScore",
    "S03NCEPODUrgency",
    "S03ECG",
    "S03NumberOfOperativeProcedures",
    "S03CardiacSigns",
    "S03RespiratorySigns",
    "S03Pred_Peritsoil",
    "S03Pred_TBL",
    "S03DiagnosedMalignancy",
    "S03WhatIsTheOperativeSeverity",
    "S03GlasgowComaScore",
    "S01AgeOnArrival",
    "S03SerumCreatinine",
    "S03Sodium",
    "S03Potassium",
    "S03Urea",
    "S03WhiteCellCount",
    "S03Pulse",
    "S03SystolicBloodPressure",
    "S02PreOpCTPerformed",
    "S03PreOpArterialBloodLactate",
    "S03PreOpLowestAlbumin",
    "Target"
] + indications
df = df[lap_risk_vars]


# TODO: Remove this testing code
comparison = pd.read_pickle(os.path.join(
    os.pardir,
    'nelarisk',
    'data',
    'lap_risk_df_after_univariate_wrangling.pkl'))
assert df.equals(comparison)


# TODO: Uncomment when script rewriting is complete
# reporter.report('Saving wrangled data')
# df.to_pickle(NELA_DATA_FILEPATH)


reporter.last('Done.')
