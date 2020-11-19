import os
import numpy as np
import pandas as pd

from utils.constants import RAW_NELA_DATA_FILEPATH, NELA_DATA_FILEPATH
from utils.wrangling import (
    remap_categories,
    remove_non_whole_numbers,
    drop_values_under_threshold
)
from utils.report import Reporter


reporter = Reporter()
reporter.title("Initial data wrangling")


reporter.report('Loading raw data')
df = pd.read_csv(RAW_NELA_DATA_FILEPATH)
print(f'Dataset contains {df.shape[0]} patients')


reporter.first('Processing S01AgeOnArrival')
reporter.report('Excluding patients < 18 years old')
df = df.loc[df['S01AgeOnArrival'] > 17]
print(df.shape[0], 'patients remain')
reporter.report('Excluding patients >= 110')
df = df.loc[df['S01AgeOnArrival'] < 110]
print(df.shape[0], 'patients remain')


reporter.first('Preprocessing HospitalId.anon')
# Remove 'trust' prefix and convert to integers
df['HospitalId.anon'] = df['HospitalId.anon'].str[4:].astype(int)


reporter.report('Processing S01Sex')
# 1 = male, 2 = female.
# We convert this to binary
df = remap_categories(df, 'S01Sex', [(1, 0), (2, 1)])


reporter.report('Processing S02PreOpCTPerformed')
# - 1 = yes
# - 0 = no
# - 9 = unknown
# We convert the unknowns to missing values
df.loc[df['S02PreOpCTPerformed'] == 9,
       'S02PreOpCTPerformed'] = np.nan


reporter.first('Processing urea and creatinine')
# Looks like some creatinine and urea values are swapped
# **Decision to redact very low creatinine values.**
# Our search reveals assays can't detect below 8.8 (see GitHub reference).
redact = df[['S03SerumCreatinine', 'S03Urea']].copy()
redact.loc[redact['S03SerumCreatinine'] < 8.8,
           'S03SerumCreatinine'] = np.nan
print(df.loc[df['S03SerumCreatinine'].notnull()].shape[0])
print(redact.loc[redact['S03SerumCreatinine'].notnull()].shape[0])
# **Decision to redact very high urea values.**
# Highest we found in a case report was 72.
redact.loc[redact['S03Urea'] > 72,
           'S03Urea'] = np.nan
print(df.loc[df['S03Urea'].notnull()].shape[0])
print(redact.loc[redact['S03Urea'].notnull()].shape[0])
# There also appear to be some patients where Urea and creatinine are exactly
# the same
# **Decision to remove both values in these cases**
redact.loc[(redact['S03SerumCreatinine'] - redact['S03Urea']) < 0.1,
           ['S03SerumCreatinine', 'S03Urea']] = np.nan
print(df.loc[df['S03SerumCreatinine'].notnull()].shape[0])
print(redact.loc[redact['S03SerumCreatinine'].notnull()].shape[0])
# **Decision to apply the rule**: redact any creatinine and urea where
# (a * urea) > creatinine, where urea is over some threshold?
a = 2.0
redact.loc[((a * redact['S03Urea'] > redact['S03SerumCreatinine'])
            & (redact['S03Urea'] > 30)),
           ['S03SerumCreatinine', 'S03Urea']] = np.nan
print(df.loc[df['S03SerumCreatinine'].notnull()].shape[0])
print(redact.loc[redact['S03SerumCreatinine'].notnull()].shape[0])
# update the main data with these changes
df[['S03SerumCreatinine', 'S03Urea']] = redact[
    ['S03SerumCreatinine', 'S03Urea']]


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
# We convert this to a binary variable
df = remap_categories(df, 'S03WhatIsTheOperativeSeverity', [(8, 1), (4, 0)])


reporter.report('Processing S03NCEPODUrgency')
# - 1 = 3 expedited (>18 hours)
# - 2 = 2B urgent (6-18 hours)
# - 3 = 2A urgent (2-6 hours)
# - 4 = Emergency, resuscitation of >2 hours possible (no longer available)
# - 8 = 1 immediate (<2 hours)
df.loc[df['S03NCEPODUrgency'] == 4., 'S03NCEPODUrgency'] = 3.


reporter.report('Processing indications')
indications = [c for c in df.columns if "S05Ind_" in c]
# - 1 = this is the indication
# - 2 = This isn't the indication
# 
# There are some NaNs - unclear why
# Convert indications to binary, eliminating NaNs
for c in indications:
    df.loc[df[c] != 1, c] = 0
    df.loc[:, c] = df[c].astype(int).values


reporter.report('Processing S07Status_Disch')
# - 0 - Dead
# - 1 - Alive
# - 60 - still in hospital at 60 days
# **Decision to combine 1 and 60 for the purposes of mortality prediction,
# accepting that the 60 patients are likely to be systematically different
# for the 1 patients**
df = remap_categories(df, 'S07Status_Disch', [
    (60, 1),
    (1, 2),  # make temporary category 2 so that (0, 1) works properly below
    (0, 1),
    (2, 0)  # reassign from temporary category
])
df['Target'] = df['S07Status_Disch'].copy()
df.drop('S07Status_Disch', axis=1)


reporter.report('Dropping variables unused in downstream analysis')
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


reporter.report('Resetting DataFrame index')
df = df[lap_risk_vars].reset_index(drop=True)


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


reporter.report('Done.')
