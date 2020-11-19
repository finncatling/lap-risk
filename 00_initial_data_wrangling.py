import os
import numpy as np
import pandas as pd


def binarize(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Display some useful attributes of a categorical feature. Note
        that binary=True alters the underlying input DataFrame."""
    df.loc[df[col_name] == 1, col_name] = 0
    df.loc[df[col_name] == 2, col_name] = 1
    return df


def remove_non_whole_numbers(df, var_name):
    """Removes non-whole-number floats. Preserves other missing
        values."""
    unrounded = df.loc[df[var_name].notnull(), var_name]
    rounded = unrounded.round()
    diff = rounded != unrounded
    diff_i = diff[diff == True].index
    df.loc[diff_i, var_name] = np.nan
    return df


def sev_to_binary(x):
    if x == 8:
        return 1
    elif x == 4:
        return 0


def combine60(x):
    if x == 60:
        # if status is still alive at 60 days
        x = 1

    # flip values so they make more sense
    if x == 1:
        x = 0
    elif x == 0:
        x = 1
    return (int(x))


data_path = os.path.join(os.pardir,
                         os.pardir,
                         'extract',
                         'datadownload_20190524',
                         'hqip254NELAdata21May2019.csv')
df = pd.read_csv(data_path)

num_rows = df.shape[0]
print(f'Dataset contains {num_rows} patients')

comparison = pd.read_pickle(os.path.join(
    os.pardir,
    'nelarisk',
    'data',
    'lap_risk_df_after_univariate_wrangling.pkl'))

print('comparison shape:', comparison.shape)

# ## TrustId.anon
# Remove 'trust' prefix and convert to integers
df['TrustId.anon'] = df['TrustId.anon'].str[5:].astype(int)

# ## HospitalId.anon
# Remove 'trust' prefix and convert to integers
df['HospitalId.anon'] = df['HospitalId.anon'].str[4:].astype(int)

# ## S01AgeOnArrival
# - **Decision to exclude the <18yo patients**
# - **Decision to exclude these 3 oldest patients as very likely erroneous**
print(df.shape)
df = df.loc[df['S01AgeOnArrival'] < 110]
print(df.shape)
df = df.loc[df['S01AgeOnArrival'] > 17]
print(df.shape)

# ## S01Sex
# 1 = male, 2 = female
binarize(df, 'S01Sex')


# ## 'S02PreOpCTPerformed'
# 
# - 1 = yes
# - 0 = no
# - 9 = unknown
# **Decision to convert the unknowns to NaNs**
df.loc[df['S02PreOpCTPerformed'] == 9,
       'S02PreOpCTPerformed'] = np.nan

## Have people mixed up the creatinine and urea fields?
# **Decision to redact very low creatinine values.** Our search reveals assays can't detect below 8.8 (see GitHub reference).
redact = df[['S03SerumCreatinine', 'S03Urea']].copy()
redact.loc[redact['S03SerumCreatinine'] < 8.8,
           'S03SerumCreatinine'] = np.nan
print(df.loc[df['S03SerumCreatinine'].notnull()].shape[0])
print(redact.loc[redact['S03SerumCreatinine'].notnull()].shape[0])
# **Decision to redact very high urea values.** Highest we found in a case report was 72.
redact.loc[redact['S03Urea'] > 72,
           'S03Urea'] = np.nan
print(df.loc[df['S03Urea'].notnull()].shape[0])
print(redact.loc[redact['S03Urea'].notnull()].shape[0])
# There also appear to be some patients where Urea and creatinine are exactly the same (the straight line above). **Decision to remove both values in these cases**
redact.loc[(redact['S03SerumCreatinine'] - redact['S03Urea']) < 0.1,
           ['S03SerumCreatinine', 'S03Urea']] = np.nan
print(df.loc[df['S03SerumCreatinine'].notnull()].shape[0])
print(redact.loc[redact['S03SerumCreatinine'].notnull()].shape[0])
# **Decision to apply the rule**: redact any creatinine and urea where ($a$ * urea) > creatinine, where urea is over some threshold?
a = 2.0
redact.loc[((a * redact['S03Urea'] > redact['S03SerumCreatinine'])
            & (redact['S03Urea'] > 30)),
           ['S03SerumCreatinine', 'S03Urea']] = np.nan
print(df.loc[df['S03SerumCreatinine'].notnull()].shape[0])
print(redact.loc[redact['S03SerumCreatinine'].notnull()].shape[0])
# update the main DataFrame with these changes
df[['S03SerumCreatinine', 'S03Urea']] = redact[
    ['S03SerumCreatinine', 'S03Urea']]

# ## 'S03PreOpArterialBloodLactate'
# **decision to remove values which are zero**.
print(df[df['S03PreOpArterialBloodLactate'].notnull()].shape[0])
df.loc[df['S03PreOpArterialBloodLactate'] < 0.001,
       'S03PreOpArterialBloodLactate'] = np.nan
print(df[df['S03PreOpArterialBloodLactate'].notnull()].shape[0])

# ## 'S03PreOpLowestAlbumin'
# **decision to remove values which are zero**.
print(df[df['S03PreOpLowestAlbumin'].notnull()].shape[0])
df.loc[df['S03PreOpLowestAlbumin'] < 0.001,
       'S03PreOpLowestAlbumin'] = np.nan
print(df[df['S03PreOpLowestAlbumin'].notnull()].shape[0])

# **Decision to redact very low systolic BP and HR values:**
for v, lower in [('S03SystolicBloodPressure', 60.0),
                 ('S03Pulse', 30.0), ]:
    print(v)
    print(df.loc[df[v].notnull()].shape[0])
    df.loc[df[v] < lower, v] = np.nan
    print(df.loc[df[v].notnull()].shape[0], '\n')

# ## 'S03GlasgowComaScore'
df = remove_non_whole_numbers(df, 'S03GlasgowComaScore')

## ## S03WhatIsTheOperativeSeverity
# - 8 is 'major+'
# - 4 is 'major' 
df.loc[:, 'S03WhatIsTheOperativeSeverity'] = df[
    'S03WhatIsTheOperativeSeverity'].apply(sev_to_binary)

# ## 'S03NCEPODUrgency'
# - 1 = 3 expedited (>18 hours)
# - 2 = 2B urgent (6-18 hours)
# - 3 = 2A urgent (2-6 hours)
# - 4 = Emergency, resuscitation of >2 hours possible (no longer available)
# - 8 = 1 immediate (<2 hours)
df.loc[df['S03NCEPODUrgency'] == 4., 'S03NCEPODUrgency'] = 3.

# ## S05Ind...
indications = [c for c in df.columns if "S05Ind_" in c]
# - 1 = this is the indication
# - 2 = This isn't the indication
# 
# There are some NaNs - why?
# Convert indications to binary, eliminating NaNs
for c in indications:
    df.loc[df[c] != 1, c] = 0
    df.loc[:, c] = df[c].astype(int).values

# ## 'S07Status_Disch'
# - 0 - Dead
# - 1 - Alive
# - 60 - still in hospital at 60 days
# **Decision to combine 1 and 60 for the purposes of mortality prediction, accepting that the 60 patients are likely to be systematically different for the 1 patients**
df['Target'] = df['S07Status_Disch'].apply(combine60)

# ## Export only those variables used in downstream preoperative mortality modelling by `lap-risk`
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

df = df[lap_risk_vars].reset_index(drop=True)

assert df.equals(comparison)


# save wrangled data
# df[lap_risk_vars + indications].reset_index(drop=True).to_pickle(
#    os.path.join('data',
#                 'lap_risk_df_after_univariate_wrangling.pkl'))
