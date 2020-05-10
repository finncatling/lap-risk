#!/usr/bin/env python
# coding: utf-8

# # Reimplementation of current NELA risk model

# In[1]:


import os, sys, copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression

sys.path.append('')
from nelarisk.constants import CURRENT_NELA_RISK_MODEL_VARS, RANDOM_SEED
from nelarisk.helpers import drop_incomplete_cases, split_into_folds
from nelarisk.evaluate import evaluate_predictions

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_pickle(
    os.path.join('data', 'df_after_univariate_wrangling.pkl'))


# ## Select only the features current NELA risk model uses

# NB. Web form for NELA risk model includes Hb, but only uses this to calculate P-POSSUM.

# In[3]:


nela_vars = (list(CURRENT_NELA_RISK_MODEL_VARS['cat']) +
             list(CURRENT_NELA_RISK_MODEL_VARS['cont']) +
             [CURRENT_NELA_RISK_MODEL_VARS['target']])

df = df[nela_vars]

df.columns.tolist()


# ## Drop incomplete cases (as per their paper)

# In[4]:


df, _, _ = drop_incomplete_cases(df)


# ## Split data into train and test folds

# In[5]:


X_train_df, y_train, X_test_df, y_test = split_into_folds(df)


# # Perform transformations

# ## Categorical variables

# In[6]:


cat_vars = list(CURRENT_NELA_RISK_MODEL_VARS['cat'])


# In[7]:


def discretise_gcs(df):
    """Discretise GCS. 13-15 category is eliminated to avoid dummy
        variable effect:"""
    for gcs_threshold in ((3, 9), (9, 13)):
        c = 'gcs_{}_{}'.format(gcs_threshold[0], gcs_threshold[1])

        df[c] = np.zeros(df.shape[0])

        df.loc[(df['S03GlasgowComaScore'] >= gcs_threshold[0]) &
               (df['S03GlasgowComaScore'] < gcs_threshold[1]), c] = 1
    df = df.drop('S03GlasgowComaScore', axis=1)
    return df


# In[8]:


def combine_ncepod_urgencies(df):
    """Convert NCEPOD urgency 4 (Emergency but resuscitation of >2h
        possible) to category 3 (Urgent 2-6h) as the NELA model doesn't
        include category 4:"""
    df.loc[df['S03NCEPODUrgency'] == 4., 'S03NCEPODUrgency'] = 3.
    return df


# In[9]:


def combine_highest_resp_levels(df):
    """Combine 2 highest levels of respiratory disease"""
    df.loc[df['S03RespiratorySigns'] == 8., 'S03RespiratorySigns'] = 4.
    return df


# In[10]:


def binarize_categorical(df, label_binarizers):
    """Binarize categorical features. Can use pre-fit label binarizers
        if these are available (e.g. ones fit on the train fold)."""
    if label_binarizers:
        lb = label_binarizers
    else:
        lb = {}

    for v in cat_vars:
        if v != 'S03GlasgowComaScore':
            if label_binarizers:
                trans = lb[v].transform(df[v].astype(int).values)
            else:
                lb[v] = LabelBinarizer()
                trans = lb[v].fit_transform(df[v].astype(int).values)

            if len(lb[v].classes_) == 2:
                df[v] = trans
            else:
                cat_names = ['{}_{}'.format(v, c) for c in lb[v].classes_]
                v_df = pd.DataFrame(trans, columns=cat_names)
                df = pd.concat([df, v_df], axis=1)
                if v != 'S03ASAScore':
                    df = df.drop(v, axis=1)
    
    return df, label_binarizers


# In[11]:


def drop_base_categories(df):
    """For each categorical variable with k categories, we need to
        end up with k-1 variables to avoid the dummy variable trap.
        So, we drop the category that corresponds to baseline risk in
        for each variable."""
    return df.drop(['S03RespiratorySigns_1',
                    'S03NCEPODUrgency_1',
                    'S03ECG_1',
                    'S03NumberOfOperativeProcedures_1',
                    'S03CardiacSigns_1',
                    'S03Pred_Peritsoil_1',
                    'S03Pred_TBL_1',
                    'S03DiagnosedMalignancy_1',
                    'S03ASAScore_1',
                    'S03ASAScore_2'], axis=1)


# In[12]:


def preprocess_categorical(df, label_binarizers):
    """Preprocess categorical variables in NELA data."""
    df = discretise_gcs(df)
    df = combine_ncepod_urgencies(df)
    df = combine_highest_resp_levels(df)
    df, label_binarizers = binarize_categorical(df, label_binarizers)
    df = drop_base_categories(df)
    return df, label_binarizers


# ## Continuous variables

# In[13]:


cont_log_vars = list(copy.deepcopy(CURRENT_NELA_RISK_MODEL_VARS['cont']))
for original, logged in (('S03SerumCreatinine', 'logcreat'),
                         ('S03Urea', 'logurea')):
    cont_log_vars.remove(original)
    cont_log_vars.append(logged)


# In[14]:


def log_urea_creat(df):
    """Log transform urea and creatinine."""
    df['logcreat'] = np.log(df['S03SerumCreatinine'])
    df['logurea'] = np.log(df['S03Urea'])
    return df.drop(['S03SerumCreatinine', 'S03Urea'], axis=1)


# In[15]:


winsor_thresholds = {
    # age seems to be excluded from winsorization
    'logcreat': (3.3, 6.0),
    'S03SystolicBloodPressure': (70, 190),
    'S03Pulse': (55, 145),
    'S03WhiteCellCount': (1.0, 42.7),
    'logurea': (0.0, 3.7),
    'S03Potassium': (2.8, 5.9),
    'S03Sodium': (124, 148)
}


# In[16]:


def winsorize(df):
    """Winsorize continuous variables at thresholds specified in
        NELA paper appendix."""
    for v, threshold in winsor_thresholds.items():
        df.loc[df[v] < threshold[0], v] = threshold[0]
        df.loc[df[v] > threshold[1], v] = threshold[1]
    return df


# In[17]:


centres = {
    'S01AgeOnArrival': -64,
    'logcreat': -4,
    'S03SystolicBloodPressure': -127,
    'S03Pulse': -91,
    'S03WhiteCellCount': -13,
    'logurea': -1.9,
    'S03Potassium': -4,
    'S03Sodium': -123
}


# In[18]:


def centre(df):
    """Centre continuous variables at centres specified in
        NELA paper appendix."""
    for v, c in centres.items():
        df.loc[:, v] += c
    return df


# In[19]:


def add_quadratic_features(df):
    """Add quadratic feature for all continuous features."""
    for v in cont_log_vars:
        df[f'{v}_2'] = df[v] ** 2
    return df


# In[20]:


def add_asa_age_resp_interaction(df):
    """Make separate age, age^2 and respiratory status features
        for levels of ASA. ASA 1 and 2 are grouped together."""
    for f in (('age_asa12', 'S01AgeOnArrival'),
              ('age_2_asa12', 'S01AgeOnArrival_2'),
              ('resp2_asa12', 'S03RespiratorySigns_2'),
              ('resp4_asa12', 'S03RespiratorySigns_4')):

        df[f[0]] = np.zeros(df.shape[0])

        df.loc[(df['S03ASAScore'] == 1) |
               (df['S03ASAScore'] == 2), f[0]] = df.loc[
            (df['S03ASAScore'] == 1) |
            (df['S03ASAScore'] == 2), f[1]]


    for asa in range(3, 6):
        for f in (('age_asa{}'.format(asa), 'S01AgeOnArrival'),
                  ('age_2_asa{}'.format(asa), 'S01AgeOnArrival_2'),
                  ('resp2_asa{}'.format(asa), 'S03RespiratorySigns_2'),
                  ('resp4_asa{}'.format(asa), 'S03RespiratorySigns_4')):

            df[f[0]] = np.zeros(df.shape[0])

            df.loc[df['S03ASAScore'] == asa, f[0]] = df.loc[
                df['S03ASAScore'] == asa, f[1]]
    
    return df.drop(['S01AgeOnArrival', 'S01AgeOnArrival_2',
                    'S03RespiratorySigns_2', 'S03RespiratorySigns_4',
                    'S03ASAScore'], axis=1)


# In[21]:


def transform_sodium(df):
    """Transform sodium as described in NELA paper."""
    df['S03Sodium_3'] = df['S03Sodium'] ** 3
    df['S03Sodium_3_log'] = df['S03Sodium_3'] * np.log(df['S03Sodium'])
    return df.drop('S03Sodium', axis=1)


# In[22]:


def preprocess_continuous(df):
    """Preprocess continuous variables in NELA data."""
    df = log_urea_creat(df)
    df = winsorize(df)
    df = centre(df)
    df = add_quadratic_features(df)
    df = add_asa_age_resp_interaction(df)
    df = transform_sodium(df)
    return df


# In[23]:


def preprocess_df(data, label_binarizers=None):
    """Preprocess NELA data."""
    df = data.copy()
    df = df.reset_index(drop=True)
    df, label_binarizers = preprocess_categorical(df, label_binarizers)
    df = preprocess_continuous(df)
    return df, label_binarizers


# ## Run preprocessing

# In[24]:


X_train_df, lb = preprocess_df(X_train_df, label_binarizers=None)
X_test_df, _ = preprocess_df(X_test_df, lb)

X_train_df.columns.tolist()


# ## Fit and evaluate model

# Below, we set $C$ (the inverse of regularisation strength) to a very large number in order to avoid any meaningful regularisation. 

# In[25]:


rnd = np.random.RandomState(RANDOM_SEED)


# In[26]:


lr = LogisticRegression(C=10 ** 50, solver='liblinear',
                        random_state=rnd)
lr.fit(X_train_df.values, y_train)


# In[27]:


y_pred = lr.predict_proba(X_test_df.values)[:, 1]


# In[28]:


evaluate_predictions(y_test, y_pred)

