#!/usr/bin/env python
# coding: utf-8

# # Reimplementation of current NELA risk model


import os, sys, copy
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from utils.current_nela_model import preprocess_df

sys.path.append('')
from nelarisk.constants import CURRENT_NELA_RISK_MODEL_VARS, RANDOM_SEED
from nelarisk.helpers import drop_incomplete_cases, split_into_folds
from nelarisk.evaluate import evaluate_predictions

get_ipython().run_line_magic('matplotlib', 'inline')



df = pd.read_pickle(
    os.path.join('data', 'df_after_univariate_wrangling.pkl'))


# ## Select only the features current NELA risk model uses

# NB. Web form for NELA risk model includes Hb, but only uses this to calculate P-POSSUM.


nela_vars = (list(CURRENT_NELA_RISK_MODEL_VARS['cat']) +
             list(CURRENT_NELA_RISK_MODEL_VARS['cont']) +
             [CURRENT_NELA_RISK_MODEL_VARS['target']])

df = df[nela_vars]

df.columns.tolist()


# ## Drop incomplete cases (as per their paper)


df, _ = drop_incomplete_cases(df)


# ## Split data into train and test folds


X_train_df, y_train, X_test_df, y_test = split_into_folds(df)


# # Perform transformations

# ## Categorical variables


cat_vars = list(CURRENT_NELA_RISK_MODEL_VARS['cat'])

# ## Continuous variables


cont_log_vars = list(copy.deepcopy(CURRENT_NELA_RISK_MODEL_VARS['cont']))
for original, logged in (('S03SerumCreatinine', 'logcreat'),
                         ('S03Urea', 'logurea')):
    cont_log_vars.remove(original)
    cont_log_vars.append(logged)

winsor_thresholds = {
    # age seems to be excluded from winsorization
    'logcreat': (3.3, 6.0),
    'S03SystolicBloodPressure': (70.0, 190.0),
    'S03Pulse': (55.0, 145.0),
    'S03WhiteCellCount': (1.0, 42.7),
    'logurea': (0.0, 3.7),
    'S03Potassium': (2.8, 5.9),
    'S03Sodium': (124.0, 148.0)
}

centres = {
    'S01AgeOnArrival': -64.0,
    'logcreat': -4.0,
    'S03SystolicBloodPressure': -127.0,
    'S03Pulse': -91.0,
    'S03WhiteCellCount': -13.0,
    'logurea': -1.9,
    'S03Potassium': -4.0,
    'S03Sodium': -123.0
}

# ## Run preprocessing


X_train_df, lb = preprocess_df(X_train_df, label_binarizers=None)
X_test_df, _ = preprocess_df(X_test_df, lb)

X_train_df.columns.tolist()


# ## Fit and evaluate model

# Below, we set $C$ (the inverse of regularisation strength) to a very large number in order to avoid any meaningful regularisation.


rnd = np.random.RandomState(RANDOM_SEED)



lr = LogisticRegression(C=10 ** 50, solver='liblinear',
                        random_state=rnd)
lr.fit(X_train_df.values, y_train)



y_pred = lr.predict_proba(X_test_df.values)[:, 1]



evaluate_predictions(y_test, y_pred)

