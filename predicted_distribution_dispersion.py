#!/usr/bin/env python
# coding: utf-8

# # Calculate dispersion of predicted risk distributions

# In[18]:


import os, sys
sys.path.append(os.pardir)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from arviz.stats.density_utils import kde, _kde_linear

from utils.io import load_object
from utils.model.novel import NovelModel
from utils.constants import NOVEL_MODEL_OUTPUT_DIR
from utils.evaluate import stratify_y_pred
from utils.gam import quick_sample
from utils.filter import get_indices_of_case_imputed_using_target

pd.set_option('display.max_columns', 50)


# In[2]:


novel_model: NovelModel = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "08_novel_model.pkl"))


# In[3]:


y_true, y_pred = novel_model.get_observed_and_predicted('test', 0, 5)


# In[12]:


y_pred_percentiles = np.percentile(y_pred, (2.5, 97.5), axis=0)


# In[14]:


y_pred_95ci = y_pred_percentiles[1, :] - y_pred_percentiles[0, :]


# In[34]:


mice_cat_imputed_case_i = get_indices_of_case_imputed_using_target(
    'test',
    novel_model.cat_imputer.swm,
    novel_model.cat_imputer
)


# In[40]:


y_i = set(range(y_true.size))
len(y_i)


# In[41]:


non_mice_cat_imputed_case_i = y_i - mice_cat_imputed_case_i[0]
len(non_mice_cat_imputed_case_i)


# In[47]:


lac_imputed_case_i = set(
    novel_model.lac_imputer.missing_i['test'][0]['S03PreOpArterialBloodLactate'])
len(lac_imputed_case_i)


# In[49]:


non_lac_imputed_case_i = y_i - lac_imputed_case_i
len(non_lac_imputed_case_i)


# In[51]:


alb_imputed_case_i = set(
    novel_model.alb_imputer.missing_i['test'][0]['S03PreOpLowestAlbumin'])
len(alb_imputed_case_i)


# In[52]:


non_alb_imputed_case_i = y_i - alb_imputed_case_i
len(non_alb_imputed_case_i)


# In[53]:


no_imputation_case_i = (
    non_mice_cat_imputed_case_i - lac_imputed_case_i
) - alb_imputed_case_i
len(no_imputation_case_i)


# In[54]:


just_alb_imputation_case_i = (
    alb_imputed_case_i - lac_imputed_case_i
) - mice_cat_imputed_case_i[0]
len(just_alb_imputation_case_i)


# In[55]:


just_lac_imputation_case_i = (
    lac_imputed_case_i - alb_imputed_case_i
) - mice_cat_imputed_case_i[0]
len(just_lac_imputation_case_i)


# In[56]:


just_lac_alb_imputation_case_i = (
    lac_imputed_case_i.intersection(alb_imputed_case_i)
) - mice_cat_imputed_case_i[0]
len(just_lac_alb_imputation_case_i)


# In[72]:


just_mice_cat_imputation_case_i = (
    mice_cat_imputed_case_i[0] - alb_imputed_case_i
) - lac_imputed_case_i
len(just_mice_cat_imputation_case_i)


# In[59]:


def y_pred_95ci_subset(indices, y_pred_95ci=y_pred_95ci):
    mask = np.in1d(range(y_pred_95ci.size), list(indices))
    return y_pred_95ci[mask]


# In[74]:


for x in (
    y_pred_95ci_subset(no_imputation_case_i),
    y_pred_95ci_subset(just_mice_cat_imputation_case_i),
    y_pred_95ci_subset(just_alb_imputation_case_i),
    y_pred_95ci_subset(just_lac_imputation_case_i),
    y_pred_95ci_subset(just_lac_alb_imputation_case_i)
):
    print(np.round(np.percentile(x, (2.5, 97.5)), 3))


# In[75]:


hist_args = {'bins': 50, 'density': True, 'alpha': 0.5}
plt.hist(y_pred_95ci_subset(no_imputation_case_i),
         **hist_args,
         label='No imputation')
plt.hist(y_pred_95ci_subset(just_mice_cat_imputation_case_i),
         **hist_args,
         label='Just categorical imputation')

plt.xlabel('Predicted mortality risk (2.5- to 97.5-percentile range)')
plt.xlim(0, 0.5)
plt.legend()
plt.show()


# In[65]:


hist_args = {'bins': 50, 'density': True, 'alpha': 0.5}
plt.hist(y_pred_95ci_subset(no_imputation_case_i),
         **hist_args,
         label='No imputation')
plt.hist(y_pred_95ci_subset(just_lac_imputation_case_i),
         **hist_args,
         label='Just lactate imputation')

plt.xlabel('Predicted mortality risk (2.5- to 97.5-percentile range)')
plt.xlim(0, 0.5)
plt.legend()
plt.show()


# In[66]:


hist_args = {'bins': 50, 'density': True, 'alpha': 0.5}
plt.hist(y_pred_95ci_subset(no_imputation_case_i),
         **hist_args,
         label='No imputation')
plt.hist(y_pred_95ci_subset(just_alb_imputation_case_i),
         **hist_args,
         label='Just albumin imputation')

plt.xlabel('Predicted mortality risk (2.5- to 97.5-percentile range)')
plt.xlim(0, 0.5)
plt.legend()
plt.show()


# In[69]:


hist_args = {'bins': 50, 'density': True, 'alpha': 0.5}
plt.hist(y_pred_95ci_subset(no_imputation_case_i),
         **hist_args,
         label='No imputation')
plt.hist(y_pred_95ci_subset(just_lac_alb_imputation_case_i),
         **hist_args,
         label='Just lactate & albumin imputation')

plt.xlabel('Predicted mortality risk (2.5- to 97.5-percentile range)')
plt.xlim(0, 0.5)
plt.legend()
plt.show()

