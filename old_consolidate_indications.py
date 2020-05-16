#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding: utf-8

# # Consolidate pre-operative indications for surgery

# In[1]:


import os, copy, sys
import pandas as pd
import numpy as np
from typing import List

sys.path.append('')
from nelarisk.explore import dot_chart, multi_dot_chart

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_pickle(
    os.path.join('data', 'df_after_univariate_wrangling.pkl'))


# In[3]:


indications = [c for c in df.columns if 'S05Ind_' in c]
indications


# In[4]:


inds_df = df[indications].copy()


# Which indications occur on their own?

# In[5]:


inds_df.loc[inds_df.sum(1) == 1].sum(0).sort_values(ascending=False)


# Too many indications to choose from / having to choose multiple indications will likely harm user experience. Proposal to include indications which occur on their own greater than a certain number of times in the data, and group all other cases together under 'other'.

# In[6]:


threshold = 1300

top_inds = inds_df.loc[inds_df.sum(1) == 1].sum(0)[
    inds_df.loc[inds_df.sum(1) == 1].sum(0) > threshold
].sort_values(ascending=False).index.tolist()

top_inds


# But, some of these terms will co-occur. It is unclear which indication we should prioritise in these cases. Let's examine the frequency of the co-occurrence.
# 
# We limit ourselves to considering co-occurrence in cases with exactly 2 indications. Cases with <= 2 indications represent 90% of the dataset, and if we consider cases with more than 2 indications we will have to factor in pairs co-occurring within trios, etc.

# In[7]:


""" Only pairs occurring more than theshold times in the dataset
    are considered""" 

threshold = 500

ind2_counts = {}

top_inds_no_other = copy.deepcopy(top_inds)
top_inds_no_other.remove('S05Ind_Other')
top_inds_no_other2 = copy.deepcopy(top_inds_no_other)

for c in top_inds_no_other:
    top_inds_no_other2.remove(c)
    for c2 in top_inds_no_other2:
        ind2_counts[(c, c2)] = inds_df.loc[
            ((inds_df.sum(1) == 2) &
             (inds_df[c] == 1) &
             (inds_df[c2] == 1))].shape[0]

ranked_ind_pairs = {k: v for k, v in sorted(ind2_counts.items(),
                                            key=lambda item: item[1],
                                            reverse=True)
                    if v > threshold}
ranked_ind_pairs


# ## Make new one-hot-encoded indications DataFrame using above logic

# In[8]:


new_ind_df = pd.DataFrame(np.zeros((df.shape[0], len(top_inds))),
                          columns=top_inds)

for k in ranked_ind_pairs.keys():
    new_ind_df['+'.join(list(k))] = np.zeros(df.shape[0])


# In[9]:


for ti in top_inds:
    new_ind_df.loc[((inds_df.sum(1) == 1) &
                    (inds_df[ti] == 1)), ti] = 1.


# In[10]:


for ind_pair in ranked_ind_pairs.keys():
    new_ind_df.loc[((inds_df.sum(1) == 2) &
                    (inds_df[ind_pair[0]] == 1) &
                    (inds_df[ind_pair[1]] == 1)),
                   '+'.join(list(ind_pair))] = 1.


# We treat laparotomies with 0 indications as missing and later impute using MICE.
# 
# For laparotomies with > 2 indications:
# 
# - If at least one of the indication is in `top_inds`, we treat the indications as missing and later impute using MICE
# - If none of the indications are in `top_inds`, we reassign the indication as 'S05Ind_Other'
# 
# TODO: sanity checks on the imputed indications.

# In[11]:


new_ind_df['S05Ind_Missing'] = np.zeros(df.shape[0])

new_ind_df.loc[inds_df.sum(1) == 0, 'S05Ind_Missing'] = 1.

new_ind_df.loc[((inds_df.sum(1) > 2) &
                (inds_df[top_inds].sum(1) > 0)),
               'S05Ind_Missing'] = 1.

new_ind_df.loc[new_ind_df.sum(1) == 0, 'S05Ind_Other'] = 1.


# In[12]:


df = df.drop(indications, axis=1)
df_new_inds = pd.concat((df, new_ind_df), axis=1)


df_new_inds.to_pickle(os.path.join(
    'data', 'df_after_univariate_wrangling_new_indications.pkl'))

