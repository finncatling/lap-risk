#!/usr/bin/env python
# coding: utf-8

# # Train-test split (by trusts)

# In[1]:


import os, sys
import pandas as pd
import numpy as np

sys.path.append('')
from nelarisk.constants import CURRENT_NELA_RISK_MODEL_VARS, RANDOM_SEED
from nelarisk.explore import missingness_perc
from nelarisk.helpers import drop_incomplete_cases, save_object


# In[2]:


df = pd.read_pickle(
    os.path.join('data', 'df_after_univariate_wrangling.pkl'))


# ## Split into train and test trusts

# In[3]:


n_trusts = df['TrustId.anon'].nunique()
trust_ids = df['TrustId.anon'].unique()
n_trusts


# In[4]:


test_fraction = 0.2
test_n_trusts = int(np.round(n_trusts * test_fraction))
print(f'The test fold is from {test_n_trusts} trusts')


# In[5]:


rnd = np.random.RandomState(RANDOM_SEED)


# In[6]:


test_trust_ids = np.sort(rnd.choice(trust_ids, test_n_trusts,
                                    replace=False))


# In[7]:


train_trust_ids = np.array(list(set(trust_ids) -
                                set(test_trust_ids)))


# In[8]:


test_trust_ids


# In[9]:


train_trust_ids


# In[10]:


train_i = df.index[
    df['TrustId.anon'].isin(train_trust_ids)].to_numpy()


# ## Select only the features current NELA risk model uses

# In[11]:


nela_vars = (list(CURRENT_NELA_RISK_MODEL_VARS['cat']) +
             list(CURRENT_NELA_RISK_MODEL_VARS['cont']) +
             [CURRENT_NELA_RISK_MODEL_VARS['target']])

df = df[nela_vars + ['TrustId.anon']]


# Check that `TrustId.anon` has no missing values (so shouldn't change the number of complete cases below by being dropped).

# In[12]:


missingness_perc(df, 'TrustId.anon')


# In[13]:


df, total_n, complete_n = drop_incomplete_cases(df)


# ## Select consistent test fold across models, with test fold all from trusts not in train fold

# We will use these as the test fold for all models, i.e. we will not use incomplete cases from the test fold trusts for training or testing (though we may use them for MICE).
# 
# Are these complete cases likely to be systemically different from the incomplete cases?

# In[14]:


test_i = df.index[
    df['TrustId.anon'].isin(test_trust_ids)].to_numpy()

print(f'There are {test_i.shape[0]} cases in the test fold.',
      f'This is {np.round(100 * (test_i.shape[0] / complete_n), 3)}%',
      'of complete cases and',
      f'{np.round(100 * (test_i.shape[0] / total_n), 3)}%',
       'of all cases')


# Save train/test arrays for use elsewhere:

# In[15]:


train_test_split = {'train_trust_ids': train_trust_ids,
                    'test_trust_ids': test_trust_ids,
                    'test_i': test_i,
                    'train_i': train_i}

save_object(train_test_split,
            os.path.join('data', 'train_test_split.pkl'))

