# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd

from utils.constants import RANDOM_SEED, DATA_DIR
from utils.current_nela_model import CURRENT_NELA_MODEL_VARS
from utils.explore import missingness_perc
from utils.io import save_object
from utils.split_data import drop_incomplete_cases
from utils.report import Reporter

reporter = Reporter()

reporter.report('Loading manually-wrangled NELA data')
df = pd.read_pickle(
    os.path.join(DATA_DIR, 'df_after_univariate_wrangling.pkl'))



# Split into train and test trusts


n_trusts = df['TrustId.anon'].nunique()
trust_ids = df['TrustId.anon'].unique()
n_trusts

test_fraction = 0.2
test_n_trusts = int(np.round(n_trusts * test_fraction))
print(f'The test fold is from {test_n_trusts} trusts')

rnd = np.random.RandomState(RANDOM_SEED)

test_trust_ids = np.sort(rnd.choice(trust_ids, test_n_trusts,
                                    replace=False))

train_trust_ids = np.array(list(set(trust_ids) -
                                set(test_trust_ids)))

test_trust_ids

train_trust_ids

train_i = df.index[
    df['TrustId.anon'].isin(train_trust_ids)].to_numpy()

# ## Select only the features current NELA risk model uses


nela_vars = (list(CURRENT_NELA_MODEL_VARS['cat']) +
             list(CURRENT_NELA_MODEL_VARS['cont']) +
             [CURRENT_NELA_MODEL_VARS['target']])

df = df[nela_vars + ['TrustId.anon']]

# Check that `TrustId.anon` has no missing values (so shouldn't change the
# number of complete cases below by being dropped).


missingness_perc(df, 'TrustId.anon')

df, total_n, complete_n = drop_incomplete_cases(df)

# ## Select consistent test fold across models, with test fold all from
# trusts not in train fold

# We will use these as the test fold for all models, i.e. we will not use
# incomplete cases from the test fold trusts for training or testing (though
# we may use them for MICE).
# 
# Are these complete cases likely to be systemically different from the
# incomplete cases?


test_i = df.index[
    df['TrustId.anon'].isin(test_trust_ids)].to_numpy()

print(f'There are {test_i.shape[0]} cases in the test fold.',
      f'This is {np.round(100 * (test_i.shape[0] / complete_n), 3)}%',
      'of complete cases and',
      f'{np.round(100 * (test_i.shape[0] / total_n), 3)}%',
      'of all cases')

# Save train/test arrays for use elsewhere:


train_test_split = {'train_trust_ids': train_trust_ids,
                    'test_trust_ids': test_trust_ids,
                    'test_i': test_i,
                    'train_i': train_i}

save_object(train_test_split,
            os.path.join('data', 'train_test_split.pkl'))
