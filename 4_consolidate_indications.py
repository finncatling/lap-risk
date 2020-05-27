import os
import numpy as np
import pandas as pd
from typing import List

from utils.inspect import report_ohe_category_assignment
from utils.report import Reporter
from utils.io import make_directory, save_object
from utils.constants import (DATA_DIR, STATS_OUTPUT_DIR)
from utils.model.novel import INDICATION_PREFIX, MISSING_IND_CATEGORY

SINGLE_IND_FREQUENCY_THRESHOLD = 1200

reporter = Reporter()
reporter.title('Rationalise indications for surgery (retaining common single '
               'indications and reassigning common combinations of '
               'indications) in preparation for data input to the novel risk '
               'model. See additional data exploration and summary statistics '
               'in the consolidate_indications.ipynb notebook')


reporter.report("Creating output dirs (if they don't already exist)")
make_directory(STATS_OUTPUT_DIR)


reporter.report('Loading manually-wrangled NELA data')
df = pd.read_pickle(
    os.path.join(DATA_DIR, 'df_after_univariate_wrangling.pkl')
).reset_index(drop=True)


reporter.report('Isolating indication variables')
indications = [c for c in df.columns if INDICATION_PREFIX in c]
ind_df = df[indications].copy()


reporter.report("Defining 'common single indications' as those that occur in "
                f'isolation more than {SINGLE_IND_FREQUENCY_THRESHOLD} times')
common_single_inds: List[str] = ind_df.loc[ind_df.sum(1) == 1].sum(0)[
    ind_df.loc[ind_df.sum(1) == 1].sum(0) > SINGLE_IND_FREQUENCY_THRESHOLD
    ].sort_values(ascending=False).index.tolist()


print('The common single indications are',
      ', '.join([i[len(INDICATION_PREFIX):] for i in common_single_inds]))


reporter.first('Making a new one-hot-encoded DataFrame containing just the '
               'common single indications')
new_ind_df = pd.DataFrame(np.zeros((df.shape[0], len(common_single_inds))),
                          columns=common_single_inds)
for csi in common_single_inds:
    new_ind_df.loc[((ind_df.sum(1) == 1) & (ind_df[csi] == 1)), csi] = 1.
report_ohe_category_assignment(new_ind_df, 'indication')


reporter.first('Changing the most-commonly-occurring pairs of '
               'indications to the closest single indication')
for ind_pair, keep_ind_index in (
        (('Perforation', 'Peritonitis'), 0),
        (('SmallBowelObstruction', 'IncarceratedHernia'), 0),
        (('Perforation', 'AbdominalAbscess'), 0)
):
    print(f'Changing {ind_pair} to {ind_pair[keep_ind_index]}')
    new_ind_df.loc[((ind_df.sum(1) == 2) &
                    (ind_df[f'{INDICATION_PREFIX}{ind_pair[0]}'] == 1) &
                    (ind_df[f'{INDICATION_PREFIX}{ind_pair[1]}'] == 1)),
                   f'{INDICATION_PREFIX}{ind_pair[keep_ind_index]}'] = 1
    report_ohe_category_assignment(new_ind_df, 'indication')


reporter.first('Changing the most-commonly-occurring trios of '
               'indications to the closest single indication')
for ind_trio, keep_ind_index in (
        (('Perforation', 'Peritonitis', 'AbdominalAbscess'), 0),
):
    print(f'Changing {ind_trio} to {ind_trio[keep_ind_index]}')
    new_ind_df.loc[((ind_df.sum(1) == 3) &
                    (ind_df[f'{INDICATION_PREFIX}{ind_trio[0]}'] == 1) &
                    (ind_df[f'{INDICATION_PREFIX}{ind_trio[1]}'] == 1) &
                    (ind_df[f'{INDICATION_PREFIX}{ind_trio[2]}'] == 1)),
                   f'{INDICATION_PREFIX}{ind_trio[keep_ind_index]}'] = 1
    report_ohe_category_assignment(new_ind_df, 'indication')


reporter.first("Assigning cases with no indication in the original data to a "
               "new 'indication missing' category")
new_ind_df[MISSING_IND_CATEGORY] = np.zeros(df.shape[0])
new_ind_df.loc[ind_df.sum(1) == 0, MISSING_IND_CATEGORY] = 1.
report_ohe_category_assignment(new_ind_df, 'indication')


reporter.first('Assigning cases with >1 indication, at least one of which is '
               'a common single indication, and where the case was not '
               'reassigned earlier, to the missing category.')
new_ind_df.loc[((ind_df.sum(1) > 1) &
                (ind_df[common_single_inds].sum(1) > 0) &
                (new_ind_df.sum(1) == 0)),
               MISSING_IND_CATEGORY] = 1.
report_ohe_category_assignment(new_ind_df, 'indication')


reporter.first('Assigning remaining cases (those with one or more indications '
               'in the original data, but where none of these indications is a '
               "common single indication) to the 'other indication' "
               'category')
new_ind_df.loc[new_ind_df.sum(1) == 0, f'{INDICATION_PREFIX}Other'] = 1.
report_ohe_category_assignment(new_ind_df, 'indication')


reporter.first('Confirming each case now has exactly one assigned indication')
assert new_ind_df.loc[new_ind_df.sum(1) == 1].shape[0] == new_ind_df.shape[0]


reporter.report('Adding new indications encoding to existing data.')
df = df.drop(indications, axis=1)
df_with_new_inds = pd.concat((df, new_ind_df), axis=1)


reporter.report('Saving data for later use')
df_with_new_inds.to_pickle(os.path.join(
    DATA_DIR, 'df_after_univariate_wrangling_new_indications.pkl'))


reporter.report('Saving a few summary statistics')
ci_stats = {'indication_counts': new_ind_df.sum(0).sort_values(ascending=False)}
ci_stats['indication_proportions'] = (ci_stats['indication_counts'] /
                                      new_ind_df.shape[0])
save_object(ci_stats,
            os.path.join(STATS_OUTPUT_DIR, '4_consolidate_indications.pkl'))


reporter.last('Done.')
