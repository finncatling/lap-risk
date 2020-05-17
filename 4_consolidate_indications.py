import os
import numpy as np
import pandas as pd
from typing import List

from utils.inspect import report_ohe_category_assignment
from utils.report import Reporter
from utils.io import make_directory, save_object
from utils.constants import DATA_DIR, STATS_OUTPUT_DIR

reporter = Reporter()
reporter.title('Rationalise indications for surgery (retaining common single '
               'indications and reassigning common combinations of indications '
               'in preparation for data input to the novel risk model. See '
               'relevant data exploration and summary statistics in '
               'consolidate_indications.ipynb notebook (as well as in initial '
               'manual data wrangling notebook)')


reporter.report("Creating output dirs (if they don't already exist)")
make_directory(STATS_OUTPUT_DIR)

reporter.report('Loading manually-wrangled NELA data')
df = pd.read_pickle(
    os.path.join(DATA_DIR, 'df_after_univariate_wrangling.pkl')
).reset_index(drop=True)


reporter.report('Isolating indication variables')
IND_PREFIX = 'S05Ind_'
indications = [c for c in df.columns if IND_PREFIX in c]
ind_df = df[indications].copy()


reporter.report("Setting frequency threshold to define list of 'common "
                "single indications' (i.e. these occur in isolation)")
SINGLE_IND_FREQUENCY_THRESHOLD = 1200
common_single_inds: List[str] = ind_df.loc[ind_df.sum(1) == 1].sum(0)[
    ind_df.loc[ind_df.sum(1) == 1].sum(0) > SINGLE_IND_FREQUENCY_THRESHOLD
    ].sort_values(ascending=False).index.tolist()


print('The common single indications are',
      [i[len(IND_PREFIX):] for i in common_single_inds])


reporter.report('Making a new one-hot-encoded DataFrame containing just the '
                'common single indications')
new_ind_df = pd.DataFrame(np.zeros((df.shape[0], len(common_single_inds))),
                          columns=common_single_inds)
for csi in common_single_inds:
    new_ind_df.loc[((ind_df.sum(1) == 1) & (ind_df[csi] == 1)), csi] = 1.
report_ohe_category_assignment(new_ind_df, 'indication')


reporter.first('Changing the cases with the most-commonly-occurring pairs of '
               'indications so that they have the closest single indication')
for ind_pair, keep_ind_index in (
        (('Perforation', 'Peritonitis'), 0),
        (('SmallBowelObstruction', 'IncarceratedHernia'), 0),
        (('Perforation', 'AbdominalAbscess'), 0)
):
    print(f'Changing {ind_pair} to {ind_pair[keep_ind_index]}')
    new_ind_df.loc[((ind_df.sum(1) == 2) &
                    (ind_df[f'{IND_PREFIX}{ind_pair[0]}'] == 1) &
                    (ind_df[f'{IND_PREFIX}{ind_pair[1]}'] == 1)),
                   f'{IND_PREFIX}{ind_pair[keep_ind_index]}'] = 1
    report_ohe_category_assignment(new_ind_df, 'indication')


reporter.first('Changing the cases with the most-commonly-occurring trios of '
               'indications so that they have the closest single indication')
for ind_trio, keep_ind_index in (
        (('Perforation', 'Peritonitis', 'AbdominalAbscess'), 0),
):
    print(f'Changing {ind_trio} to {ind_trio[keep_ind_index]}')
    new_ind_df.loc[((ind_df.sum(1) == 3) &
                    (ind_df[f'{IND_PREFIX}{ind_trio[0]}'] == 1) &
                    (ind_df[f'{IND_PREFIX}{ind_trio[1]}'] == 1) &
                    (ind_df[f'{IND_PREFIX}{ind_trio[2]}'] == 1)),
                   f'{IND_PREFIX}{ind_trio[keep_ind_index]}'] = 1
    report_ohe_category_assignment(new_ind_df, 'indication')


reporter.first("Assigning the following cases to a new 'indication missing' "
               'category: 1) cases missing an indication in the original data, '
               '2) cases not reassigned above, where at least one of their '
               'indications is a common single indications. These missing '
               'indications will be multiply imputed later')
MISSING_IND_CATEGORY = f'{IND_PREFIX}Missing'
new_ind_df[MISSING_IND_CATEGORY] = np.zeros(df.shape[0])
new_ind_df.loc[ind_df.sum(1) == 0, MISSING_IND_CATEGORY] = 1.
new_ind_df.loc[((ind_df.sum(1) > 2) & (ind_df[common_single_inds].sum(1) > 0)),
               MISSING_IND_CATEGORY] = 1.
report_ohe_category_assignment(new_ind_df, 'indication')


reporter.first('Assigning remaining cases (those with one or more indications '
               'in the original data, but where none of these indications is a '
               "common single indication) to the 'other indication' "
               'category')
new_ind_df.loc[new_ind_df.sum(1) == 0, f'{IND_PREFIX}Other'] = 1.
report_ohe_category_assignment(new_ind_df, 'indication')


reporter.report('Checking each case now has exactly one assigned indication '
                "(remember this assignment may be 'indication missing')")
assert new_ind_df.loc[new_ind_df.sum(1) == 1].shape[0] == new_ind_df.shape[0]


reporter.first('Adding new indications encoding to existing data.')
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
