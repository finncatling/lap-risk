import os

import numpy as np
import pandas as pd
from pprint import PrettyPrinter

from utils.constants import DATA_DIR, STATS_OUTPUT_DIR
from utils.data_check import load_nela_data_and_sanity_check
from utils.io import save_object
from utils.indications import (
    INDICATION_PREFIX,
    MISSING_IND_CATEGORY,
    get_indication_variable_names,
    get_common_single_indications,
    ohe_single_indications, report_ohe_category_assignment
)
from utils.report import Reporter


SINGLE_IND_FREQUENCY_THRESHOLD = 1200


reporter = Reporter()
reporter.title(
    "Rationalise indications for surgery (retaining common single "
    "indications and reassigning common combinations of "
    "indications) in preparation for data input to the novel risk "
    "model. See additional data exploration and summary statistics "
    "in the consolidate_indications.ipynb notebook"
)


reporter.report("Loading manually-wrangled NELA data")
df = load_nela_data_and_sanity_check()


reporter.report("Isolating indication variables")
indications = get_indication_variable_names(df.columns)
ind_df = df[indications].copy().astype(int)


reporter.report(
    "Defining 'common single indications' as those that occur in "
    f"isolation >= {SINGLE_IND_FREQUENCY_THRESHOLD} times"
)
common_single_inds = get_common_single_indications(
    indication_df=ind_df,
    frequency_threshold=SINGLE_IND_FREQUENCY_THRESHOLD
)


print("The common single indications are:")
PrettyPrinter().pprint(common_single_inds)


reporter.first(
    "Making a new one-hot-encoded DataFrame containing just the common single "
    "indications where they occur in isolation"
)
ohe_indication_df = ohe_single_indications(
    indication_df=ind_df,
    indication_subset_names=common_single_inds
)
report_ohe_category_assignment(ohe_indication_df, "indication")


reporter.first(
    "Changing the most-commonly-occurring pairs of "
    "indications to the closest single indication"
)
for ind_pair, keep_ind_index in (
    (("Perforation", "Peritonitis"), 1),
    (("SmallBowelObstruction", "IncarceratedHernia"), 0),
    (("Perforation", "AbdominalAbscess"), 0),
):
    print(f"Changing {ind_pair} to {ind_pair[keep_ind_index]}")
    ohe_indication_df.loc[
        (
            (ind_df.sum(1) == 2)
            & (ind_df[f"{INDICATION_PREFIX}{ind_pair[0]}"] == 1)
            & (ind_df[f"{INDICATION_PREFIX}{ind_pair[1]}"] == 1)
        ),
        f"{INDICATION_PREFIX}{ind_pair[keep_ind_index]}",
    ] = 1
    report_ohe_category_assignment(ohe_indication_df, "indication")


reporter.first(
    "Changing the most-commonly-occurring trios of "
    "indications to the closest single indication"
)
for ind_trio, keep_ind_index in (
    (("Perforation", "Peritonitis", "AbdominalAbscess"), 1),
):
    print(f"Changing {ind_trio} to {ind_trio[keep_ind_index]}")
    ohe_indication_df.loc[
        (
            (ind_df.sum(1) == 3)
            & (ind_df[f"{INDICATION_PREFIX}{ind_trio[0]}"] == 1)
            & (ind_df[f"{INDICATION_PREFIX}{ind_trio[1]}"] == 1)
            & (ind_df[f"{INDICATION_PREFIX}{ind_trio[2]}"] == 1)
        ),
        f"{INDICATION_PREFIX}{ind_trio[keep_ind_index]}",
    ] = 1
    report_ohe_category_assignment(ohe_indication_df, "indication")


reporter.first(
    "Assigning cases with no indication in the original data to a "
    "new 'indication missing' category"
)
ohe_indication_df[MISSING_IND_CATEGORY] = np.zeros(df.shape[0])
ohe_indication_df.loc[ind_df.sum(1) == 0, MISSING_IND_CATEGORY] = 1
report_ohe_category_assignment(ohe_indication_df, "indication")


reporter.first(
    "Assigning cases with >1 indication, at least one of which is "
    "a common single indication, and where the case was not "
    "reassigned earlier, to the missing category."
)
ohe_indication_df.loc[
    (
        (ind_df.sum(1) > 1)
        & (ind_df[common_single_inds].sum(1) > 0)
        & (ohe_indication_df.sum(1) == 0)
    ),
    MISSING_IND_CATEGORY,
] = 1
report_ohe_category_assignment(ohe_indication_df, "indication")


reporter.first(
    "Assigning remaining cases (those with one or more indications "
    "in the original data, but where none of these indications is a "
    "common single indication) to the 'other indication' "
    "category"
)
ohe_indication_df.loc[
    ohe_indication_df.sum(1) == 0,
    f"{INDICATION_PREFIX}Other"
] = 1
report_ohe_category_assignment(ohe_indication_df, "indication")


reporter.first("Confirming each case now has exactly one assigned indication")
assert ohe_indication_df.loc[
   ohe_indication_df.sum(1) == 1
].shape[0] == ohe_indication_df.shape[0]


reporter.report("Adding new indications encoding to existing data.")
df = df.drop(indications, axis=1)
df_with_new_inds = pd.concat((df, ohe_indication_df.astype(float)), axis=1)


reporter.report("Saving data for later use")
df_with_new_inds.to_pickle(
    os.path.join(DATA_DIR, "04_output_df.pkl")
)


reporter.report("Saving a few summary statistics")
ci_stats = {
    "indication_counts": ohe_indication_df.sum(0).sort_values(ascending=False)
}
ci_stats["indication_proportions"] = (
    ci_stats["indication_counts"] / ohe_indication_df.shape[0]
)
save_object(
    ci_stats,
    os.path.join(STATS_OUTPUT_DIR, "04_indication_stats.pkl")
)


reporter.last("Done.")
