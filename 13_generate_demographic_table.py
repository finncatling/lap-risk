import os
import pandas as pd

from utils.constants import DATA_DIR, TABLES_OUTPUT_DIR, INTERNAL_OUTPUT_DIR
from utils.io import load_object
from utils.split import TrainTestSplitter, tt_splitter_all_test_case_modifier
from utils.table import DemographicTableVariable, generate_demographic_table
from utils.report import Reporter


reporter = Reporter()
reporter.title("Generate Table 1 (demographic information")


reporter.report('Loading preprocessed data')
# These data are not Winsorized, so safe to use for summary stats
df = pd.read_pickle(os.path.join(DATA_DIR, "05_preprocessed_df.pkl"))


reporter.report(
    'Loading raw data and adding absent variables to preprocessed data'
)
raw_df = pd.read_pickle(os.path.join(
    DATA_DIR,
    "lap_risk_df_after_univariate_wrangling_all_variables.pkl"
))
for variable_name in ("S01Sex",):  # TODO: Add more variables here if needed
    df[variable_name] = raw_df[variable_name]


reporter.report("Loading data needed for train-test splitting")
tt_splitter: TrainTestSplitter = load_object(
    os.path.join(INTERNAL_OUTPUT_DIR, "01_train_test_splitter.pkl")
)


reporter.report("Modifying train-test splitter to include all test cases")
tt_splitter = tt_splitter_all_test_case_modifier(tt_splitter)


reporter.report('Specifying Table 1 variables')
table_1_variables = (
    DemographicTableVariable(
        "S01AgeOnArrival",
        "Age (years)",
        True,
        'continuous'
    ),
    DemographicTableVariable(
        "S03Pulse",
        "Heart rate (BPM)",
        True,
        'continuous'
    ),
    DemographicTableVariable(
        "S03SystolicBloodPressure",
        "Systolic pressure (mmHg)",
        True,
        'continuous'
    ),
    DemographicTableVariable(
        "S03Sodium",
        "Sodium (mmol/L)",
        True,
        'continuous'
    ),
    DemographicTableVariable(
        "S03Potassium",
        "Potassium (mmol/L)",
        True,
        'continuous',
        1
    ),
    DemographicTableVariable(
        "S03WhiteCellCount",
        r"White cell count (x10^9/L)",
        True,
        'continuous',
        1
    ),
    DemographicTableVariable(
        "S03SerumCreatinine",
        "Creatinine (mmol/L)",
        True,
        'continuous'
    ),
    DemographicTableVariable(
        "S03Urea",
        "Urea (mmol/L)",
        True,
        'continuous',
        1
    ),
    DemographicTableVariable(
        "S03PreOpArterialBloodLactate",
        "Lactate (mmol/L)",
        True,
        'continuous',
        1
    ),
    DemographicTableVariable(
        "S03PreOpLowestAlbumin",
        "Albumin (g/L)",
        True,
        'continuous'
    ),
    DemographicTableVariable(
        "S03GlasgowComaScore",
        "Glasgow Coma Score",
        True,
        "continuous"  # this approximation suffices here
    ),
    # Table1Variable(
    #     "S01Sex",
    #     "Female",
    #     False,
    #     'binary',
    # ),
    # Table1Variable(
    #     "Target",
    #     "Died",
    #     True,
    #     'binary'
    # ),
    # Table1Variable(
    #     "S03ASAScore",
    #     "ASA physical status",
    #     True,
    #     'ordinal_multicat'
    # ),
    # Table1Variable(
    #     "S03CardiacSigns",
    #     "Cardiovascular status",
    #     True,
    #     'ordinal_multicat'
    # ),
    # Table1Variable(
    #     "S03RespiratorySigns",
    #     "Respiratory status",
    #     True,
    #     'ordinal_multicat'
    # ),
    # Table1Variable(
    #     "S03ECG",  # TODO: Need to use binarized version
    #     "Non-sinus rhythm",
    #     True,
    #     'binary'
    # ),
    # Table1Variable(
    #     "S03Pred_Peritsoil",
    #     "Peritoneal soiling",
    #     True,
    #     'ordinal_multicat'
    # ),
    # Table1Variable(
    #     "S03DiagnosedMalignancy",
    #     "Malignancy",
    #     True,
    #     'ordinal_multicat'
    # ),
    # Table1Variable(
    #     "S03WhatIsTheOperativeSeverity",),
    # Table1Variable(
    #     "S02PreOpCTPerformed",),
    # Table1Variable(
    #     "S03Pred_TBL",),
    # Table1Variable(
    #     "S03NumberOfOperativeProcedures",),
    # Table1Variable(
    #     "S03NCEPODUrgency",),
)


print(
    generate_demographic_table(
        variables=table_1_variables,
        df=df,
        modified_tts=tt_splitter,
        output_filepath='FILL THIS IN PROPERLY'
    )
)


# TODO: 6 Columns - variable, missingness (all), all, train0, test0, in n. model
# TODO: Table needs key to define ordinal categories
