import os
import pandas as pd

from utils.constants import DATA_DIR, TABLES_OUTPUT_DIR, INTERNAL_OUTPUT_DIR
from utils.io import load_object
from utils.split import TrainTestSplitter
from utils.table import Table1Variable
from utils.report import Reporter


reporter = Reporter()
reporter.title("Generate Table 1 (demographic information")


reporter.report('Loading raw data')
raw_df = pd.read_pickle(os.path.join(
    DATA_DIR,
    "lap_risk_df_after_univariate_wrangling_all_variables.pkl"
))


reporter.report('Loading preprocessed data')
# These data are not Winsorized, so safe to use for summary stats
df = pd.read_pickle(os.path.join(DATA_DIR, "05_preprocessed_df.pkl"))


reporter.report("Loading data needed for train-test splitting")
tt_splitter: TrainTestSplitter = load_object(
    os.path.join(INTERNAL_OUTPUT_DIR, "01_train_test_splitter.pkl")
)


reporter.report('Defining Table 1 specification')
table_1_spec = (
    Table1Variable(
        "S01AgeOnArrival",
        "Age (years)",
        True,
        'continuous'
    ),
    Table1Variable(
        "S01Sex",
        "Female",
        False,
        'binary',
        True
    ),
    Table1Variable(
        "Target",
        "Died",
        True,
        'binary'
    ),
    Table1Variable(
        "S03ASAScore",
        "ASA physical status",
        True,
        'ordinal_multicat'
    ),
    Table1Variable(
        "S03Pulse",),
    Table1Variable(
        "S03SystolicBloodPressure",
        "Systolic pressure (mmHg)"),
    Table1Variable(
        "S03CardiacSigns",
        "Cardiovascular status",
        True,
        'ordinal_multicat'
    ),
    Table1Variable(
        "S03RespiratorySigns",
        "Respiratory status",
        True,
        'ordinal_multicat'
    ),
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
    #     "S03GlasgowComaScore",
    #     "Glasgow Coma Score",
    #     True,
    #     "ordinal_multicat"
    # ),
    # Table1Variable(
    #     "S03SerumCreatinine",
    #     "Creatinine (mmol/L)",
    #     True,
    #     'continuous'
    # ),
    # Table1Variable(
    #     "S03Sodium",),
    # Table1Variable(
    #     "S03Potassium",),
    # Table1Variable(
    #     "S03Urea",),
    # Table1Variable(
    #     "S03WhiteCellCount",
    #     r"White cell count ($\times$10${^9}$/L)"),
    # Table1Variable(
    #     "S02PreOpCTPerformed",),
    # Table1Variable(
    #     "S03PreOpArterialBloodLactate",),
    # Table1Variable(
    #     "S03PreOpLowestAlbumin",),
    # Table1Variable(
    #     "S03Pred_TBL",),
    # Table1Variable(
    #     "S03NumberOfOperativeProcedures",),
    # Table1Variable(
    #     "S03NCEPODUrgency",),
)


# TODO: 6 Columns - variable, missingness (all), all, train0, test0, in n. model
# TODO: Table needs key to define ordinal categories
