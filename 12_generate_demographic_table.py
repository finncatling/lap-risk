import os
import pandas as pd

from utils.constants import (
    DATA_DIR,
    TABLES_OUTPUT_DIR,
    INTERNAL_OUTPUT_DIR,
    NOVEL_MODEL_OUTPUT_DIR
)
from utils.io import load_object
from utils.split import TrainTestSplitter, tt_splitter_all_test_case_modifier
from utils.indications import INDICATION_VAR_NAME, IndicationNameProcessor
from utils.model.novel import LactateAlbuminImputer
from utils.table import DemographicTableVariable, generate_demographic_table
from utils.report import Reporter


reporter = Reporter()
reporter.title("Generate demographic table")


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
for variable_name in ("S01Sex",):
    df[variable_name] = raw_df[variable_name]


reporter.report("Loading data needed for train-test splitting")
tt_splitter: TrainTestSplitter = load_object(
    os.path.join(INTERNAL_OUTPUT_DIR, "01_train_test_splitter.pkl")
)


reporter.report("Modifying train-test splitter to include all test cases")
tt_splitter_all_test_cases = tt_splitter_all_test_case_modifier(tt_splitter)


reporter.report("Sanitising indication names")
albumin_imputer: LactateAlbuminImputer = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "07_albumin_imputer.pkl")
)
indication_names = IndicationNameProcessor(
    multi_category_levels=albumin_imputer.multi_cat_vars,
    remove_missing_category=True,
    max_line_length=100  # Long enough so never uses new line character
)


reporter.report('Specifying demographic table variables')
table_1_variables = (
    DemographicTableVariable(
        "S01AgeOnArrival",
        "Age (years)",
        True,
        'continuous'
    ),
    DemographicTableVariable(
        "S01Sex",
        "Female",
        False,
        'binary',
    ),
    DemographicTableVariable(
        "Target",
        "Died",
        True,
        'binary'
    ),
    DemographicTableVariable(
        "S03ASAScore",
        "ASA physical status",
        True,
        'ordinal_multicat',
        0,
        '1 2 3 4 5'.split()
    ),
    DemographicTableVariable(
        "S03CardiacSigns",
        "Cardiovascular status",
        True,
        'multicat',
        0,
        [
            'No failure',
            'CVS medications',
            'Peripheral oedema / warfarin',
            'Cardiomegaly / raised JVP'
        ]
    ),
    DemographicTableVariable(
        "S03RespiratorySigns",
        "Respiratory status",
        True,
        'multicat',
        0,
        [
            'No dyspnoea',
            'Mild COPD / dyspnoea',
            'Moderate COPD / dyspnoea',
            'Fibrosis / consolidation / severe dyspnoea'
        ]
    ),
    DemographicTableVariable(
        "S03Pulse",
        "Heart rate (BPM)",
        True,
        'continuous'
    ),
    DemographicTableVariable(
        "S03ECG",
        "Non-sinus rhythm",
        True,
        'binary'
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
        # "Creatinine (mmol/L)",
        "Creatinine (mg/dL)",
        True,
        'continuous',
        2
    ),
    DemographicTableVariable(
        "S03Urea",
        # "Urea (mmol/L)",
        "BUN (mg/dL)",
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
    DemographicTableVariable(
        "S02PreOpCTPerformed",
        "CT performed",
        True,
        'binary'
    ),
    DemographicTableVariable(
        "S03Pred_Peritsoil",
        "Peritoneal soiling",
        True,
        'multicat',
        0,
        ["None", "Serous", "Local pus", "Free pus / blood / faeces"]
    ),
    DemographicTableVariable(
        "S03DiagnosedMalignancy",
        "Malignancy",
        True,
        'multicat',
        0,
        ["None", "Primary only", "Nodal mets.", "Distant mets."]
    ),
    DemographicTableVariable(
        INDICATION_VAR_NAME,
        'Indication',
        True,
        'multicat',
        0,
        indication_names.sanitized
    )
)


reporter.report('Constructing and saving demographic table (test fold with '
                'just current model-complete cases)')
generate_demographic_table(
    variables=table_1_variables,
    df=df,
    tts=tt_splitter,
    us_units=True,
    output_filepath=os.path.join(
        TABLES_OUTPUT_DIR, 'demographic_table_us_units.csv')
)


reporter.report('Constructing and saving demographic table (test fold with '
                'all cases)')
generate_demographic_table(
    variables=table_1_variables,
    df=df,
    tts=tt_splitter_all_test_cases,
    us_units=True,
    output_filepath=os.path.join(
        TABLES_OUTPUT_DIR, 'demographic_table_all_test_cases_us_units.csv')
)


reporter.last('Done.')
