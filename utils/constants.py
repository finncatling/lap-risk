import os
import numpy as np

RANDOM_SEED = 1

"""
Project root directory. This is dependent on constants.py being one directory
down from the root directory.
"""
ROOT_DIR = os.path.split(
    os.path.dirname(os.path.abspath(__file__))
)[0]

# Folder where the raw data is kept
RAW_NELA_DATA_FILEPATH = os.path.join(
    ROOT_DIR,
    os.pardir,
    os.pardir,
    'extract',
    'datadownload_20190524',
    'hqip254NELAdata21May2019.csv'
)

# Folder where data after initial wrangling are kept
DATA_DIR = os.path.join(ROOT_DIR, os.pardir, "nelarisk", "data")

# Path to data after initial manual wrangling
NELA_DATA_FILEPATH = os.path.join(
    DATA_DIR,
    "lap_risk_df_after_univariate_wrangling.pkl"
)

# Locations where 'internal' outputs are saved for use later in the analysis
INTERNAL_OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
CURRENT_MODEL_OUTPUT_DIR = os.path.join(INTERNAL_OUTPUT_DIR, "current_model")
CURRENT_MODEL_FEATHER_DIR = os.path.join(CURRENT_MODEL_OUTPUT_DIR, "feather")
NOVEL_MODEL_OUTPUT_DIR = os.path.join(INTERNAL_OUTPUT_DIR, "novel_model")

# Locations where 'external' (non-PHI) outputs are saved
EXTERNAL_OUTPUT_DIR = os.path.join(ROOT_DIR, os.pardir, "lap-risk-outputs")
STATS_OUTPUT_DIR = os.path.join(EXTERNAL_OUTPUT_DIR, "statistics")
FIGURES_OUTPUT_DIR = os.path.join(EXTERNAL_OUTPUT_DIR, "figures")
TABLES_OUTPUT_DIR = os.path.join(EXTERNAL_OUTPUT_DIR, "tables")
PRODUCTION_OUTPUT_DIR = os.path.join(EXTERNAL_OUTPUT_DIR, "production")

# For use in model evaluation
CALIB_GAM_N_SPLINES = 5
CALIB_GAM_LAM_CANDIDATES = np.logspace(-3, 3)

# For use when plotting gam output
GAM_CONFIDENCE_INTERVALS = (95, 70, 45, 20)
