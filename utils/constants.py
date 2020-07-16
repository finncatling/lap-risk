import os
import numpy as np

RANDOM_SEED = 1

# Location of source data, plus this data after initial manual wrangling
DATA_DIR = os.path.join(os.pardir, "nelarisk", "data")

# Locations where 'internal' outputs are saved for use later in the analysis
INTERNAL_OUTPUT_DIR = "outputs"
CURRENT_MODEL_OUTPUT_DIR = os.path.join(INTERNAL_OUTPUT_DIR, "current_model")
NOVEL_MODEL_OUTPUT_DIR = os.path.join(INTERNAL_OUTPUT_DIR, "novel_model")

# Locations where 'external' (non-PHI) outputs are saved
EXTERNAL_OUTPUT_DIR = os.path.join(os.pardir, "lap-risk-outputs")
STATS_OUTPUT_DIR = os.path.join(EXTERNAL_OUTPUT_DIR, "statistics")
FIGURES_OUTPUT_DIR = os.path.join(EXTERNAL_OUTPUT_DIR, "figures")

# For use in model evaluation
CALIB_GAM_N_SPLINES = 5
CALIB_GAM_LAM_CANDIDATES = np.logspace(-3, 3)

# For use when plotting gam output
GAM_CONFIDENCE_INTERVALS = (95, 70, 45, 20)
