import os

RANDOM_SEED = 1

# Location of source data, plus this data after initial manual wrangling
DATA_DIR = os.path.join(os.pardir, 'nelarisk', 'data')

# Location where (non-PHI) outputs are saved
EXTERNAL_OUTPUT_DIR = os.path.join(os.pardir, 'lap-risk-outputs')
STATS_OUTPUT_DIR = os.path.join(EXTERNAL_OUTPUT_DIR, 'statistics')

# For use when plotting gam output
N_GAM_CONFIDENCE_INTERVALS = 5
GAM_OUTER_CONFIDENCE_INTERVALS = (0.025, 0.975)
