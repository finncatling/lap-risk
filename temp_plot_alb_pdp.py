import os
from typing import Dict, Tuple

from utils.constants import NOVEL_MODEL_OUTPUT_DIR, FIGURES_OUTPUT_DIR
from utils.io import make_directory, load_object
from utils.model.novel import (INDICATION_VAR_NAME)
from utils.plot.helpers import sanitize_indication, plot_saver
from utils.plot.pdp import PDPTerm, plot_partial_dependence
from utils.report import Reporter

reporter = Reporter()
reporter.title('Plot albumin PDP')

reporter.report("Creating output dirs (if they don't already exist)")
make_directory(NOVEL_MODEL_OUTPUT_DIR)
make_directory(FIGURES_OUTPUT_DIR)

reporter.report('Loading previous analysis outputs needed for imputation')
multi_category_levels: Dict[str, Tuple] = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR,
                 'multi_category_levels_with_indications.pkl'))
alb_imputer = load_object(os.path.join(NOVEL_MODEL_OUTPUT_DIR,
                                       'draft_albumin_imputer.pkl'))

reporter.report('Specifying properties of albumin GAM partial dependence plot')
alb_pdp_terms = [
    PDPTerm('S01AgeOnArrival',
            'Age (years)',
            (0, 0)),
    PDPTerm('S03Sodium',
            'Sodium (mmol/L)',
            (1, 1)),
    PDPTerm('S03Potassium',
            'Potassium (mmol/L)',
            (1, 2)),
    PDPTerm('S03Urea',
            'Urea (mmol/L)',
            (2, 0)),
    PDPTerm('S03WhiteCellCount',
            r'White cell count ($\times$10${^9}$/L)',
            (1, 0)),
    PDPTerm('S03SystolicBloodPressure',
            'Systolic pressure (mmHg)',
            (0, 1)),
    PDPTerm('S03ASAScore',
            'ASA physical status',
            (2, 1),
            list(range(1, 6))),
    PDPTerm('S03DiagnosedMalignancy',
            'Malignancy',
            (2, 2),
            ['None', 'Primary\nonly', 'Nodal\nmets.', 'Distant\nmets.'],
            ['No CT', 'CT'],
            'upper left'),
    PDPTerm('S03Pred_Peritsoil',
            'Peritoneal soiling',
            (3, 0),
            ['None', 'Serous', 'Local\npus', 'Free pus /\nblood / faeces'],
            ['No CT', 'CT'],
            'upper left'),
    PDPTerm(INDICATION_VAR_NAME,
            'Indication',
            (slice(3, 5), slice(1, 3)),
            [sanitize_indication(s) for s in
             multi_category_levels[INDICATION_VAR_NAME]],
            ['No CT', 'CT'],
            'upper left'),
    PDPTerm('S03Pulse',
            'Heart rate (BPM)',
            (0, 2),
            None,
            ['Sinus', 'Arrhythmia'],
            'lower right')
]

# TODO: Flip axes given that albumin is transformed?


reporter.report('Plotting albumin imputer partial dependence plot')
plot_saver(plot_partial_dependence,
           gam=alb_imputer._imputers[0],
           pdp_terms=alb_pdp_terms,
           output_dir=FIGURES_OUTPUT_DIR,
           output_filename='alb_imputer_pdp')

reporter.last('Done.')
