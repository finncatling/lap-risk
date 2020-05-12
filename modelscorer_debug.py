import os
from utils.io import load_object
from utils.report import Reporter
from utils.evaluate import ModelScorer

reporter = Reporter()
reporter.first('Loading SplitterTrainerPredictor for use in model evaluation')
stp = load_object(os.path.join('outputs', 'splitter_trainer_predictor.pkl'))


reporter.report('Scoring model performance')
scorer = ModelScorer(stp.y_test, stp.y_pred)
scorer.calculate_scores()
scorer.print_scores(dec_places=3)
