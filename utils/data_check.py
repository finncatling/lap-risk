import os

from utils.io import load_object


def get_initial_df_specification(
    specification_filepath: str = os.path.join(
        os.pardir, 'config', 'initial_df_univariate_specification.pkl')
) -> dict:
    """Specification for the continuous and categorical variables in the NELA
        data. Contains all the variables names, the categories (and
        associated probabilities) for each categorical variable, plus parameters
        for the parametric distribution that most closely fits the univariate
        empirical distributions of each continuous variable."""
    return load_object(specification_filepath)
