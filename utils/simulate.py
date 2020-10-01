import numpy as np
import pandas as pd
from scipy import stats
from typing import Iterable

from utils.model.novel import get_indication_variable_names


class TruncatedDistribution:
    """Thin wrapper around a frozen scipy continuous random variable to allow
        truncation of samples from that variable.

        Adapted from https://stackoverflow.com/a/11492527/1684046 """

    def __init__(
        self,
        rv: stats._distn_infrastructure.rv_frozen,
        lower_bound: float = -np.inf,
        upper_bound: float = np.inf,
        random_state: np.random.RandomState = None
    ):
        """
        Args:
            rv: Frozen scipy continuous random variables, e.g. norm(0, 1)
            lower_bound: Samples should be >= this value
            upper_bound: Samples should be <= this value
            random_state: Pass this for deterministic sampling
        """
        self.rv = rv
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        if random_state is None:
            self.rnd = np.random.RandomState()
        else:
            self.rnd = random_state

    @property
    def lower_quantile(self) -> float:
        return self.rv.cdf(self.lower_bound)

    @property
    def normaliser(self) -> float:
        return self.rv.cdf(self.upper_bound) - self.lower_quantile

    def sample(self, n_samples: int) -> np.ndarray:
        """
        Args:
            n_samples: Number of samples from truncated random variables

        Returns:
            1D ndarray of samples
        """
        quantiles = (self.rnd.random_sample(n_samples) *
                     self.normaliser +
                     self.lower_quantile)
        return self.rv.ppf(quantiles)


def simulate_initial_df(
    specification: dict,
    n_rows: int,
    n_hospitals: int,
    missing_frac: float,
    complete_indications: bool,
    complete_target: bool,
    complete_institution: bool,
    round_1dp_variables: Iterable[str],
    random_seed
) -> pd.DataFrame:
    """Simulates NELA data after initial univariate wrangling and variable
        selection (i.e. the output of 0_univariate_wrangling.ipynb), which is
        input to 01_train_test_split.py

    Args:
        specification: Specification for the continuous and
            categorical variables in the NELA data. Load this from
            config/initial_df_univariate_specification.pkl
        n_rows: Number of rows in DataFrame
        n_hospitals: Number of hospitals in DataFrame
        missing_frac: Fraction of data that is missing (for incomplete
            variables)
        complete_indications: if True, don't introduce missingness into the
            binary indications variables
        complete_target: if True, don't introduce missingness into the
            target variable
        complete_institution: if True, don't introduce missingness into the
            hospital / trust ID variable
        round_1dp_variables: These variables are rounded to one decimal place.
            All other continuous variables are rounded to zero decimal places.
        random_seed

    Returns:
        Simulated NELA data
    """
    rnd = np.random.RandomState(random_seed)
    df = pd.DataFrame()

    # Create institution (hospital or trust) ID column
    df[specification['var_names']['institutions'][0]] = rnd.randint(
        n_hospitals, size=n_rows)

    # Create other categorical columns
    for var_name, probabilities in specification['cat_fits'].items():
        cat_samples_i_2d = np.random.multinomial(
            n=1,
            pvals=probabilities.values,
            size=n_rows)
        cat_samples_i_1d = np.argmax(cat_samples_i_2d, 1)
        cat_samples = [probabilities.index[i] for i in cat_samples_i_1d]
        df[var_name] = cat_samples

    # Create continuous columns
    for var_name, params in specification['cont_fits'].items():
        dist = getattr(stats, params['dist_name'])
        # TODO: Truncation is slow - find out why and try and fix. Is it .ppf()?
        truncated_rv = TruncatedDistribution(
            rv=dist(*params['dist_params']),
            lower_bound=params['min'],
            upper_bound=params['max'],
            random_state=rnd
        )
        df[var_name] = truncated_rv.sample(n_rows)
        if var_name in round_1dp_variables:
            df[var_name] = df[var_name].round(1)
        else:
            df[var_name] = df[var_name].round(0)

    # Make list of columns which will have missing values
    missing_columns = df.columns.tolist()
    if complete_indications:
        for c in get_indication_variable_names(df.columns):
            missing_columns.remove(c)
    if complete_target:
        missing_columns.remove(specification['var_names']['target'])
    if complete_institution:
        missing_columns.remove(specification['var_names']['institutions'][0])

    # Introduce missing values
    for col in missing_columns:
        df.loc[df.sample(
            frac=missing_frac,
            random_state=rnd
        ).index, col] = np.nan

    return df
