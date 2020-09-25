from typing import List
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from scipy import stats


def cat_data(df: pd.DataFrame, col_name: str, binary: bool = False) -> None:
    """Display some useful attributes of a categorical feature. Note
        that binary=True alters the underlying input DataFrame."""
    if binary:
        df.loc[:, col_name] = df[col_name].apply(convert_to_binary)
    print(f"Data type: {df[col_name].dtype}")
    print(f"Missing rows: {missingness_perc(df, col_name)}%")
    dot_chart(df[col_name].value_counts(dropna=False), col_name)


def con_data(df: pd.DataFrame, col_name: str, bins: int = 50) -> None:
    """Display some useful attributes of a continuous feature.
        NB. Normality test is likely to fail even for Gaussian-looking
        distributions given that the dataset is large."""
    print(f"Data type: {df[col_name].dtype}")
    print(f"Missing rows: {missingness_perc(df, col_name)}%")
    display(pd.DataFrame(df[col_name].describe()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df[col_name].values, bins=bins)
    stats.probplot(df.loc[df[col_name].notnull(), col_name].values,
                   plot=axes[1])
    plt.show()


def dummies_data(
    df: pd.DataFrame, categories: list, concept: str, col_name_prefix_len: int
) -> None:
    """Summarises groups of columns containing related indicator
        variables.
    
    Args:
        df: DataFrame containing the columns of indicator variables
        categories: column names of the related variables
        concept: the thing that groups the columns together
        col_name_prefix_len: length of the common prefix to all
            the column names
    """
    ddf = df[categories].copy()

    for c in ddf.columns:
        ddf.loc[ddf[c] != 1, c] = 0
        ddf[c] = ddf[c].astype(int)

    ind_vc = ddf.sum(0).sort_values(ascending=False)
    ind_vc.index = ind_vc.index.str[col_name_prefix_len:]
    dot_chart(ind_vc, f"How common is each {concept}?")

    dot_chart(
        ddf.sum(1).value_counts().sort_index(),
        f"How many {concept}s per laparotomy?"
    )

    for c in ddf.columns:
        cat_data(df, c)


def last_digit(df: pd.DataFrame, col_name: str) -> None:
    """Inspect the frequency of the last digit a DataFrame column."""
    digits = []
    for i in df[col_name]:
        k = i % 10
        digits.append(k)
    plt.figure(2)
    plt.hist(digits)


def missingness_perc(df: pd.DataFrame, col_name: str) -> float:
    """Calculate percent missing values for column of data."""
    return df[col_name].isnull().sum() / df.shape[0] * 100


def convert_to_binary(x):
    """Convert binary input (e.g. DataFrame row) coded as 1s and 2s,
        to 0s and 1s."""
    if x == 1:
        return 0
    elif x == 2:
        return 1


def dot_chart(vc: pd.Series, title: str) -> None:
    """Plot Bill Cleveland-style dot chart for categorical features.
    
    Args:
        vc: Output from pandas value_counts(dropna=False) on
            categorical feature
        title: Name of feature
    """
    f, ax = plt.subplots(figsize=(6, vc.shape[0] / 3))

    data = vc.values
    data_labels = list(vc.index)

    n = len(data)
    y = np.arange(n)[::-1]

    ax.plot(data, y, marker=".", linestyle="", markersize=10, markeredgewidth=0)

    ax.set_yticks(list(range(n)))
    ax.set_yticklabels(data_labels[::-1])

    ax.set_ylim(-1, n)

    ax.tick_params(axis="y", which="major", right=True, left=True, color="0.8")
    ax.grid(axis="y", which="major", zorder=-10, linestyle="-", color="0.8")

    ax2 = ax.twinx()
    ax2.set_yticks(list(range(n)))
    total = data.sum()
    count_labels = ["{} ({:.2f}%)".format(x, 100 * x / total) for x in
                    data[::-1]]
    ax2.set_yticklabels(count_labels)
    ax2.set_ylim(-1, n)

    ax.set_title(title)

    plt.show()


def multi_dot_chart(
    vcs: List[pd.Series], labels: List[str], normalise: bool = True
) -> None:
    """Plot Bill Cleveland-style dot chart for categorical features,
        where categories are stratified according to something.
    
    Args:
        vcs: List of outputs from pandas value_counts(dropna=False) on
            categorical feature
        labels: Labels plotted for stratum, i.e. for each element of vcs
        normalise: If True, makes counts in each stratum sum to 1
    """
    f, ax = plt.subplots(figsize=(6, vcs[0].shape[0] / 3))

    for i, vc in enumerate(vcs):
        if not i:
            data_labels = list(vc.index)
            n = len(data_labels)
            y = np.arange(n)[::-1]

        data = vc.reindex(data_labels).values
        if normalise:
            data = data / data.sum()

        ax.plot(
            data,
            y,
            marker=".",
            linestyle="",
            markersize=10,
            markeredgewidth=0,
            label=labels[i],
        )

    ax.set_yticks(list(range(n)))
    ax.set_yticklabels(data_labels[::-1])
    ax.set_ylim(-1, n)
    ax.tick_params(axis="y", which="major", right=True, left=True, color="0.8")
    ax.grid(axis="y", which="major", zorder=-10, linestyle="-", color="0.8")
    if normalise:
        ax.set_xlabel("Fraction of per-stratum total")

    plt.legend(loc="lower right")
    plt.show()


def report_ohe_category_assignment(data: pd.DataFrame,
                                   category_name: str) -> None:
    """Given a DataFrame where the columns one-hot-encode a categorical
        variable, reports the number / proportion of rows where a category is
        assigned."""
    if data.loc[data.sum(1) > 1].shape[0]:
        warn("Data contains some cases with more than one category encoded")
    n_cases_total = data.shape[0]
    n_cases_assigned = data.loc[data.sum(1) == 1].shape[0]
    print(
        f"{n_cases_assigned} cases out of {n_cases_total} ("
        f"{100 * np.round(n_cases_assigned / n_cases_total, 3)}%) now have "
        f"an assigned {category_name}"
    )
