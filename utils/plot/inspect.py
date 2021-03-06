from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from utils.model.novel import LactateAlbuminImputer
from utils.plot.helpers import convert_creatinine_urea


def plot_creatinine_urea_redaction(
    pre_redaction_df: pd.DataFrame,
    post_redaction_df: pd.DataFrame,
    us_units: bool = False
) -> Tuple[Figure, Axes]:
    fig, axs = plt.subplots(1, 2, figsize=(8.1, 4))
    axs = axs.ravel()
    titles = ['Raw data', 'After redaction']

    for i, data in enumerate([pre_redaction_df, post_redaction_df]):
        xlim = np.array([-35., 1235.])
        ylim = np.array([-10., 310.])

        if us_units:
            data = convert_creatinine_urea(data)
            creatinine_label = r'Creatinine (mg dL$^{-1}$)'
            urea_label = r'BUN (mg dL$^{-1}$)'
            xlim /= 88.42
            ylim /= 0.357
        else:
            creatinine_label = r'Creatinine (mmol L$^{-1}$)'
            urea_label = r'Urea (mmol L$^{-1}$)'

        axs[i].scatter(
            data['S03SerumCreatinine'].values,
            data['S03Urea'].values,
            alpha=0.1,
            s=5
        )
        axs[i].set(
            xlabel=creatinine_label,
            ylabel=urea_label,
            title=titles[i],
            xlim=list(xlim),
            ylim=list(ylim)
        )

    fig.tight_layout()
    return fig, axs


def inspect_transformed_lac_alb(
    imputer: LactateAlbuminImputer,
    train_test_split_i: int,
    hist_args: Dict[str, Any] = {"bins": 20, "alpha": 0.5},
) -> Tuple[Figure, Axes]:
    """Compare original target (albumin or lactate) and its transformation."""
    target_train, _, target_test, _ = imputer._split(train_test_split_i)
    target = {"train": target_train, "test": target_test}

    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax = ax.ravel()

    for i, fold in enumerate(("train", "test")):
        target[fold] = imputer._get_observed_values(
            fold, train_test_split_i, target[fold]
        )
        obs_target = imputer._winsorize(train_test_split_i, target[fold])
        ax[i].hist(obs_target.values.flatten(), label="original", **hist_args)

        obs_target_trans = imputer.transformers[train_test_split_i].transform(
            obs_target
        )
        ax[i].hist(
            obs_target_trans.values.flatten(),
            label="transformed",
            **hist_args
        )

        ax[i].set(title=fold)
        ax[i].legend()

    return fig, ax
