import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from utils.io import load_object
from utils.constants import (
    NOVEL_MODEL_OUTPUT_DIR,
    GAM_CONFIDENCE_INTERVALS
)
from utils.plot.helpers import generate_ci_quantiles
from utils.model.novel import LogOddsTransformer


nm_refit = load_object(os.path.join(
    NOVEL_MODEL_OUTPUT_DIR,
    "11_novel_model_lacalb_sensitivity.pkl"))
nm_original = load_object(os.path.join(
    NOVEL_MODEL_OUTPUT_DIR,
    "08_novel_model.pkl"))


terms = ((6, 19), (7, 17))
cis = generate_ci_quantiles(GAM_CONFIDENCE_INTERVALS)
n_cis = len(cis)
colors = "tab:blue", "tab:orange"
legend_locs = "upper left", "upper right"
titles = "Lactate (mmol/L)", "Albumin (g/L)"
trans = LogOddsTransformer()
trans_centre = trans.inverse_transform(np.zeros((1, 1))).flatten()[0]


for space_i, space in enumerate(
    ('Log mortality odds', 'Mortality risk ratio')
):
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax = ax.ravel()
    
    y_lim = [np.inf, -np.inf]
    
    for ax_i, (term_i, xx_i) in enumerate(terms):
        lines = []

        for model_i, model in enumerate((nm_original, nm_refit)):

            xx = model.models[0].generate_X_grid(term=term_i, n=100)
            _, confi = model.models[0].partial_dependence(
                term=term_i, X=xx, quantiles=cis)
            
            if space_i:
                confi = trans.inverse_transform(confi) / trans_centre

            if confi.min() < y_lim[0]:
                y_lim[0] = confi.min()
            if confi.max() > y_lim[1]:
                y_lim[1] = confi.max()

            for k in range(n_cis):
                ax[ax_i].fill_between(
                    xx[:, xx_i],
                    confi[:, k],
                    confi[:, -(k + 1)],
                    alpha=1 / n_cis,
                    color=colors[model_i],
                    lw=0.0)

            lines.append(Line2D([0], [0], color=colors[model_i]))

        ax[ax_i].legend(
            lines, ('original', 'refit'), loc=legend_locs[ax_i])
        ax[ax_i].set_xlim(xx[:, xx_i].min(), xx[:, xx_i].max())
        ax[ax_i].set_title(titles[ax_i])
        ax[ax_i].set_ylabel(space)

    for ax_i, _ in enumerate(ax):
        ax[ax_i].set_ylim(y_lim)


    fig.tight_layout()
    plt.show()

