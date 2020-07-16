#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding: utf-8

# # Multivariate GAM for mortality prediction

# In[1]:


import os, copy, sys, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from progressbar import progressbar as pb
from pygam import GAM, LogisticGAM, s, f, l, te
from pygam.distributions import BinomialDist
from matplotlib.lines import Line2D
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Tuple, List
from dataclasses import dataclass

sys.path.append("")
from nelarisk.constants import RANDOM_SEED
from nelarisk.helpers import load_object, save_object, check_system_resources
from nelarisk.gam import combine_mi_gams, quick_sample
from nelarisk.evaluate import evaluate_predictions
from nelarisk.constants import (
    N_GAM_CONFIDENCE_INTERVALS,
    GAM_OUTER_CONFIDENCE_INTERVALS,
)


# TODO: Make SBO the default indication category
# TODO: Consider defining edge knots for variables sparse at extremes


# In[2]:


op = load_object(os.path.join("data", "imputation_all_output.pkl"))


# In[3]:


for i, c in enumerate(op["train"]["imp_all"][0].columns):
    print(i, c)


# In[4]:


gams = []

for i in pb(range(len(op["train"]["imp_all"])), prefix="GAM fit"):
    # for i in pb(range(2), prefix='quick GAM fit'):
    gams.append(
        LogisticGAM(
            s(0, lam=200)
            + s(2, lam=220)  # age
            + s(3, lam=300)  # sodium
            + s(5, lam=50)  # potassium
            + s(7, lam=300)  # wcc
            + s(8, lam=150, n_splines=13)  # sbp
            + s(19, lam=150)  # gcs
            + s(20, lam=150)  # lactate
            + f(12, coding="dummy", lam=50)  # albumin
            + f(21, coding="dummy", lam=200)  # asa
            + f(22, coding="dummy", lam=200)  # lactate missing
            +  # albumin missing
            #         te(9, 10,  # op severity & ct
            #            lam=(200, 200),
            #            n_splines=(2, 2),
            #            spline_order=(0, 0),
            #            dtype=('categorical', 'categorical')) +
            te(
                15,
                10,  # malig & ct
                lam=(200, 200),
                n_splines=(len(op["multi_cat_levels"]["S03DiagnosedMalignancy"]), 2),
                spline_order=(0, 0),
                dtype=("categorical", "categorical"),
            )
            + te(
                16,
                10,  # perit soil & ct
                lam=(400, 200),
                n_splines=(len(op["multi_cat_levels"]["S03Pred_Peritsoil"]), 2),
                spline_order=(0, 0),
                dtype=("categorical", "categorical"),
            )
            +
            #         te(17, 10,  # ncepod urgency & ct
            #            lam=(300, 200),
            #            n_splines=(len(op['multi_cat_levels']['S03NCEPODUrgency']), 2),
            #            spline_order=(0, 0),
            #            dtype=('categorical', 'categorical')) +
            te(
                18,
                10,  # indication & ct
                lam=(30, 200),
                n_splines=(len(op["multi_cat_levels"]["Indication"]), 2),
                spline_order=(0, 0),
                dtype=("categorical", "categorical"),
            )
            + te(
                6,
                11,  # pulse & ecg
                lam=(250, 2),
                n_splines=(20, 2),
                spline_order=(3, 0),
                dtype=("numerical", "categorical"),
            )
            + te(
                13,
                14,  # cardiac & resp
                lam=150,
                n_splines=(
                    len(op["multi_cat_levels"]["S03CardiacSigns"]),
                    len(op["multi_cat_levels"]["S03RespiratorySigns"]),
                ),
                spline_order=(0, 0),
                dtype=("categorical", "categorical"),
            )
            + te(1, 4, lam=18.0)  # creat & urea
        ).fit(op["train"]["imp_all"][i].values, op["train"]["y"])
    )

gam = combine_mi_gams(gams)


# In[ ]:


gam.summary()


# In[ ]:


n_cis = N_GAM_CONFIDENCE_INTERVALS
cis = np.linspace(*GAM_OUTER_CONFIDENCE_INTERVALS, n_cis * 2)


# In[ ]:


terms = gam.terms.info["terms"][:-1]

n_rows = int(np.ceil((len(terms)) / 2) - 1)
fig, ax = plt.subplots(n_rows, 2, figsize=(8, 2.5 * n_rows))
ax = ax.ravel()

subplot_labels = {
    11: {"lines": ["No CT", "CT"], "title": "Pred. malignancy"},
    #                   11: {'lines': ['No CT', 'CT'],
    #                        'title': 'Op. severity'},
    #                   14: {'lines': ['No CT', 'CT'],
    #                        'title': 'NCEPOD urgency'},
    12: {"lines": ["No CT", "CT"], "title": "Pred. perit. soiling"},
    13: {"lines": ["No CT", "CT"], "title": "Indication"},
    14: {"lines": ["Normal ECG", "Arrhythmia"], "title": "Heart rate"},
}

for i, term in enumerate(terms):
    if term["term_type"] != "tensor_term":
        XX = gam.generate_X_grid(term=i)
        pdep, confi = gam.partial_dependence(term=i, quantiles=cis)

        j = term["feature"]

        for k in range(n_cis):
            ax[i].fill_between(
                XX[:, j],
                confi[:, k],
                confi[:, -(k + 1)],
                alpha=1 / n_cis,
                color="black",
                lw=0.0,
            )

        ax[i].set_title(op["train"]["imp_all"][0].columns[j])

    else:
        XX = gam.generate_X_grid(term=i, meshgrid=True)
        Z, confi = gam.partial_dependence(term=i, X=XX, quantiles=cis, meshgrid=True)

        if i in subplot_labels.keys():
            colours = ["blue", "red"]
            lines = []

            for l, sli in enumerate([0, -1]):
                for k in range(n_cis):
                    ax[i].fill_between(
                        XX[0][:, 0],
                        confi[:, sli, k],
                        confi[:, sli, -(k + 1)],
                        lw=0.0,
                        alpha=1 / n_cis,
                        color=colours[l],
                    )
                lines.append(Line2D([0], [0], color=colours[l]))

            ax[i].legend(lines, subplot_labels[i]["lines"], loc="upper left")
            ax[i].set_title(subplot_labels[i]["title"])

        else:
            if i == list(subplot_labels.keys())[-1] + 1:
                # Finish plotting other PD subplots before starting 3D plots
                plt.tight_layout()
                plt.show()

            #             sns.heatmap(Z, square=True)
            #             plt.show()

            # Make separate 3d plots for cont-cont tensor interactions
            ax3d = plt.axes(projection="3d")
            ax3d.plot_surface(XX[0], XX[1], Z, cmap="viridis")
            ax3d.view_init(30, 150)
            ax3d.set_xlabel(
                op["train"]["imp_all"][0].columns[term["terms"][0]["feature"]]
            )
            ax3d.set_ylabel(
                op["train"]["imp_all"][0].columns[term["terms"][1]["feature"]]
            )
            plt.show()


# In[ ]:


i = 13
colours = ["blue", "red"]
lines = []

j = gam.terms.info["terms"][i]["terms"][0]["feature"]

XX = gam.generate_X_grid(
    term=i, n=len(op["multi_cat_levels"]["Indication"]), meshgrid=True
)
Z, confi = gam.partial_dependence(term=i, X=XX, quantiles=cis, meshgrid=True)

fig, ax = plt.subplots(figsize=(9, 4))

for l, sli in enumerate([0, -1]):
    for k in range(n_cis):
        ax.fill_between(
            XX[0][:, 0],
            confi[:, sli, k],
            confi[:, sli, -(k + 1)],
            lw=0.0,
            alpha=1 / n_cis,
            color=colours[l],
        )
    lines.append(Line2D([0], [0], color=colours[l]))

ax.legend(lines, subplot_labels[i]["lines"], loc="upper left")
ax.set_title(subplot_labels[i]["title"])
ax.set_xticks(XX[0][:, j])
ax.set_xticklabels(
    op["multi_cat_levels"]["Indication"],
    rotation=45,
    rotation_mode="anchor",
    horizontalalignment="right",
    verticalalignment="top",
)
ax.set_xlim([XX[0][0, j], XX[0][-1, j]])
plt.grid(linewidth=0.5, linestyle=":")

plt.show()


# ## Evaluate performance

# In[ ]:


# EVAL_FOLD = 'train'
EVAL_FOLD = "test"  # only use for final evaluation!!


# In[ ]:


imp_samples = []

for i in pb(range(len(op[EVAL_FOLD]["imp_all"]))):
    imp_samples.append(
        quick_sample(
            gam,
            op[EVAL_FOLD]["imp_all"][i].values,
            RANDOM_SEED,
            quantity="mu",
            n_draws=50,
        )
    )

samples = np.vstack(imp_samples)
samples.shape


# Inspect some risk distributions for individual patients:

# In[ ]:


for i in range(10, 20):
    plt.hist(samples[:, i], alpha=0.3)
plt.xlim((-0.03, 1.03))
plt.show()


# Look at overall distribution of predicted risks:

# In[ ]:


hist_args = {"bins": 50, "density": True, "alpha": 0.5}

for i, outcome in enumerate(("lived", "died")):
    strat_samples = samples[:, np.where(op[EVAL_FOLD]["y"] == i)[0]].flatten()
    plt.hist(strat_samples, label=outcome, **hist_args)

plt.legend()
plt.show()


# In[ ]:


y_point_pred = np.median(samples, 0)


# In[ ]:


evaluate_predictions(op[EVAL_FOLD]["y"], y_point_pred)


# ## Figures for ASA abstract

# ### Illustrating distributions of predicted risks vs. point predictions

# In[ ]:


from arviz.plots import plot_kde


# In[ ]:


p95 = np.percentile(samples, (2.5, 97.5), 0).T
p95_range = p95[:, 1] - p95[:, 0]


# In[ ]:


n = 5 * 4
fig, ax = plt.subplots(5, 4, figsize=(10, 10))
ax = ax.ravel()

for i, j in enumerate(np.argpartition(p95_range, -n)[-n:]):
    ax[i].set(xlim=(0, 1), title=j)
    ax[i].hist(samples[:, j], bins=30)

plt.tight_layout()
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(9, 5))
plt.rcParams.update({"font.size": 16})

hist_args = {"bins": 50, "range": (0, 1), "density": True, "alpha": 1.0}
axvline_args = {"color": "black", "ls": "--"}

for i, j in enumerate((13672, 1)):
    ax.hist(samples[:, j], label=f"Patient {i + 1}", **hist_args)
    if i:
        ax.axvline(np.median(samples[:, j]), label="Point prediction", **axvline_args)
    else:
        ax.axvline(np.median(samples[:, j]), **axvline_args)


ax.set(xlim=(0, 1), xlabel="Predicted mortality risk", ylabel="Probability density")
ax.legend()
plt.show()


# ### Partial dependence plots

# In[ ]:


@dataclass
class GAMTerm:
    name: str  # Tuple[str, str] for 3d plots
    pretty_name: str  # Tuple[str, str] for 3d plots
    gs_pos: Tuple[int, int]
    labels: List = None  # Tuple[List, List] for 3d plots
    strata: List[str] = None
    legend_loc: str = None
    view_3d: Tuple[int, int] = None


def sanitize_ind(ind: str):
    inds = ind.split("+")
    for i in range(len(inds)):
        inds[i] = inds[i][7:]
        inds[i] = " ".join(re.findall("[A-Z][^A-Z]*", inds[i]))
    inds = (" + ".join(inds)).lower()
    return inds[0].upper() + inds[1:]


# In[ ]:


aes = [
    GAMTerm("S01AgeOnArrival", "Age (years)", (0, 0)),
    GAMTerm("S03Sodium", "Sodium (mmol/L)", (1, 1)),
    GAMTerm("S03Potassium", "Potassium (mmol/L)", (1, 2)),
    GAMTerm("S03WhiteCellCount", r"White cell count ($\times$10${^9}$/L)", (1, 0)),
    GAMTerm("S03SystolicBloodPressure", "Systolic pressure (mmHg)", (0, 1)),
    GAMTerm(
        "S03GlasgowComaScore",
        "Glasgow Coma Score",
        (2, 2),
        None
        #             list(range(3, 16))
    ),
    GAMTerm("S03PreOpArterialBloodLactate", "Lactate (mmol/L)", (2, 0)),
    GAMTerm("S03PreOpLowestAlbumin", "Albumin (g/L)", (2, 1)),
    GAMTerm("S03ASAScore", "ASA physical status", (3, 0), list(range(1, 6))),
    GAMTerm(
        "S03PreOpArterialBloodLactate_missing", "Lactate missing", (3, 1), ["No", "Yes"]
    ),
    GAMTerm("S03PreOpLowestAlbumin_missing", "Albumin missing", (3, 2), ["No", "Yes"]),
    GAMTerm(
        "S03DiagnosedMalignancy",
        "Malignancy",
        (4, 1),
        ["None", "Primary\nonly", "Nodal\nmets.", "Distant\nmets."],
        ["No CT", "CT"],
        "upper left",
    ),
    GAMTerm(
        "S03Pred_Peritsoil",
        "Peritoneal soiling",
        (4, 0),
        ["None", "Serous", "Local\npus", "Free pus /\nblood / faeces"],
        ["No CT", "CT"],
        "upper left",
    ),
    GAMTerm(
        "Indication",
        "Indication",
        (slice(5, 7), slice(0, 2)),
        [sanitize_ind(s) for s in op["multi_cat_levels"]["Indication"]],
        ["No CT", "CT"],
        "upper left",
    ),
    GAMTerm(
        "S03Pulse",
        "Heart rate (BPM)",
        (0, 2),
        None,
        ["Sinus", "Arrhythmia"],
        "lower right",
    ),
    GAMTerm(
        ("S03CardiacSigns", "S03RespiratorySigns"),
        ("Cardiovascular", "Respiratory"),
        (5, 2),
        (None, None),
        None,
        None,
        (45, 205),
    ),
    GAMTerm(
        ("S03SerumCreatinine", "S03Urea"),
        ("Creatinine (mmol/L)", "Urea (mmol/L)"),
        (4, 2),
        (None, None),
        None,
        None,
        (45, 115),
    ),
]

cardio_resp_key = (
    (
        "Cardiovascular: 0 = No failure,\n"
        + "1 = CVS drugs, 2 = Oedema / warfarin,\n"
        + "3 = Cardiomegaly / raised JVP\n\n"
        + "Respiratory: 0 = No SOB, 1 = Mild SOB / COPD,\n"
        + "2 = Moderate SOB / COPD,\n"
        + "3 = Fibrosis / consolidation / severe SOB"
    ),
    (5, 2),
)


# In[ ]:


n_cols = 3
row_height = 3.0
ticks_per_cat = 21
scaler = 1.2
plt.rcParams.update({"font.size": 12})

terms = gam.terms.info["terms"][:-1]
mid_cat_i = int((ticks_per_cat - 1) / 2)

n_rows = int(np.ceil((len(terms)) / n_cols))
fig = plt.figure(figsize=(scaler * 12, scaler * row_height * n_rows))
gs = fig.add_gridspec(n_rows, n_cols)

for i, term in enumerate(terms):
    if aes[i].view_3d is None:
        ax = fig.add_subplot(gs[aes[i].gs_pos])
        ax.set_title(aes[i].pretty_name)
    else:
        ax = fig.add_subplot(gs[aes[i].gs_pos], projection="3d")

    if aes[i].labels is None:
        n = 100
    else:
        n = len(aes[i].labels) * ticks_per_cat

    if term["term_type"] != "tensor_term":
        XX = gam.generate_X_grid(term=i, n=n)
        pdep, confi = gam.partial_dependence(term=i, X=XX, quantiles=cis)

        j = term["feature"]

        for k in range(n_cis):
            ax.fill_between(
                XX[:, j],
                confi[:, k],
                confi[:, -(k + 1)],
                alpha=1 / n_cis,
                color="tab:blue",
                lw=0.0,
            )

        if aes[i].labels is not None:
            ax.set_xticks(XX[:, j][range(mid_cat_i, n, ticks_per_cat)])
            ax.set_xticklabels(aes[i].labels)

    else:
        XX = gam.generate_X_grid(term=i, n=n, meshgrid=True)
        Z, confi = gam.partial_dependence(term=i, X=XX, quantiles=cis, meshgrid=True)

        if aes[i].view_3d is None:
            colours = ["tab:blue", "tab:orange"]
            lines = []

            for l, sli in enumerate([0, -1]):
                for k in range(n_cis):
                    ax.fill_between(
                        XX[0][:, 0],
                        confi[:, sli, k],
                        confi[:, sli, -(k + 1)],
                        lw=0.0,
                        alpha=1 / n_cis,
                        color=colours[l],
                    )

                lines.append(Line2D([0], [0], color=colours[l]))
            ax.legend(lines, aes[i].strata, loc=aes[i].legend_loc)

            if aes[i].labels is not None:
                ax.set_xticks(XX[0][:, 0][range(mid_cat_i, n, ticks_per_cat)])
                ax.set_xticklabels(aes[i].labels)
                ax.set_xlim([XX[0][0, 0], XX[0][-1, 0]])
                if aes[i].name == "Indication":
                    ax.set_xticklabels(
                        aes[i].labels,
                        rotation=45,
                        rotation_mode="anchor",
                        horizontalalignment="right",
                        verticalalignment="top",
                    )

        else:
            ax.plot_surface(XX[0], XX[1], Z, cmap="Blues")
            ax.view_init(*aes[i].view_3d)
            ax.set_xlabel(aes[i].pretty_name[0])
            ax.set_ylabel(aes[i].pretty_name[1])

# # Cardio / resp key text
# ax = fig.add_subplot(gs[cardio_resp_key[1]])
# ax.legend('lower_left', cardio_resp_key[0], bbox_to_anchor=(-2, 0))
# ax.set_axis_off()

plt.tight_layout()
plt.show()
