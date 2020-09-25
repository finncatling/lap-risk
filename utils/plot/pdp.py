from typing import Tuple, List, Union

import matplotlib.pyplot as plt
from dataclasses import dataclass
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from pygam import GAM

from utils.constants import GAM_CONFIDENCE_INTERVALS
from utils.plot.helpers import generate_ci_quantiles


@dataclass
class PDPTerm:
    name: Union[str, Tuple[str, str]]
    pretty_name: Union[str, Tuple[str, str]]
    gs_pos: Union[Tuple[int, int], Tuple[slice, slice]]
    labels: Union[
        None, List[str], Tuple[None, None], Tuple[List[str], List[str]]
    ] = None
    strata: Union[None, List[str]] = None
    legend_loc: Union[None, str] = None
    view_3d: Union[None, Tuple[int, int]] = None


def plot_partial_dependence(
    gam: GAM,
    pdp_terms: List[PDPTerm],
    n_cols: int = 3,
    row_height: float = 3.0,
    ticks_per_cat: int = 21,
    confidence_intervals: Tuple = GAM_CONFIDENCE_INTERVALS,
) -> Tuple[Figure, Axes]:
    """Plot partial dependence for each GAM feature."""
    terms = gam.terms.info["terms"][:-1]
    mid_cat_i = int((ticks_per_cat - 1) / 2)

    # calculate number of rows needed
    n_rows = 0
    for pdp_term in pdp_terms:
        row_for_this_term = pdp_term.gs_pos[0]
        if isinstance(row_for_this_term, slice):
            row_for_this_term = row_for_this_term.stop - 2
        if n_rows < row_for_this_term + 1:
            n_rows = row_for_this_term + 1

    fig = plt.figure(figsize=(12, row_height * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols)

    cis = generate_ci_quantiles(confidence_intervals)
    n_cis = len(cis)

    for i, term in enumerate(terms):
        if pdp_terms[i].view_3d is None:
            ax = fig.add_subplot(gs[pdp_terms[i].gs_pos])
            ax.set_title(pdp_terms[i].pretty_name)
        else:
            ax = fig.add_subplot(gs[pdp_terms[i].gs_pos], projection="3d")

        if pdp_terms[i].labels is None:
            n = 100
        else:
            n = len(pdp_terms[i].labels) * ticks_per_cat

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

            if pdp_terms[i].labels is not None:
                ax.set_xticks(XX[:, j][range(mid_cat_i, n, ticks_per_cat)])
                ax.set_xticklabels(pdp_terms[i].labels)

        else:
            XX = gam.generate_X_grid(term=i, n=n, meshgrid=True)
            Z, confi = gam.partial_dependence(
                term=i, X=XX, quantiles=cis, meshgrid=True
            )

            if pdp_terms[i].view_3d is None:
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
                ax.legend(lines, pdp_terms[i].strata,
                          loc=pdp_terms[i].legend_loc)

                if pdp_terms[i].labels is not None:
                    ax.set_xticks(
                        XX[0][:, 0][range(mid_cat_i, n, ticks_per_cat)])
                    ax.set_xticklabels(pdp_terms[i].labels)
                    ax.set_xlim([XX[0][0, 0], XX[0][-1, 0]])
                    if pdp_terms[i].name == "Indication":
                        ax.set_xticklabels(
                            pdp_terms[i].labels,
                            rotation=45,
                            rotation_mode="anchor",
                            horizontalalignment="right",
                            verticalalignment="top",
                        )

            else:
                ax.plot_surface(XX[0], XX[1], Z, cmap="Blues")
                ax.view_init(*pdp_terms[i].view_3d)
                ax.set_xlabel(pdp_terms[i].pretty_name[0])
                ax.set_ylabel(pdp_terms[i].pretty_name[1])

    fig.tight_layout()
    return fig, ax
