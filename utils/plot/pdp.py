from dataclasses import dataclass
from typing import Tuple, List, Union, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from pygam import GAM, LogisticGAM, LinearGAM
from sklearn.preprocessing import QuantileTransformer

from utils.constants import GAM_CONFIDENCE_INTERVALS
from utils.plot.helpers import generate_ci_quantiles
from utils.model.shared import LogOddsTransformer


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


class PDPFigure:
    """Plot partial dependence for each GAM feature. If transformer is not
        None, transforms PDP back into their original space e.g. transforms
        Gaussian back into lactate space. Optionally adds rug plots for
        univariate continuous features."""

    def __init__(self,
        gam: GAM,
        pdp_terms: List[PDPTerm],
        ylabel: str,
        transformer: Union[
            None, QuantileTransformer, LogOddsTransformer] = None,
        plot_just_outer_ci: bool = False,
        plot_hists: bool = False,
        hist_data: Union[None, pd.DataFrame] = None,
        max_hist_bins: int = 20,
        standardise_y_scale: bool = True,
        fig_width: float = 12.0,
        n_cols: int = 3,
        row_height: float = 3.0,
        ticks_per_cat: int = 21,
        strata_colours: Tuple[str] = ("tab:blue", "tab:orange"),
        confidence_intervals: Tuple = GAM_CONFIDENCE_INTERVALS
    ):
        self.gam = gam
        self.pdp_terms = pdp_terms
        self.ylabel = ylabel
        self.transformer = transformer
        self.plot_just_outer_ci = plot_just_outer_ci
        self.plot_hists = plot_hists
        self.hist_data = hist_data
        self.max_hist_bins = max_hist_bins
        self.standardise_y_scale = standardise_y_scale
        self.fig_width = fig_width
        self.n_cols = n_cols
        self.row_height = row_height
        self.ticks_per_cat = ticks_per_cat
        self.strata_colours = strata_colours
        self.n_rows = self._calculate_n_rows()
        self.trans_centre = self._calculate_transformation_centre()
        self.cis = generate_ci_quantiles(confidence_intervals)
        self.fig, self.gs = None, None
        self.y_min, self.y_max = {'2d': 0., '3d': 0.}, {'2d': 0., '3d': 0.}

    @property
    def terms(self) -> List[Dict]:
        """Removes intercept."""
        return self.gam.terms.info["terms"][:-1]

    @property
    def mid_cat_i(self) -> int:
        """Used to centre the x axis labels for each category."""
        return int((self.ticks_per_cat - 1) / 2)

    @property
    def n_cis(self) -> int:
        return len(self.cis)

    def _calculate_n_rows(self) -> int:
        n_rows = 0
        for pdp_term in self.pdp_terms:
            row_for_this_term = pdp_term.gs_pos[0]
            if isinstance(row_for_this_term, slice):
                row_for_this_term = row_for_this_term.stop - 2
            if n_rows < row_for_this_term + 1:
                n_rows = row_for_this_term + 1
        return n_rows

    def _calculate_transformation_centre(self) -> Union[None, float]:
        if self.transformer is None:
            return None
        else:
            return self.transformer.inverse_transform(
                np.zeros((1, 1))).flatten()[0]

    def _update_y_min_max(
        self,
        min_candidate: np.ndarray,
        max_candidate: np.ndarray,
        plot_type: str
    ):
        if min_candidate.min() < self.y_min[plot_type]:
            self.y_min[plot_type] = min_candidate.min()
        if max_candidate.max() > self.y_max[plot_type]:
            self.y_max[plot_type] = max_candidate.max()

    def plot(self) -> Tuple[Figure, None]:
        """Generate figure of partial dependence plots."""
        self._init_figure()
        for i, term in enumerate(self.terms):
            self._plot_single_pdp(i, term)
        self._modify_axes()
        self.fig.tight_layout()
        return self.fig, None

    def _init_figure(self):
        self.fig = plt.figure(
            figsize=(self.fig_width, self.row_height * self.n_rows))
        self.gs = self.fig.add_gridspec(self.n_rows, self.n_cols)

    def _plot_single_pdp(self, i: int, term: Dict):
        ax = self._init_ax(i)
        x_length = self._set_x_length(i)
        if term["term_type"] != "tensor_term":
            self._non_tensor_pdp(i, term, ax, x_length)
        else:
            self._tensor_pdp(i, ax, x_length)

    def _init_ax(self, i: int) -> Axes:
        if self.pdp_terms[i].view_3d is None:
            ax = self.fig.add_subplot(self.gs[self.pdp_terms[i].gs_pos])
            ax.set_title(self.pdp_terms[i].pretty_name)
        else:
            ax = self.fig.add_subplot(
                self.gs[self.pdp_terms[i].gs_pos], projection="3d")
        return ax

    def _set_x_length(self, i: int) -> int:
        if self.pdp_terms[i].labels is None:
            x_length = 100
        else:
            x_length = len(self.pdp_terms[i].labels) * self.ticks_per_cat
        return x_length

    def _non_tensor_pdp(self, i: int, term: Dict, ax: Axes, x_length: int):
        xx = self.gam.generate_X_grid(term=i, n=x_length)
        _, confi = self.gam.partial_dependence(
            term=i, X=xx, quantiles=self.cis)
        if self.transformer is not None:
            confi = self._inverse_transform(confi)
        self._update_y_min_max(confi[:, 0], confi[:, -1], plot_type='2d')
        if self.plot_just_outer_ci:
            ax.fill_between(
                xx[:, term["feature"]],
                confi[:, 0],
                confi[:, -1],
                alpha=0.5,
                color="tab:blue",
                lw=2.0)
        else:
            for k in range(self.n_cis):
                ax.fill_between(
                    xx[:, term["feature"]],
                    confi[:, k],
                    confi[:, -(k + 1)],
                    alpha=1 / self.n_cis,
                    color="tab:blue",
                    lw=0.0)
        self._set_non_tensor_x_labels(i, ax, xx, term["feature"], x_length)
        ax.set_xlim(xx[:, term["feature"]].min(), xx[:, term["feature"]].max())
        ax.set_ylabel(self.ylabel)

    def _set_non_tensor_x_labels(
        self,
        i: int,
        ax: Axes,
        xx: np.ndarray,
        feature_i: int,
        x_length: int
    ):
        if self.pdp_terms[i].labels is not None:
            ax.set_xticks(xx[:, feature_i][
                range(self.mid_cat_i, x_length, self.ticks_per_cat)])
            ax.set_xticklabels(self.pdp_terms[i].labels)

    def _tensor_pdp(self, i: int, ax: Axes, x_length: int):
        xx = self.gam.generate_X_grid(term=i, n=x_length, meshgrid=True)
        z, confi = self.gam.partial_dependence(
            term=i, X=xx, quantiles=self.cis, meshgrid=True)
        if self.pdp_terms[i].view_3d is None:
            self._binary_interaction_pdp(i, ax, x_length, xx, confi)
        else:
            self._3d_pdp(i, ax, xx, z)

    def _binary_interaction_pdp(
        self,
        i: int,
        ax: Axes,
        x_length: int,
        xx: Tuple[np.ndarray, np.ndarray],
        confi: np.ndarray
    ):
        lines = []
        for slice_i, sli in enumerate([0, -1]):
            if self.transformer is not None:
                confi[:, sli, :] = self._inverse_transform(confi[:, sli, :])
            self._update_y_min_max(
                confi[:, sli, 0], confi[:, sli, -1], plot_type='2d')
            if self.plot_just_outer_ci:
                ax.fill_between(
                    xx[0][:, 0],
                    confi[:, sli, 0],
                    confi[:, sli, -1],
                    lw=2.0,
                    alpha=0.5,
                    color=self.strata_colours[slice_i])
            else:
                for k in range(self.n_cis):
                    ax.fill_between(
                        xx[0][:, 0],
                        confi[:, sli, k],
                        confi[:, sli, -(k + 1)],
                        lw=0.0,
                        alpha=1 / self.n_cis,
                        color=self.strata_colours[slice_i])
            lines.append(Line2D([0], [0], color=self.strata_colours[slice_i]))
        ax.legend(lines, self.pdp_terms[i].strata,
                  loc=self.pdp_terms[i].legend_loc)
        ax.set_xlim(xx[0][0, 0], xx[0][-1, 0])
        self._set_tensor_x_labels(i, ax, xx, x_length)
        ax.set_ylabel(self.ylabel)

    def _set_tensor_x_labels(
        self,
        i: int,
        ax: Axes,
        xx: Tuple[np.ndarray, np.ndarray],
        x_length: int
    ):
        if self.pdp_terms[i].labels is not None:
            ax.set_xticks(xx[0][:, 0][
                              range(self.mid_cat_i, x_length,
                                    self.ticks_per_cat)])
            ax.set_xticklabels(self.pdp_terms[i].labels)
            if self.pdp_terms[i].name == "Indication":
                ax.set_xticklabels(
                    self.pdp_terms[i].labels,
                    rotation=45,
                    horizontalalignment="center",
                    verticalalignment="top",
                )

    def _3d_pdp(
        self,
        i: int,
        ax: Axes,
        xx: Tuple[np.ndarray, np.ndarray],
        z: np.ndarray
    ):
        if self.transformer is not None:
            z = self._inverse_transform(z)
        self._update_y_min_max(z, z, plot_type='3d')
        ax.plot_surface(xx[0], xx[1], z, cmap="Blues")
        ax.view_init(*self.pdp_terms[i].view_3d)
        ax.set_xlabel(self.pdp_terms[i].pretty_name[0])
        ax.set_ylabel(self.pdp_terms[i].pretty_name[1])
        ax.set_zlabel(self.ylabel)

    def _inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Converts probability to risk ration for logistic models."""
        inv_trans_x = self.transformer.inverse_transform(
            x.reshape(np.prod(x.shape), 1)
        ).reshape(x.shape)
        if isinstance(self.gam, LinearGAM):
            return inv_trans_x - self.trans_centre
        elif isinstance(self.gam, LogisticGAM):
            return inv_trans_x / self.trans_centre
        else:
            raise NotImplementedError

    def _modify_axes(self):
        """Loop back over axes, optionally standardising their y axis scale and
            adding rug plots. We have to do these operations in a second loop
            after the initial plotting as we only know the correct global y
            axis scale at this point, and if we are standardising the y axis
            then we need to know what its lower limit is in order to place the
            rug at the bottom of each plot. Also autoscales x limits."""
        for i, ax in enumerate(self.fig.axes):
            if self.pdp_terms[i].view_3d is None:
                if self.standardise_y_scale:
                    ax.set_ylim(self.y_min['2d'], self.y_max['2d'])
                if self.plot_hists:
                    self._plot_hist(i, ax)
            else:
                if self.standardise_y_scale:
                    ax.set_zlim3d(self.y_min['3d'], self.y_max['3d'])

    def _plot_hist(self, i: int, ax: Axes):
        hist, bins = np.histogram(
            self.hist_data[self.pdp_terms[i].name].values,
            bins=self._determine_n_hist_bins(i))
        if self.pdp_terms[i].labels is not None:
            x_ticks = ax.get_xticks()
            xlim = ax.get_xlim()
            width = (xlim[1] - xlim[0]) / len(x_ticks)
        else:
            x_ticks = (bins[:-1] + bins[1:]) / 2
            width = bins[1] - bins[0]
        ylim = ax.get_ylim()
        hist_height_frac = 0.9
        ax.bar(
            x=x_ticks,
            height=(hist / hist.max()) * (ylim[1] - ylim[0]) * hist_height_frac,
            align='center',
            width=width,
            bottom=ylim[0],
            color='black',
            alpha=0.15)

    def _determine_n_hist_bins(self, i: int):
        if self.pdp_terms[i].name == "S03GlasgowComaScore":
            return 13
        elif self.pdp_terms[i].labels is not None:
            return len(self.pdp_terms[i].labels)
        else:
            return self.max_hist_bins


def compare_pdps_from_different_gams_plot(
    gams: Tuple[LogisticGAM, LogisticGAM],
    gam_names: Tuple[str, str],
    term_indices: Tuple[int, int],
    term_names: Tuple[str, str],
    column_indices: Tuple[int, int],
    transformer: Union[None, LogOddsTransformer] = None,
    figsize: Tuple[int, int] = (8, 3),
    legend_locs: Tuple[str, str] = ("upper left", "upper right"),
    gam_colours: Tuple[str] = ("tab:blue", "tab:orange"),
    confidence_intervals: Tuple = GAM_CONFIDENCE_INTERVALS
) -> (Figure, Axes):
    """Only works with LogisticGAM currently. term_indices are the indices
        of the terms of interest in the GAM specification. column_indices are
        the indices of the columns for these terms in the GAM input data."""
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax = ax.ravel()
    y_lim = [np.inf, -np.inf]
    cis = generate_ci_quantiles(confidence_intervals)
    n_cis = len(cis)

    for ax_i, term_i in enumerate(term_indices):
        legend_lines = []

        for gam_i, gam in enumerate(gams):
            xx = gam.generate_X_grid(term=term_i, n=100)
            _, confi = gam.partial_dependence(term=term_i, X=xx, quantiles=cis)

            if transformer:
                ylabel = 'Relative mortality risk'
                trans_centre = transformer.inverse_transform(
                    np.zeros((1, 1))
                ).flatten()[0]
                confi = transformer.inverse_transform(confi) / trans_centre
            else:
                ylabel = 'Log-odds of mortality'

            if confi.min() < y_lim[0]:
                y_lim[0] = confi.min()
            if confi.max() > y_lim[1]:
                y_lim[1] = confi.max()

            for k in range(n_cis):
                ax[ax_i].fill_between(
                    xx[:, column_indices[ax_i]],
                    confi[:, k],
                    confi[:, -(k + 1)],
                    alpha=1 / n_cis,
                    color=gam_colours[gam_i],
                    lw=0.0)

            legend_lines.append(Line2D([0], [0], color=gam_colours[gam_i]))

        ax[ax_i].legend(legend_lines, gam_names, loc=legend_locs[ax_i])
        ax[ax_i].set_xlim(
            xx[:, column_indices[ax_i]].min(),
            xx[:, column_indices[ax_i]].max())
        ax[ax_i].set_title(term_names[ax_i])
        ax[ax_i].set_ylabel(ylabel)

    for ax_i, _ in enumerate(ax):
        ax[ax_i].set_ylim(y_lim)

    fig.tight_layout()
    return fig, ax
