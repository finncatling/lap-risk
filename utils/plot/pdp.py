from typing import Tuple, List, Union, Dict, Iterable

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from pygam import GAM
from sklearn.preprocessing import QuantileTransformer

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


class PDPFigure:
    """Plot partial dependence for each GAM feature. If transformer is not
        None, transforms PDP back into their original space e.g. transforms
        Gaussian back into lactate space."""

    def __init__(
        self,
        gam: GAM,
        pdp_terms: List[PDPTerm],
        transformer: Union[None, QuantileTransformer] = None,
        fig_width: float = 12.0,
        n_cols: int = 3,
        row_height: float = 3.0,
        ticks_per_cat: int = 21,
        strata_colours: Tuple[str] = ("tab:blue", "tab:orange"),
        confidence_intervals: Tuple = GAM_CONFIDENCE_INTERVALS
    ):
        self.gam = gam
        self.pdp_terms = pdp_terms
        self.transformer = transformer
        self.fig_width = fig_width
        self.n_cols = n_cols
        self.row_height = row_height
        self.ticks_per_cat = ticks_per_cat
        self.strata_colours = strata_colours
        self.n_rows = self._calculate_n_rows()
        self.trans_centre = self._calculate_transformation_centre()
        self.cis = generate_ci_quantiles(confidence_intervals)
        self.fig, self.gs = None, None

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

    def plot(self) -> Tuple[Figure, None]:
        """Generate figure of partial dependence plots."""
        self._init_figure()
        for i, term in enumerate(self.terms):
            self._plot_single_pdp(i, term)
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
        pdep, confi = self.gam.partial_dependence(
            term=i, X=xx, quantiles=self.cis)
        for k in range(self.n_cis):
            ax.fill_between(
                xx[:, term["feature"]],
                confi[:, k],
                confi[:, -(k + 1)],
                alpha=1 / self.n_cis,
                color="tab:blue",
                lw=0.0,
            )
        self._set_non_tensor_x_labels(i, ax, xx, term["feature"], x_length)

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
            for k in range(self.n_cis):
                ax.fill_between(
                    xx[0][:, 0],
                    confi[:, sli, k],
                    confi[:, sli, -(k + 1)],
                    lw=0.0,
                    alpha=1 / self.n_cis,
                    color=self.strata_colours[slice_i],
                )
            lines.append(Line2D([0], [0], color=self.strata_colours[slice_i]))
        ax.legend(lines, self.pdp_terms[i].strata,
                  loc=self.pdp_terms[i].legend_loc)
        self._set_tensor_x_labels(i, ax, xx, x_length)

    def _set_tensor_x_labels(
        self,
        i: int,
        ax: Axes,
        xx: Tuple[np.ndarray, np.ndarray],
        x_length: int
    ):
        if self.pdp_terms[i].labels is not None:
            ax.set_xticks(xx[0][:, 0][
                range(self.mid_cat_i, x_length, self.ticks_per_cat)])
            ax.set_xticklabels(self.pdp_terms[i].labels)
            ax.set_xlim([xx[0][0, 0], xx[0][-1, 0]])
            if self.pdp_terms[i].name == "Indication":
                ax.set_xticklabels(
                    self.pdp_terms[i].labels,
                    rotation=45,
                    horizontalalignment="centre",
                    verticalalignment="top",
                )

    def _3d_pdp(
        self,
        i: int,
        ax: Axes,
        xx: Tuple[np.ndarray, np.ndarray],
        z: np.ndarray
    ):
        ax.plot_surface(xx[0], xx[1], z, cmap="Blues")
        ax.view_init(*self.pdp_terms[i].view_3d)
        ax.set_xlabel(self.pdp_terms[i].pretty_name[0])
        ax.set_ylabel(self.pdp_terms[i].pretty_name[1])

    def _inverse_transform_pdp(self, pdp: np.ndarray) -> np.ndarray:
        if self.transformer is None:
            return pdp
        else:
            raise NotImplementedError
