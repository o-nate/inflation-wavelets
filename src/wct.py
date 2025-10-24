"""Wavelet coherence transformation"""

from __future__ import division
from dataclasses import dataclass, field
import logging
from pathlib import Path
import sys
from typing import List, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

import pycwt as wavelet

# Ensure project root is on sys.path so top-level packages like `constants` resolve
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from constants import ids, results_configs

from src import retrieve_data
from src.utils import helpers, wavelet_helpers
from src.utils.logging_helpers import define_other_module_log_level

# * Logging settings
logger = logging.getLogger(__name__)
define_other_module_log_level("warning")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# * Constants
DT = 1 / 12  # Delta t
DJ = 1 / 8  # Delta j
S0 = 2 * DT  # Initial scale
MOTHER = "morlet"  # Morlet wavelet with :math:`\omega_0=6`.
MOTHER_DICT = {
    "morlet": wavelet.Morlet(6),
    "paul": wavelet.Paul(),
    "DOG": wavelet.DOG(),
    "mexicanhat": wavelet.MexicanHat(),
}
LEVELS = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
# Coherence magnitude levels (bounded in [0, 1]) for plotting
WCT_LEVELS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]


WCT_PLOT_PROPS = {
    "cmap": "jet",
    "sig_colors": "k",
    "sig_linewidths": 2,
    "coi_color": "k",
    "coi_alpha": 0.3,
    "coi_hatch": "--",
    "phase_diff_units": "width",
    "phase_diff_angles": "uv",
    "phase_diff_pivot": "mid",
    "phase_diff_linewidth": 0.5,
    "phase_diff_edgecolor": "k",
    "phase_diff_alpha": 0.7,
}


@dataclass
class DataForWCT:
    """Holds data for WCT"""

    t_values: npt.NDArray = field(init=False)
    y1_values: npt.NDArray
    y2_values: npt.NDArray
    mother_wavelet: Type
    delta_t: float
    delta_j: float
    initial_scale: float
    levels: List[float]
    actual_times: npt.NDArray = None  # Optional actual time values

    def __post_init__(self):
        if self.actual_times is not None:
            self.t_values = self.actual_times
        else:
            self.t_values = np.linspace(1, self.y1_values.size + 1, self.y1_values.size)


@dataclass
class ResultsFromWCT:
    """Holds results from Wavelet Coherence Transform"""

    coherence: npt.NDArray
    period: npt.NDArray
    significance_levels: npt.NDArray
    coi: npt.NDArray
    phase_diff_u: npt.NDArray
    phase_diff_v: npt.NDArray


def run_wct(
    wavelet_coherence_transform: Type[DataForWCT],
    significance_level: float = 0.95,
) -> Type[ResultsFromWCT]:
    """Conduct Wavelet Coherence Transformation on two series.
    Returns coherence magnitude, period, significance levels, cone of influence,
    and phase"""

    # * Perform wavelet coherence transform
    coherence, phase, coi, freqs, signif = wavelet.wct(
        wavelet_coherence_transform.y1_values,
        wavelet_coherence_transform.y2_values,
        wavelet_coherence_transform.delta_t,
        dj=wavelet_coherence_transform.delta_j,
        s0=wavelet_coherence_transform.initial_scale,
        J=-1,
        significance_level=significance_level,
        wavelet=wavelet_coherence_transform.mother_wavelet,
        normalize=True,
        cache=True,
    )

    period = 1 / freqs

    # Significance levels array shaped to time axis
    signal_size = wavelet_coherence_transform.y1_values.size
    sig95 = np.ones([1, signal_size]) * signif[:, None]
    sig95 = np.abs(coherence) / sig95  # significant where ratio > 1

    # # Build COI polygon in log2 space for plotting helper (like XWT path)
    # coi_plot = np.concatenate(
    #     [
    #         np.log2(coi),
    #         [np.log2(period[-1:])],
    #         np.log2(period[-1:]),
    #         [np.log2(period[-1:])],
    #     ]
    # )

    # * Calculate phase difference
    phase_diff_u, phase_diff_v = calculate_phase_difference(phase)

    return ResultsFromWCT(coherence, period, sig95, coi, phase_diff_u, phase_diff_v)


def calculate_phase_difference(
    wct_phase: npt.NDArray,
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Calculates the phase between both time series.
    Returns u and v phase difference vectors
    Description:
    These phase arrows follow the convention of Torrence and Webster (1999).
    In-phase points north, and anti-phase point south. If y1 leads y2, the arrows
    point east, and if y2 leads y1, the arrows point west."""

    angle = 0.5 * np.pi - wct_phase
    u_component, v_component = np.cos(angle), np.sin(angle)
    logger.debug(
        "Comparing length of phase arrays: %s, %s", len(u_component), len(v_component)
    )
    return u_component, v_component


def plot_wct(
    wct_ax: plt.Axes,
    wct_data: Type[DataForWCT],
    wct_results: Type[ResultsFromWCT],
    include_significance: bool = True,
    include_cone_of_influence: bool = True,
    include_phase_difference: bool = True,
    **kwargs,
) -> None:
    """Plot Wavelet Coherence of two series
    **kwargs**
    For coherence image: `cmap="jet"`
    For significance levels: `sig_colors="k"`, `sig_linewidths=2`
    For cone of influence: `coi_color="k"`, `coi_alpha=0.3`, `coi_hatch="--"`
    For phase difference: `phase_diff_units="width"`, `phase_diff_angles="uv"`,
    `phase_diff_pivot="mid"`,`phase_diff_linewidth=0.5`, `phase_diff_edgecolor="k"`,
    `phase_diff_alpha=0.7`"""
    extent = [
        min(wct_data.t_values),
        max(wct_data.t_values),
        min(wct_results.coi),
        max(wct_results.period),
    ]

    # * Coherence magnitude image
    wct_ax.contourf(
        wct_data.t_values,
        np.log2(wct_results.period),
        np.abs(wct_results.coherence),
        WCT_LEVELS,
        extend="both",
        cmap=kwargs["cmap"],
        extent=extent,
    )

    # * Add WCT features
    if include_significance:
        wavelet_helpers.plot_signficance_levels(
            wct_ax,
            wct_results.significance_levels,
            wct_data.t_values,
            wct_results.period,
            **kwargs,
        )
    if include_cone_of_influence:
        wavelet_helpers.plot_cone_of_influence(
            wct_ax,
            wct_results.coi,
            wct_data.t_values,
            wct_data.levels,
            wct_results.period,
            wct_data.delta_t,
            tranform_type="cwt",
            **kwargs,
        )
    if include_phase_difference:
        plot_phase_difference(
            wct_ax,
            wct_data.t_values,
            wct_results.period,
            wct_results.phase_diff_u,
            wct_results.phase_diff_v,
            **kwargs,
        )


def plot_phase_difference(
    wct_ax: plt.Axes,
    t_values: npt.NDArray,
    period: npt.NDArray,
    phase_diff_u: npt.NDArray,
    phase_diff_v: npt.NDArray,
    **kwargs,
) -> None:
    """Plot phase difference indicator vector arrows
    **kwargs**
    `phase_diff_units=`: "width"
    `phase_diff_angles=`: "uv"
    `phase_diff_pivot=`: "mid"
    `phase_diff_linewidth=`: 0.5
    `phase_diff_edgecolor=`: "k"
    `phase_diff_alpha=`: 0.7"""
    num_scales, num_times = phase_diff_u.shape
    desired_num_arrows_x = 48
    desired_num_arrows_y = 12
    x_step = max(1, num_times // desired_num_arrows_x)
    y_step = max(1, num_scales // desired_num_arrows_y)

    X = t_values[::x_step]
    Y = np.log2(period[::y_step])
    U = phase_diff_u[::y_step, ::x_step]
    V = phase_diff_v[::y_step, ::x_step]

    wct_ax.quiver(
        X,
        Y,
        U,
        V,
        units=kwargs["phase_diff_units"],
        angles=kwargs["phase_diff_angles"],
        pivot=kwargs["phase_diff_pivot"],
        linewidth=kwargs["phase_diff_linewidth"],
        edgecolor=kwargs["phase_diff_edgecolor"],
        alpha=kwargs["phase_diff_alpha"],
    )


def main() -> None:
    """Run script"""

    series_comparisons = [
        (ids.DIFF_LOG_CPI, ids.EXPECTATIONS),
        (ids.EXPECTATIONS, ids.NONDURABLES_CHG),
        (ids.EXPECTATIONS, ids.DURABLES_CHG),
        (ids.EXPECTATIONS, ids.SAVINGS_CHG),
        (ids.DIFF_LOG_CPI, ids.DIFF_LOG_EXPECTATIONS),
        (ids.DIFF_LOG_CPI, ids.DIFF_LOG_REAL_NONDURABLES),
        (ids.DIFF_LOG_CPI, ids.DIFF_LOG_REAL_DURABLES),
        (ids.DIFF_LOG_CPI, ids.DIFF_LOG_REAL_SAVINGS),
        (ids.EXPECTATIONS, ids.DIFF_LOG_NONDURABLES),
        (ids.EXPECTATIONS, ids.DIFF_LOG_DURABLES),
        (ids.EXPECTATIONS, ids.DIFF_LOG_SAVINGS),
        (ids.EXPECTATIONS, ids.DIFF_LOG_REAL_NONDURABLES),
        (ids.EXPECTATIONS, ids.DIFF_LOG_REAL_DURABLES),
        (ids.EXPECTATIONS, ids.DIFF_LOG_REAL_SAVINGS),
        (ids.DIFF_LOG_EXPECTATIONS, ids.DIFF_LOG_NONDURABLES),
        (ids.DIFF_LOG_EXPECTATIONS, ids.DIFF_LOG_DURABLES),
        (ids.DIFF_LOG_EXPECTATIONS, ids.DIFF_LOG_SAVINGS),
        (ids.DIFF_LOG_EXPECTATIONS, ids.DIFF_LOG_REAL_NONDURABLES),
        (ids.DIFF_LOG_EXPECTATIONS, ids.DIFF_LOG_REAL_DURABLES),
        (ids.DIFF_LOG_EXPECTATIONS, ids.DIFF_LOG_REAL_SAVINGS),
    ]

    series_titles = {
        ids.INFLATION: "CPI inflation",
        ids.EXPECTATIONS: "Inflation expectations",
        ids.SAVINGS_RATE: "Savings rate",
        ids.NONDURABLES_CHG: "Nondurables consumption (% chg)",
        ids.DURABLES_CHG: "Durables consumption (% chg)",
        ids.SAVINGS_CHG: "Savings (% chg)",
        ids.DIFF_LOG_CPI: "CPI inflation (diff in log)",
        ids.DIFF_LOG_EXPECTATIONS: "Inflation expectations (diff in log)",
        ids.DIFF_LOG_NONDURABLES: "Nondurables consumption (diff in log)",
        ids.DIFF_LOG_DURABLES: "Durables consumption (diff in log)",
        ids.DIFF_LOG_SAVINGS: "Savings (diff in log)",
        ids.DIFF_LOG_REAL_NONDURABLES: "Real nondurables consumption (diff in log)",
        ids.DIFF_LOG_REAL_DURABLES: "Real durables consumption (diff in log)",
        ids.DIFF_LOG_REAL_SAVINGS: "Real savings (diff in log)",
    }

    # US data
    raw_data = retrieve_data.get_fed_data(
        ids.US_CPI,
    )
    cpi, _, _ = retrieve_data.clean_fed_data(raw_data)
    cpi.rename(columns={"value": ids.CPI}, inplace=True)

    raw_data = retrieve_data.get_fed_data(
        ids.US_CPI,
        units="pc1",
        freq="m",
    )
    measured_inf, _, _ = retrieve_data.clean_fed_data(raw_data)
    measured_inf.rename(columns={"value": ids.INFLATION}, inplace=True)

    raw_data = retrieve_data.get_fed_data(
        ids.US_INF_EXPECTATIONS,
    )
    inf_exp, _, _ = retrieve_data.clean_fed_data(raw_data)
    inf_exp.rename(columns={"value": ids.EXPECTATIONS}, inplace=True)

    raw_data = retrieve_data.get_fed_data(
        ids.US_NONDURABLES_CONSUMPTION,
    )
    nondur_consump, _, _ = retrieve_data.clean_fed_data(raw_data)
    nondur_consump.rename(columns={"value": ids.NONDURABLES}, inplace=True)

    raw_data = retrieve_data.get_fed_data(
        ids.US_DURABLES_CONSUMPTION,
    )
    dur_consump, _, _ = retrieve_data.clean_fed_data(raw_data)
    dur_consump.rename(columns={"value": ids.DURABLES}, inplace=True)

    raw_data = retrieve_data.get_fed_data(
        ids.US_NONDURABLES_CONSUMPTION,
        units="pc1",
        freq="m",
    )
    nondur_consump_chg, _, _ = retrieve_data.clean_fed_data(raw_data)
    nondur_consump_chg.rename(columns={"value": ids.NONDURABLES_CHG}, inplace=True)

    raw_data = retrieve_data.get_fed_data(
        ids.US_DURABLES_CONSUMPTION,
        units="pc1",
        freq="m",
    )
    dur_consump_chg, _, _ = retrieve_data.clean_fed_data(raw_data)
    dur_consump_chg.rename(columns={"value": ids.DURABLES_CHG}, inplace=True)

    raw_data = retrieve_data.get_fed_data(
        ids.US_SAVINGS,
    )
    save, _, _ = retrieve_data.clean_fed_data(raw_data)
    save.rename(columns={"value": ids.SAVINGS}, inplace=True)

    raw_data = retrieve_data.get_fed_data(
        ids.US_SAVINGS,
        units="pc1",
        freq="m",
    )
    save_chg, _, _ = retrieve_data.clean_fed_data(raw_data)
    save_chg.rename(columns={"value": ids.SAVINGS_CHG}, inplace=True)

    raw_data = retrieve_data.get_fed_data(
        ids.US_SAVINGS_RATE,
    )
    save_rate, _, _ = retrieve_data.clean_fed_data(raw_data)
    save_rate.rename(columns={"value": ids.SAVINGS_RATE}, inplace=True)

    dataframes = [
        cpi,
        measured_inf,
        inf_exp,
        nondur_consump,
        nondur_consump_chg,
        dur_consump,
        dur_consump_chg,
        save,
        save_chg,
        save_rate,
    ]
    us_data = helpers.combine_series(dataframes, on=[ids.DATE], how="left")
    us_data.dropna(inplace=True)

    logger.info(
        "Using constant dollars from %s, CPI: %s",
        results_configs.CONSTANT_DOLLAR_DATE,
        us_data[
            us_data[ids.DATE] == pd.Timestamp(results_configs.CONSTANT_DOLLAR_DATE)
        ][ids.CPI].iat[0],
    )
    us_data = helpers.add_real_value_columns(
        data=us_data,
        nominal_columns=[ids.NONDURABLES, ids.DURABLES, ids.SAVINGS],
        cpi_column=ids.CPI,
        constant_date=results_configs.CONSTANT_DOLLAR_DATE,
    )
    us_data = helpers.calculate_diff_in_log(
        data=us_data,
        columns=[
            ids.CPI,
            ids.EXPECTATIONS,
            ids.NONDURABLES,
            ids.DURABLES,
            ids.SAVINGS,
            ids.REAL_NONDURABLES,
            ids.REAL_DURABLES,
            ids.REAL_SAVINGS,
        ],
    )

    measured_inf = helpers.calculate_diff_in_log(
        data=measured_inf,
        columns=[ids.INFLATION],
    )
    logger.debug("df shape %s", measured_inf.shape)

    for comp in series_comparisons[14:15]:
        logger.debug(comp)

        data = us_data.copy()
        logger.debug("NaNs: %s", data.isna().sum())
        data = data.dropna(subset=[comp[0], comp[1]])
        logger.debug(data.shape)

        y1 = data[comp[0]].to_numpy()
        y2 = data[comp[1]].to_numpy()
        y1 = wavelet_helpers.standardize_series(y1, detrend=False, remove_mean=True)
        y2 = wavelet_helpers.standardize_series(y2, detrend=True, remove_mean=False)

        mother_wct = MOTHER_DICT[MOTHER]

        wct_data = DataForWCT(
            y1_values=y1,
            y2_values=y2,
            mother_wavelet=mother_wct,
            delta_t=DT,
            delta_j=DJ,
            initial_scale=S0,
            levels=LEVELS,
        )

        results_from_wct = run_wct(wct_data)

        _, ax = plt.subplots(1, 1, figsize=(10, 8), sharex=True)

        plot_wct(
            ax,
            wct_data,
            results_from_wct,
            include_significance=True,
            include_cone_of_influence=True,
            include_phase_difference=True,
            **WCT_PLOT_PROPS,
        )

        ax.set_ylim(ax.get_ylim()[::-1])

        y_ticks = 2 ** np.arange(
            np.ceil(np.log2(results_from_wct.period.min())),
            np.ceil(np.log2(results_from_wct.period.max())),
        )
        ax.set_yticks(np.log2(y_ticks))
        ax.set_yticklabels(y_ticks, size=12)

        dates = us_data["date"].to_list()
        x_dates = [dates[0]] + [dates[i + 99] for i in range(0, 500, 100)]
        x_ticks = [str(date.year) for date in x_dates]
        x_tick_positions = [i for i in range(0, 600, 100)]
        logger.debug("dates %s", x_ticks)
        logger.debug("dates %s", x_tick_positions)
        logger.debug("dates len %s", len(x_ticks))
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels(x_ticks, size=12)

        ax.set_title(f"{series_titles[comp[0]]} X {series_titles[comp[1]]}", size=16)
        ax.set_ylabel("Period (years)", size=14)

        parent_dir = Path(__file__).parents[1]
        export_file = parent_dir / "results" / f"wct_module_{comp[0]}_{comp[1]}.png"
        plt.savefig(export_file, bbox_inches="tight")


if __name__ == "__main__":
    main()
