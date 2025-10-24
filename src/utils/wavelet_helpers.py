"""Helper functions for wavelet transforms"""

import logging
import sys
from typing import List

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from src.utils.logging_helpers import define_other_module_log_level

# * Logging settings
logger = logging.getLogger(__name__)
define_other_module_log_level("Error")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def align_series(t_values: npt.NDArray, series_vlaues: npt.NDArray) -> npt.NDArray:
    """Aligns series lengths when they are not equal by removing the first value"""
    if len(series_vlaues) != len(t_values):
        logger.warning("Trimming series signal")
        difference = np.abs(len(series_vlaues) - len(t_values))
        return series_vlaues[difference:]


def standardize_series(
    series: npt.NDArray,
    detrend: bool = True,
    standardize: bool = True,
    remove_mean: bool = False,
) -> npt.NDArray:
    """
    Helper function for pre-processing data, specifically for wavelet analysis
    From: https://github.com/regeirk/pycwt/issues/35#issuecomment-809588607
    """

    # Derive the variance prior to any detrending
    std = series.std()
    smean = series.mean()

    if detrend and remove_mean:
        raise ValueError(
            "Only standardize by either removing secular trend or mean, not both."
        )

    # Remove the trend if requested
    if detrend:
        arbitrary_x = np.arange(0, series.size)
        p = np.polyfit(arbitrary_x, series, 1)
        snorm = series - np.polyval(p, arbitrary_x)
    else:
        snorm = series

    if remove_mean:
        snorm = snorm - smean

    # Standardize by the standard deviation
    if standardize:
        snorm = snorm / std

    return snorm


def normalize_xwt_results(
    signal_size: npt.NDArray,
    xwt_coeffs: npt.NDArray,
    coi: npt.NDArray,
    coi_min: float,
    freqs: npt.NDArray,
    signif: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Normalize results for plotting"""
    period = 1 / freqs
    power = (np.abs(xwt_coeffs)) ** 2  ## Normalize wavelet power spectrum
    sig95 = np.ones([1, signal_size]) * signif[:, None]
    sig95 = power / sig95  ## Want where power / sig95 > 1
    coi_plot = np.concatenate(
        [np.log2(coi), [1e-9], np.log2(period[-1:]), np.log2(period[-1:]), [1e-9]]
    ).clip(
        min=coi_min
    )  # ! To keep cone of influence from bleeding off graph
    return period, power, sig95, coi_plot


def plot_signficance_levels(
    ax: plt.Axes,
    signficance_levels: npt.NDArray,
    t_values: npt.NDArray,
    period: npt.NDArray,
    **kwargs
) -> None:
    """Plot contours for 95% significance level\n
    **kwargs**\n
    `sig_colors=`: 'k'\n
    `sig_linewidths=`: 2"""
    extent = [t_values.min(), t_values.max(), 0, max(period)]
    ax.contour(
        t_values,
        np.log2(period),
        signficance_levels,
        [-99, 1],
        colors=kwargs["sig_colors"],
        linewidths=kwargs["sig_linewidths"],
        extent=extent,
    )


def plot_cone_of_influence(
    ax: plt.Axes,
    coi: npt.NDArray,
    t_values: npt.NDArray,
    levels: List[float],
    period: npt.NDArray,
    dt: float,
    tranform_type: str,
    **kwargs
) -> None:
    """Plot shaded area for cone of influence, where edge effects may occur\n
    **Params**\n
    `transform_type=`: "cwt" or "xwt"\n
    `coi_color=`: 'k'
    `coi_alpha =`: 0.3\n
    `coi_hatch =`: "--"
    """
    color = kwargs["coi_color"]
    alpha = kwargs["coi_alpha"]
    hatch = kwargs["coi_hatch"]

    # Handle datetime values properly
    if np.issubdtype(t_values.dtype, np.datetime64):
        # Convert dt to timedelta for datetime arithmetic
        dt_timedelta = np.timedelta64(int(dt * 365.25 * 24 * 60 * 60 * 1000000), "us")
        t_array = np.concatenate(
            [
                t_values,
                t_values[-1:] + dt_timedelta,
                t_values[-1:] + dt_timedelta,
                t_values[:1] - dt_timedelta,
                t_values[:1] - dt_timedelta,
            ]
        )
    else:
        # Original behavior for non-datetime values
        t_array = np.concatenate(
            [
                t_values,
                t_values[-1:] + dt,
                t_values[-1:] + dt,
                t_values[:1] - dt,
                t_values[:1] - dt,
            ]
        )
    if tranform_type == "cwt":
        coi_array = np.concatenate(
            [
                np.log2(coi),
                [levels[2]],
                np.log2(period[-1:]),
                np.log2(period[-1:]),
                [levels[2]],
            ]
        ).clip(
            min=-2.5
        )  # ! To keep cone of influence from bleeding off graph
    if tranform_type == "xwt":
        coi_array = coi
    ax.fill(
        t_array,
        coi_array,
        color,
        alpha=alpha,
        hatch=hatch,
    )
