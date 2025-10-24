"""
Smoothing of signals via wavelet reconstruction
"""

from dataclasses import dataclass, field
import logging
import sys
from typing import Dict, Generator, List, Tuple, Type, Union

import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
import numpy.typing as npt
import pywt

from constants import ids
from src.utils.logging_helpers import define_other_module_log_level
from src.utils.wavelet_helpers import align_series
from src import retrieve_data

# * Logging settings
logger = logging.getLogger(__name__)
define_other_module_log_level("debug")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# * Constants
MOTHER = pywt.Wavelet("db4")


@dataclass
class DataForDWT:
    """Holds data for discrete wavelet transform"""

    y_values: npt.NDArray
    mother_wavelet: Type
    levels: float = None


@dataclass
class ResultsFromDWT:
    """Holds data for discrete wavelet transform
    `coeffs`: transform coefficients
    `levels`: transform levels applied
    `smoothed_signal_dict`: dictionary of coefficients for each time scale"""

    coeffs: npt.NDArray
    levels: float
    smoothed_signal_dict: Dict[int, Dict[str, npt.NDArray]] = field(
        default_factory=dict
    )

    def smooth_signal(
        self, y_values: npt.NDArray, mother_wavelet: Type
    ) -> Dict[int, Dict[str, npt.NDArray]]:
        """Generate smoothed signals based off wavelet coefficients for each pre-defined level"""
        ## Initialize dict for reconstructed signals
        signals_dict = {}

        ## Loop through levels and remove detail level component(s)
        # ! Note: signal_dict[l] provides the signal with levels <= l removed
        logger.debug(self.levels)
        for l in range(self.levels, 0, -1):
            print(f"s_{l} stored with key {l}")
            smooth_coeffs = self.coeffs.copy()
            signals_dict[l] = {}
            ## Set remaining detail coefficients to zero
            for coeff in range(1, l + 1):
                smooth_coeffs[-1 * coeff] = np.zeros_like(smooth_coeffs[-1 * coeff])
            signals_dict[l]["coeffs"] = smooth_coeffs
            # Reconstruct the signal using only the approximation coefficients
            reconst = pywt.waverec(smooth_coeffs, mother_wavelet)
            signals_dict[l]["signal"] = trim_signal(y_values, reconst)
        self.smoothed_signal_dict = signals_dict


def trim_signal(
    original_signal: npt.NDArray, reconstructed: npt.NDArray
) -> npt.NDArray:
    """Removes first or last observation for odd-numbered datasets"""
    ## Time series with uneven result in mismatched lengths with the reconstructed
    ## signal, so we remove a value from the approximated signal
    if len(original_signal) % 2 != 0:
        logger.warning("Trimming signal at beginning")
        return reconstructed[1:]
    return reconstructed


def run_dwt(dwt_data: Type[DataForDWT]) -> Type[ResultsFromDWT]:
    """Generate levels and coefficients from discrete wavelet transform with
    given wavelet function"""
    ## Define the wavelet type
    # w = dwt_data.mother_wavelet
    ## Choose the maximum decomposition level
    if dwt_data.levels is None:
        dwt_levels = pywt.dwt_max_level(
            data_len=len(dwt_data.y_values), filter_len=dwt_data.mother_wavelet.dec_len
        )
        print(
            f"""Max decomposition level of {dwt_levels} for time series length 
            of {len(dwt_data.y_values)}"""
        )
    else:
        dwt_levels = dwt_data.levels
    dwt_coeffs = pywt.wavedec(
        dwt_data.y_values, dwt_data.mother_wavelet, level=dwt_data.levels
    )
    return ResultsFromDWT(dwt_coeffs, dwt_levels)


def reconstruct_signal_component(
    signal_coeffs: list, wavelet: str, level: int
) -> tuple[dict, int]:
    """Reconstruct individual component"""
    component_coeffs = signal_coeffs.copy()
    for l in range(len(signal_coeffs)):
        if l == level:
            component_coeffs[l] = component_coeffs[l]
        else:
            component_coeffs[l] = np.zeros_like(component_coeffs[l])
    return pywt.waverec(component_coeffs, wavelet)


def plot_components(
    label: str,
    coeffs: npt.NDArray,
    time: npt.NDArray,
    levels: int,
    wavelet: str,
    **kwargs,
) -> matplotlib.figure.Figure:
    """Plot each series component separately"""
    fig, ax = plt.subplots(levels + 1, 1, **kwargs)
    smooth_component = reconstruct_signal_component(coeffs, wavelet, 0)
    logger.warning(
        "lengths x: %s, t: %s",
        len(smooth_component),
        len(time),
    )

    # * Align array legnths
    if len(smooth_component) != len(time):
        smooth_component = align_series(time, smooth_component)

    ax[0].plot(time, smooth_component, label=label)
    ax[0].set_title(rf"$S_{{{levels}}}$", size=15)

    components = {}
    for l in range(1, levels + 1):
        components[l] = {}
        components[l][label] = reconstruct_signal_component(coeffs, wavelet, l)
        if len(time) != len(components[l][label]):
            components[l][label] = align_series(time, components[l][label])
        ax[l].plot(time, components[l][label], label=label)
        ax[l].set_title(rf"$D_{{{levels + 1 - l}}}$", size=15)
    plt.legend(loc="upper left")
    return fig


def plot_smoothing(
    smooth_signals: dict,
    original_t: npt.NDArray,
    original_y: npt.NDArray,
    ascending: bool = False,
    **kwargs,
) -> Tuple[matplotlib.figure.Figure, str]:
    """Graph series of smoothed signals with original signal"""
    fig, axs = plt.subplots(len(smooth_signals), 1, **kwargs)
    # * Loop through levels and add detail level components
    if ascending:
        order = reversed(list(smooth_signals.items()))
    else:
        order = list(smooth_signals.items())
    for i, (level, signal) in enumerate(order, 1):
        logger.debug(i)
        smooth_level = len(smooth_signals) - level
        ## Subplot for each smooth signal
        # plt.subplot(len(smooth_signals), 1, i)
        axs[i - 1].plot(original_t, original_y, label="Original")
        axs[i - 1].plot(
            original_t, signal["signal"], label=rf"$S_{{j-{smooth_level}}}$"
        )
        axs[i - 1].set_title(rf"Approximation: $S_{{j-{smooth_level}}}$", size=15)
        if i - 1 == 0:
            axs[i - 1].legend(loc="upper right")
        else:
            axs[i - 1].legend(loc="upper right")
    return fig


def main() -> None:
    """Run script"""

    raw_data = retrieve_data.get_fed_data(ids.US_INF_EXPECTATIONS)
    df, t, y = retrieve_data.clean_fed_data(raw_data)

    # * Create instance of DataForDWT class
    data_for_dwt = DataForDWT(y, MOTHER)

    # * Apply DWT and smooth signal
    results_from_dwt = run_dwt(data_for_dwt)
    results_from_dwt.smooth_signal(
        y_values=data_for_dwt.y_values, mother_wavelet=data_for_dwt.mother_wavelet
    )

    logger.debug("start: %s, end: %s", df.date.min(), df.date.max())

    fig = plot_smoothing(
        smooth_signals=results_from_dwt.smoothed_signal_dict,
        original_t=t,
        original_y=y,
        figsize=(10, 10),
        sharex=True,
    )

    plt.xlabel("Year", size=15)
    plt.ylabel("%", size=15)
    fig.tight_layout()
    plt.show()

    fig2 = plot_components(
        label=ids.EXPECTATIONS,
        coeffs=results_from_dwt.coeffs,
        time=t,
        levels=results_from_dwt.levels,
        wavelet=MOTHER,
        figsize=(10, 10),
        sharex=True,
    )
    plt.legend("", frameon=False)
    plt.show()


if __name__ == "__main__":
    main()
