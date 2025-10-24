"""Helper functions for scripts"""

import logging
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, Type, TypeVar

import pandas as pd

from constants import ids, results_configs
from src import cwt, dwt, wct
from src.utils import wavelet_helpers
from src.utils.logging_helpers import define_other_module_log_level

# * Logging settings
logger = logging.getLogger(__name__)
define_other_module_log_level("Error")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Type variables for generic functions
T = TypeVar("T")
R = TypeVar("R")


@dataclass
class ProcessingConfig:
    """Configuration for data processing"""

    mother_wavelet: Any
    delta_t: float
    delta_j: float
    initial_scale: float
    levels: List[float]
    detrend: bool = False
    remove_mean: bool = True


def process_wavelet_transforms(
    data: pd.DataFrame,
    measures: List[str],
    data_class: Type[T],
    compute_function: Callable[[T], R],
    config: ProcessingConfig,
    **kwargs,
) -> Dict[str, R]:
    """Generic function to process wavelet transforms for single series"""
    results = {}

    for measure in measures:
        try:
            # Extract and prepare data
            y_values = data[measure].dropna().to_numpy()
            y_values = wavelet_helpers.standardize_series(
                y_values, detrend=config.detrend, remove_mean=config.remove_mean
            )

            # Create data object based on class type
            # Check if it's a DWT class (only needs y_values and mother_wavelet)
            if data_class.__name__ == "DataForDWT":
                data_obj = data_class(
                    y_values=y_values, mother_wavelet=config.mother_wavelet
                )
            # Check if it's a CWT class (needs t_values and other parameters)
            elif data_class.__name__ == "DataForCWT":
                t_values = data[data[measure].notna()][ids.DATE].to_numpy()
                data_obj = data_class(
                    t_values=t_values,
                    y_values=y_values,
                    mother_wavelet=config.mother_wavelet,
                    delta_t=config.delta_t,
                    delta_j=config.delta_j,
                    initial_scale=config.initial_scale,
                    levels=config.levels,
                )
            else:
                raise ValueError(f"Unsupported data class: {data_class}")

            # Compute results
            results[measure] = compute_function(data_obj, **kwargs)
            logger.debug(f"Successfully processed {measure}")

        except Exception as e:
            logger.error(f"Failed to process {measure}: {e}")
            continue

    return results


def process_wavelet_pairs(
    data: pd.DataFrame,
    pairs: List[Tuple[str, str]],
    data_class: Type[T],
    compute_function: Callable[[T], R],
    config: ProcessingConfig,
    **kwargs,
) -> Dict[Tuple[str, str], R]:
    """Process pairs of series for cross-wavelet analysis (XWT, WCT)"""
    results = {}

    for series1, series2 in pairs:
        try:
            # Extract data for both series
            subset_data = data[[series1, series2, ids.DATE]].dropna()
            y1 = subset_data[series1].to_numpy()
            y2 = subset_data[series2].to_numpy()
            actual_times = subset_data[ids.DATE].to_numpy()

            # Standardize series
            y1 = wavelet_helpers.standardize_series(
                y1, detrend=config.detrend, remove_mean=config.remove_mean
            )
            y2 = wavelet_helpers.standardize_series(
                y2, detrend=config.detrend, remove_mean=config.remove_mean
            )

            # Create data object
            # Check if it's a WCT class (needs actual_times parameter)
            if data_class.__name__ == "DataForWCT":
                data_obj = data_class(
                    y1_values=y1,
                    y2_values=y2,
                    mother_wavelet=config.mother_wavelet,
                    delta_t=config.delta_t,
                    delta_j=config.delta_j,
                    initial_scale=config.initial_scale,
                    levels=config.levels,
                    actual_times=actual_times,
                )
            else:
                data_obj = data_class(
                    y1_values=y1,
                    y2_values=y2,
                    mother_wavelet=config.mother_wavelet,
                    delta_t=config.delta_t,
                    delta_j=config.delta_j,
                    initial_scale=config.initial_scale,
                    levels=config.levels,
                )

            # Compute results
            results[(series1, series2)] = compute_function(data_obj, **kwargs)
            logger.debug(f"Successfully processed {series1} x {series2}")

        except Exception as e:
            logger.error(f"Failed to process {series1} x {series2}: {e}")
            continue

    return results


# Legacy functions for backward compatibility (deprecated)
def create_dwt_dict(
    data_for_dwt: pd.DataFrame,
    measures_list: List[str],
    **kwargs,
) -> Dict[str, Type[Any]]:
    """Create dict of discrete wavelet transform objects from DataFrame (DEPRECATED)"""
    logger.warning(
        "create_dwt_dict is deprecated. Use process_wavelet_transforms instead."
    )
    transform_dict = {}
    logger.debug("df shape: %s", data_for_dwt.shape)
    for measure in measures_list:
        transform_dict[measure] = dwt.DataForDWT(
            y_values=data_for_dwt[measure].to_numpy(), **kwargs
        )
    return transform_dict


def create_cwt_dict(
    data_for_cwt: pd.DataFrame,
    measures_list: List[str],
    **kwargs,
) -> Dict[str, Type[Any]]:
    """Create dict of continuous wavelet transform objects from DataFrame (DEPRECATED)"""
    logger.warning(
        "create_cwt_dict is deprecated. Use process_wavelet_transforms instead."
    )
    transform_dict = {}
    for measure in measures_list:
        t_values = data_for_cwt[data_for_cwt[measure].notna()][ids.DATE].to_numpy()
        y_values = data_for_cwt[data_for_cwt[measure].notna()][measure].to_numpy()
        y_values = wavelet_helpers.standardize_series(y_values)
        transform_dict[measure] = cwt.DataForCWT(
            t_values=t_values, y_values=y_values, **kwargs
        )
    return transform_dict


def create_wct_dict(
    data_for_wct: pd.DataFrame, wct_list: List[Tuple[str, str]], **kwargs
) -> Dict[Tuple[str, str], Type[wct.DataForWCT]]:
    """Create dict of wavelet coherence transform objects from DataFrame (DEPRECATED)"""
    logger.warning("create_wct_dict is deprecated. Use process_wavelet_pairs instead.")
    transform_dict = {}
    for comparison in wct_list:
        y1 = data_for_wct.dropna()[comparison[0]].to_numpy()
        y2 = data_for_wct.dropna()[comparison[1]].to_numpy()
        y1 = wavelet_helpers.standardize_series(y1, **kwargs)
        y2 = wavelet_helpers.standardize_series(y2, **kwargs)

        transform_dict[comparison] = wct.DataForWCT(
            y1_values=y1,
            y2_values=y2,
            mother_wavelet=results_configs.XWT_MOTHER_DICT[results_configs.XWT_MOTHER],
            delta_t=results_configs.XWT_DT,
            delta_j=results_configs.XWT_DJ,
            initial_scale=results_configs.XWT_S0,
            levels=results_configs.LEVELS,
        )
    return transform_dict


def create_dwt_results_dict(
    dwt_data_dict: Dict[str, Type[dwt.DataForDWT]], measures_list: List[str], **kwargs
) -> Dict[str, Type[dwt.ResultsFromDWT]]:
    """Create dict of DWT results instances (DEPRECATED)"""
    logger.warning(
        "create_dwt_results_dict is deprecated. Use process_wavelet_transforms instead."
    )
    results_dict = {}
    for measure in measures_list:
        results_dict[measure] = dwt.run_dwt(dwt_data_dict[measure], **kwargs)
    return results_dict


def create_cwt_results_dict(
    cwt_data_dict: Dict[str, Type[cwt.DataForCWT]], measures_list: List[str], **kwargs
) -> Dict[str, Type[cwt.ResultsFromCWT]]:
    """Create dict of CWT results instances (DEPRECATED)"""
    logger.warning(
        "create_cwt_results_dict is deprecated. Use process_wavelet_transforms instead."
    )
    results_dict = {}
    for measure in measures_list:
        results_dict[measure] = cwt.run_cwt(cwt_data_dict[measure], **kwargs)
    return results_dict


def create_wct_results_dict(
    wct_data_dict: Dict[Tuple[str, str], Type[wct.DataForWCT]],
    wct_list: List[Tuple[str, str]],
    **kwargs,
) -> Dict[Tuple[str, str], Type[wct.ResultsFromWCT]]:
    """Create dict of WCT results instances (DEPRECATED)"""
    logger.warning(
        "create_wct_results_dict is deprecated. Use process_wavelet_pairs instead."
    )
    results_dict = {}
    for comparison in wct_list:
        results_dict[comparison] = wct.run_wct(wct_data_dict[comparison], **kwargs)
    return results_dict
