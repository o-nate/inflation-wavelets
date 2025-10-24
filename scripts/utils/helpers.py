"""Helper functions for scripts"""

import logging
import sys
from typing import Any, Dict, List, Tuple, Type

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


def create_dwt_dict(
    data_for_dwt: pd.DataFrame,
    measures_list: List[str],
    **kwargs,
) -> Dict[str, Type[Any]]:
    """Create dict of discrete wavelet transform objects from DataFrame"""
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
    """Create dict of continuous wavelet transform objects from DataFrame"""
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
    """Create dict of wavelet coherence transform objects from DataFrame"""
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
    """Create dict of DWT results instances"""
    results_dict = {}
    for measure in measures_list:
        results_dict[measure] = dwt.run_dwt(dwt_data_dict[measure], **kwargs)
    return results_dict


def create_cwt_results_dict(
    cwt_data_dict: Dict[str, Type[cwt.DataForCWT]], measures_list: List[str], **kwargs
) -> Dict[str, Type[cwt.ResultsFromCWT]]:
    """Create dict of CWT results instances"""
    results_dict = {}
    for measure in measures_list:
        results_dict[measure] = cwt.run_cwt(cwt_data_dict[measure], **kwargs)
    return results_dict


def create_wct_results_dict(
    wct_data_dict: Dict[str, Type[wct.DataForWCT]],
    wct_list: List[Tuple[str, str]],
    **kwargs,
) -> Type[wct.ResultsFromWCT]:
    """Create dict of WCT results instances"""
    results_dict = {}
    for comparison in wct_list:
        results_dict[comparison] = wct.run_wct(wct_data_dict[comparison], **kwargs)
    return results_dict
