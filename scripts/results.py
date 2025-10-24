"""Plot results from wavelet transformations"""

# %%
import logging
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.iolib.summary2

from constants import ids, results_configs
from src import (
    cwt,
    descriptive_stats,
    dwt,
    phase_diff_key,
    phase_diff_sines,
    regression,
    wct,
)
from src.utils import helpers, wavelet_helpers
from src.utils.logging_helpers import define_other_module_log_level
from src import retrieve_data
from scripts.utils.helpers import (
    create_cwt_dict,
    create_cwt_results_dict,
    create_dwt_dict,
    create_dwt_results_dict,
    create_wct_dict,
    create_wct_results_dict,
    process_wavelet_transforms,
    process_wavelet_pairs,
    ProcessingConfig,
)

# * Logging settings
logger = logging.getLogger(__name__)
define_other_module_log_level("Error")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# * Rounding settings
pd.set_option(
    "display.float_format", lambda x: f"%.{results_configs.DECIMAL_PLACES}f" % x
)

SERIES_COMPARISONS = [
    (ids.DIFF_LOG_CPI, ids.EXPECTATIONS),
    (ids.EXPECTATIONS, ids.NONDURABLES_CHG),
    (ids.EXPECTATIONS, ids.DURABLES_CHG),
    (ids.EXPECTATIONS, ids.SAVINGS_RATE),
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

# %% [markdown]
# Figure 1 - Example, Phase differences
phase_diff_sines.plot_phase_diff(export=False)

# %% [markdown]
## Pre-process data
# US data

#  %%
# * CPI
raw_data = retrieve_data.get_fed_data(
    ids.US_CPI,
)
cpi, _, _ = retrieve_data.clean_fed_data(raw_data)
cpi.rename(columns={"value": ids.CPI}, inplace=True)

# * Inflation rate
raw_data = retrieve_data.get_fed_data(
    ids.US_CPI,
    units="pc1",
    freq="m",
)
measured_inf, _, _ = retrieve_data.clean_fed_data(raw_data)
measured_inf.rename(columns={"value": ids.INFLATION}, inplace=True)

# * Inflation expectations
raw_data = retrieve_data.get_fed_data(
    ids.US_INF_EXPECTATIONS,
)
inf_exp, _, _ = retrieve_data.clean_fed_data(raw_data)
inf_exp.rename(columns={"value": ids.EXPECTATIONS}, inplace=True)

# * Non-durables consumption, monthly
raw_data = retrieve_data.get_fed_data(
    ids.US_NONDURABLES_CONSUMPTION,
)
nondur_consump, _, _ = retrieve_data.clean_fed_data(raw_data)
nondur_consump.rename(columns={"value": ids.NONDURABLES}, inplace=True)

# * Durables consumption, monthly
raw_data = retrieve_data.get_fed_data(
    ids.US_DURABLES_CONSUMPTION,
)
dur_consump, _, _ = retrieve_data.clean_fed_data(raw_data)
dur_consump.rename(columns={"value": ids.DURABLES}, inplace=True)

# * Non-durables consumption change, monthly
raw_data = retrieve_data.get_fed_data(
    ids.US_NONDURABLES_CONSUMPTION,
    units="pc1",
    freq="m",
)
nondur_consump_chg, _, _ = retrieve_data.clean_fed_data(raw_data)
nondur_consump_chg.rename(columns={"value": ids.NONDURABLES_CHG}, inplace=True)

# * Durables consumption change, monthly
raw_data = retrieve_data.get_fed_data(
    ids.US_DURABLES_CONSUMPTION,
    units="pc1",
    freq="m",
)
dur_consump_chg, _, _ = retrieve_data.clean_fed_data(raw_data)
dur_consump_chg.rename(columns={"value": ids.DURABLES_CHG}, inplace=True)

# * Personal savings
raw_data = retrieve_data.get_fed_data(
    ids.US_SAVINGS,
)
save, _, _ = retrieve_data.clean_fed_data(raw_data)
save.rename(columns={"value": ids.SAVINGS}, inplace=True)

# * Personal savings change
raw_data = retrieve_data.get_fed_data(
    ids.US_SAVINGS,
    units="pc1",
    freq="m",
)
save_chg, _, _ = retrieve_data.clean_fed_data(raw_data)
save_chg.rename(columns={"value": ids.SAVINGS_CHG}, inplace=True)

# * Personal savings rate
raw_data = retrieve_data.get_fed_data(
    ids.US_SAVINGS_RATE,
)
save_rate, _, _ = retrieve_data.clean_fed_data(raw_data)
save_rate.rename(columns={"value": ids.SAVINGS_RATE}, inplace=True)

# * Merge dataframes to align dates and remove extras
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

# # * Remove rows without data for all measures
# us_data.dropna(inplace=True)

# * Add real value columns
logger.info(
    "Using constant dollars from %s, CPI: %s",
    results_configs.CONSTANT_DOLLAR_DATE,
    us_data[us_data[ids.DATE] == pd.Timestamp(results_configs.CONSTANT_DOLLAR_DATE)][
        ids.CPI
    ].iat[0],
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

# ! Set fixed end date
us_data = us_data[us_data["date"] <= pd.to_datetime(results_configs.END_DATE)]

usa_sliced = pd.concat([us_data.head(), us_data.tail()])
usa_sliced

# %%
# * Create measured inflation dataframe
inf_data = us_data[
    [
        ids.DATE,
        ids.EXPECTATIONS,
        ids.INFLATION,
        ids.NONDURABLES_CHG,
        ids.DURABLES_CHG,
        ids.SAVINGS_RATE,
    ]
].copy()
inf_data.dropna(inplace=True)
inf_data.columns = [
    "Date",
    "Expectations",
    "CPI inflation",
    "Nondurables (% chg)",
    "Durables (% chg)",
    "Savings (%)",
]

# %%
inf_melt = pd.melt(inf_data, ["Date"])
inf_melt.rename(columns={"value": "%"}, inplace=True)

# %% [markdown]
##### Figure 4 - Time series: Measured Inflation
_, (ax, bx, cx, dx) = plt.subplots(4, 1, sharex=True, figsize=(10, 9))

# * Inflation x expectations
measures_to_plot = ["Expectations", "CPI inflation"]
data = inf_melt[inf_melt["variable"].isin(measures_to_plot)]
ax = sns.lineplot(data=data, x="Date", y="%", hue="variable", ax=ax)
ax.legend().set_title(None)
ax.legend(loc="upper center", frameon=False)

# * Expecatations x nondurables
measures_to_plot = ["Expectations", "Nondurables (% chg)"]
data = inf_melt[inf_melt["variable"].isin(measures_to_plot)]
bx = sns.lineplot(data=data, x="Date", y="%", hue="variable", ax=bx)
bx.legend().set_title(None)
bx.legend(loc="upper center", frameon=False)

# * Expecatations x durables
measures_to_plot = ["Expectations", "Durables (% chg)"]
data = inf_melt[inf_melt["variable"].isin(measures_to_plot)]
cx = sns.lineplot(data=data, x="Date", y="%", hue="variable", ax=cx)
cx.legend().set_title(None)
cx.legend(loc="upper center", frameon=False)

# * Expecatations x savings
measures_to_plot = ["Expectations", "Savings (%)"]
data = inf_melt[inf_melt["variable"].isin(measures_to_plot)]
dx = sns.lineplot(data=data, x="Date", y="%", hue="variable", ax=dx)
dx.legend().set_title(None)
dx.legend(loc="upper center", frameon=False)


# ! Export figure
# parent_dir = Path(__file__).parents[1]
# export_file = parent_dir / "results" / f"time_series_plots.png"
# plt.savefig(export_file, bbox_inches="tight")

# %% [markdown]
##### Table 1: Descriptive statistics
# %%
descriptive_statistics_results = descriptive_stats.generate_descriptive_statistics(
    us_data[
        [
            ids.DATE,
            ids.DIFF_LOG_CPI,
            # ids.EXPECTATIONS,
            ids.DIFF_LOG_EXPECTATIONS,
            ids.DIFF_LOG_NONDURABLES,
            ids.DIFF_LOG_DURABLES,
            ids.DIFF_LOG_SAVINGS,
        ]
    ].dropna(),
    results_configs.STATISTICS_TESTS,
    export_table=False,
)
descriptive_statistics_results

# %% [markdown]
##### Table 2: Correlation matrix
# %%
us_corr = descriptive_stats.correlation_matrix_pvalues(
    data=us_data[[c for c in us_data.columns if "log" in c]],
    hypothesis_threshold=results_configs.HYPOTHESIS_THRESHOLD,
    decimals=results_configs.DECIMAL_PLACES,
    display=False,
    export_table=False,
)
us_corr

# %% [markdown]
## 3.2) Exploratory analysis

# %% [markdown]
### 3.2.1) Time scale decomposition

# %%
# * Create dwt dict
dwt_measures = [
    ids.INFLATION,
    ids.EXPECTATIONS,
    ids.NONDURABLES_CHG,
    ids.DURABLES_CHG,
    ids.SAVINGS_RATE,
    ids.DIFF_LOG_CPI,
    ids.DIFF_LOG_EXPECTATIONS,
    ids.DIFF_LOG_NONDURABLES,
    ids.DIFF_LOG_DURABLES,
    ids.DIFF_LOG_SAVINGS,
    ids.DIFF_LOG_REAL_NONDURABLES,
    ids.DIFF_LOG_REAL_DURABLES,
    ids.DIFF_LOG_REAL_SAVINGS,
]

# * Process DWTs using new generic function
dwt_config = ProcessingConfig(
    mother_wavelet=results_configs.DWT_MOTHER_WAVELET,
    delta_t=0,  # Not used for DWT
    delta_j=0,  # Not used for DWT
    initial_scale=0,  # Not used for DWT
    levels=[],
)

dwt_results_dict = process_wavelet_transforms(
    data=us_data.dropna(),
    measures=dwt_measures,
    data_class=dwt.DataForDWT,
    compute_function=dwt.run_dwt,
    config=dwt_config,
)

# * Numpy array for date
t = us_data.dropna()[ids.DATE].to_numpy()

# %% [markdown]
## Figure 5 - Frequency decomposition of expectations
fig = dwt.plot_components(
    label=ids.EXPECTATIONS,
    coeffs=dwt_results_dict[ids.EXPECTATIONS].coeffs,
    time=t,
    levels=dwt_results_dict[ids.EXPECTATIONS].levels,
    wavelet=results_configs.DWT_MOTHER_WAVELET,
    figsize=(15, 20),
    sharex=True,
)
plt.legend("", frameon=False)

# ! Export figure
# parent_dir = Path(__file__).parents[1]
# export_file = parent_dir / "results" / "expectations_decomposition.png"
# plt.savefig(export_file, bbox_inches="tight")

# %% [markdown]
## Figure 6 - Smoothing of expectations
# Create DWT data object for smoothing using the same standardized data
# that was used in the processing
y_values = us_data.dropna()[ids.EXPECTATIONS].to_numpy()
y_values = wavelet_helpers.standardize_series(y_values, detrend=False, remove_mean=True)

dwt_data = dwt.DataForDWT(
    y_values=y_values,
    mother_wavelet=results_configs.DWT_MOTHER_WAVELET,
)

dwt_results_dict[ids.EXPECTATIONS].smooth_signal(
    y_values=dwt_data.y_values,
    mother_wavelet=dwt_data.mother_wavelet,
)

fig = dwt.plot_smoothing(
    dwt_results_dict[ids.EXPECTATIONS].smoothed_signal_dict,
    t,
    dwt_data.y_values,
    ascending=True,
    figsize=(15, 20),
    sharex=True,
)

# ! Export figure
# plt.legend("", frameon=False)
# parent_dir = Path(__file__).parents[1]
# export_file = parent_dir / "results" / "expectations_smoothing.png"
# plt.savefig(export_file, bbox_inches="tight")

# %% [markdown]
## Figure 6 - Frequency decomposition of expectations and nondurables consumption (US)
# * Plot comparison decompositions of expectations and other measure
for comp in SERIES_COMPARISONS[1:4] + SERIES_COMPARISONS[14:17]:
    fig = regression.plot_compare_components(
        a_label=comp[0],
        b_label=comp[1],
        a_coeffs=dwt_results_dict[comp[0]].coeffs,
        b_coeffs=dwt_results_dict[comp[1]].coeffs,
        time=t,
        levels=dwt_results_dict[comp[0]].levels,
        wavelet=results_configs.DWT_MOTHER_WAVELET,
        figsize=(10, 15),
        sharex=True,
    )
    # ! Export figure
    # parent_dir = Path(__file__).parents[1]
    # export_file = parent_dir / "results" / f"decomposition_{comp[0]}_{comp[1]}.png"
    # plt.savefig(export_file, bbox_inches="tight")

# %% [markdown]
### 3.2.2) Individual time series: Continuous wavelet transforms
cwt_measures = {
    # ids.INFLATION: "CPI inflation",
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
    # # # # ids.NONDURABLES,
    # # # # ids.DURABLES,
    # # # ids.SAVINGS,
    # # # ids.REAL_NONDURABLES,
    # # # ids.REAL_DURABLES,
    # # # ids.REAL_SAVINGS,
}
# * Process CWTs using new generic function
cwt_config = ProcessingConfig(
    mother_wavelet=results_configs.CWT_MOTHER,
    delta_t=results_configs.DT,
    delta_j=results_configs.DJ,
    initial_scale=results_configs.S0,
    levels=results_configs.LEVELS,
)

cwt_results_dict = process_wavelet_transforms(
    data=us_data.dropna(),
    measures=list(cwt_measures.keys()),
    data_class=cwt.DataForCWT,
    compute_function=cwt.run_cwt,
    config=cwt_config,
    normalize=True,
)

# %%
# * Plot CWTs
plt.close("all")
# _, axs = plt.subplots(len(cwt_results_dict), **results_configs.CWT_FIG_PROPS)

for measure, cwt_result in cwt_results_dict.items():
    fig, ax = plt.subplots(1, **results_configs.CWT_FIG_PROPS)

    # Create CWT data object for plotting
    cwt_data = cwt.DataForCWT(
        t_values=us_data.dropna()[ids.DATE].to_numpy(),
        y_values=us_data.dropna()[measure].to_numpy(),
        mother_wavelet=results_configs.CWT_MOTHER,
        delta_t=results_configs.DT,
        delta_j=results_configs.DJ,
        initial_scale=results_configs.S0,
        levels=results_configs.LEVELS,
    )

    cwt.plot_cwt(
        ax,
        cwt_data,
        cwt_result,
        **results_configs.CWT_PLOT_PROPS,
    )

    # * Set labels/title
    ax.set_xlabel("")
    ax.set_ylabel("Period (years)")
    ax.set_title(cwt_measures[measure])

    # ! Export figures to file
    # parent_dir = Path(__file__).parents[1]
    # export_file = parent_dir / "results" / f"cwt_{measure}.png"
    # plt.savefig(export_file, bbox_inches="tight")

plt.show()


# %% [markdown]
### 3.2.3) Time series co-movements: Wavelet coherence transforms and phase difference
phase_diff_key.plot_phase_difference_key(export=False)

# %%
comparisons = (
    SERIES_COMPARISONS[14:17]
    + SERIES_COMPARISONS[1:3]
    + [(ids.EXPECTATIONS, ids.SAVINGS_CHG)]
)

# * Process WCTs using new generic function
wct_config = ProcessingConfig(
    mother_wavelet=results_configs.XWT_MOTHER_DICT[results_configs.XWT_MOTHER],
    delta_t=results_configs.XWT_DT,
    delta_j=results_configs.XWT_DJ,
    initial_scale=results_configs.XWT_S0,
    levels=results_configs.LEVELS,
    detrend=False,
    remove_mean=True,
)

wct_results_dict = process_wavelet_pairs(
    data=us_data,
    pairs=comparisons,
    data_class=wct.DataForWCT,
    compute_function=wct.run_wct,
    config=wct_config,
)

# * Plot WCT coherence spectrum
TOTAL_SUBPLOTS = len(comparisons)
PLOT_COLS = TOTAL_SUBPLOTS // 2
PLOT_ROWS = 2
if TOTAL_SUBPLOTS % PLOT_COLS != 0:
    PLOT_ROWS += 1

fig = plt.figure(1, figsize=(20, 10))
axes = []
POSITION = 0
for i, comp in enumerate(comparisons):
    POSITION = i + 1
    ax = fig.add_subplot(PLOT_ROWS, PLOT_COLS, POSITION)
    axes.append(ax)
    # Create data object for plotting (needed for wct.plot_wct)
    # Get data without NaN values to ensure alignment
    clean_data = us_data[[comp[0], comp[1], ids.DATE]].dropna()
    y1 = clean_data[comp[0]].to_numpy()
    y2 = clean_data[comp[1]].to_numpy()
    actual_times = clean_data[ids.DATE].to_numpy()

    y1 = wavelet_helpers.standardize_series(y1, detrend=False, remove_mean=True)
    y2 = wavelet_helpers.standardize_series(y2, detrend=False, remove_mean=True)

    wct_data = wct.DataForWCT(
        y1_values=y1,
        y2_values=y2,
        mother_wavelet=results_configs.XWT_MOTHER_DICT[results_configs.XWT_MOTHER],
        delta_t=results_configs.XWT_DT,
        delta_j=results_configs.XWT_DJ,
        initial_scale=results_configs.XWT_S0,
        levels=results_configs.LEVELS,
        actual_times=actual_times,
    )

    wct.plot_wct(
        ax,
        wct_data,
        wct_results_dict[comp],
        include_significance=True,
        include_cone_of_influence=True,
        include_phase_difference=True,
        **wct.WCT_PLOT_PROPS,
    )
    # * Invert y axis
    ax.set_ylim(ax.get_ylim()[::-1])

    # * Set y axis tick labels
    y_ticks = 2 ** np.arange(
        np.ceil(np.log2(wct_results_dict[comp].period.min())),
        np.ceil(np.log2(wct_results_dict[comp].period.max())),
    )
    ax.set_yticks(np.log2(y_ticks))
    if i == 0:
        ax.set_ylabel("Period (years)")
    if i % 2 == 0:
        ax.set_yticklabels(y_ticks)
    else:
        ## Right-hand column use y axis from left-hand column
        ax.tick_params("y", labelleft=False)
    ax.set_title(f"{comp[0]} WCT {comp[1]} (US)")
for i, ax in enumerate(axes[1:]):
    ax.sharex(axes[0])

# * Format x-axis as dates
for ax in axes:
    # Convert numpy datetime64 to matplotlib dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))  # Show every 5 years
    ax.xaxis.set_minor_locator(mdates.YearLocator())  # Minor ticks every year
    # Rotate x-axis labels for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.show()

# %% [markdown]
## 3.3) Regression analysis
### 3.3.1) Baseline model
regressions = {}

for comp in SERIES_COMPARISONS[14:17]:
    regressions[comp[1]] = regression.simple_regression(
        us_data.dropna(), comp[0], comp[1]
    )

results = statsmodels.iolib.summary2.summary_col(
    list(regressions.values()),
    stars=True,
    model_names=list(regressions),
)
results

# %% [markdown]
#### Percentages
regressions = {}

for comp in SERIES_COMPARISONS[1:3] + [(ids.EXPECTATIONS, ids.SAVINGS_CHG)]:
    regressions[comp[1]] = regression.simple_regression(
        us_data.dropna(), comp[0], comp[1]
    )

results = statsmodels.iolib.summary2.summary_col(
    list(regressions.values()),
    stars=True,
    model_names=list(regressions),
)
results

# %% [markdown]
#### Real terms
regressions = {}

for comp in SERIES_COMPARISONS[17:]:
    regressions[comp[1]] = regression.simple_regression(
        us_data.dropna(), comp[0], comp[1]
    )

results = statsmodels.iolib.summary2.summary_col(
    list(regressions.values()),
    stars=True,
    model_names=list(regressions),
)
results

# %% [markdown]
## 3.3) Time scale regression
# Table 7 - Time Scale Decomposition: OLS Regression of Nondurables Consumption
# on Inflation Expectations (US)
for comp in (
    SERIES_COMPARISONS[14:17] + SERIES_COMPARISONS[1:4] + SERIES_COMPARISONS[17:]
):
    time_scale_results = regression.time_scale_regression(
        input_coeffs=dwt_results_dict[comp[0]].coeffs,
        output_coeffs=dwt_results_dict[comp[1]].coeffs,
        levels=dwt_results_dict[comp[0]].levels,
        mother_wavelet=results_configs.DWT_MOTHER_WAVELET,
    )
    print(f"\nRegressing {comp[1]} on {comp[0]}")
    print(time_scale_results.as_text())
    time_scale_results

# %% [markdown]
### Nondurables
comp = (ids.DIFF_LOG_EXPECTATIONS, ids.DIFF_LOG_NONDURABLES)
print(f"Regressing {comp[1]} on {comp[0]}")
time_scale_results = regression.time_scale_regression(
    input_coeffs=dwt_results_dict[comp[0]].coeffs,
    output_coeffs=dwt_results_dict[comp[1]].coeffs,
    levels=dwt_results_dict[comp[0]].levels,
    mother_wavelet=results_configs.DWT_MOTHER_WAVELET,
)
time_scale_results

# %% [markdown]
### Nondurables (% chg)
comp = (ids.EXPECTATIONS, ids.NONDURABLES_CHG)
print(f"Regressing {comp[1]} on {comp[0]}")
time_scale_results = regression.time_scale_regression(
    input_coeffs=dwt_results_dict[comp[0]].coeffs,
    output_coeffs=dwt_results_dict[comp[1]].coeffs,
    levels=dwt_results_dict[comp[0]].levels,
    mother_wavelet=results_configs.DWT_MOTHER_WAVELET,
)
time_scale_results

# %% [markdown]
### Nondurables (real)
comp = (ids.DIFF_LOG_EXPECTATIONS, ids.DIFF_LOG_REAL_NONDURABLES)
print(f"Regressing {comp[1]} on {comp[0]}")
time_scale_results = regression.time_scale_regression(
    input_coeffs=dwt_results_dict[comp[0]].coeffs,
    output_coeffs=dwt_results_dict[comp[1]].coeffs,
    levels=dwt_results_dict[comp[0]].levels,
    mother_wavelet=results_configs.DWT_MOTHER_WAVELET,
)
time_scale_results

# %% [markdown]
### Durables
comp = (ids.DIFF_LOG_EXPECTATIONS, ids.DIFF_LOG_DURABLES)
print(f"Regressing {comp[1]} on {comp[0]}")
time_scale_results = regression.time_scale_regression(
    input_coeffs=dwt_results_dict[comp[0]].coeffs,
    output_coeffs=dwt_results_dict[comp[1]].coeffs,
    levels=dwt_results_dict[comp[0]].levels,
    mother_wavelet=results_configs.DWT_MOTHER_WAVELET,
)
time_scale_results

# %% [markdown]
### Durables (% chg)
comp = (ids.EXPECTATIONS, ids.DURABLES_CHG)
print(f"Regressing {comp[1]} on {comp[0]}")
time_scale_results = regression.time_scale_regression(
    input_coeffs=dwt_results_dict[comp[0]].coeffs,
    output_coeffs=dwt_results_dict[comp[1]].coeffs,
    levels=dwt_results_dict[comp[0]].levels,
    mother_wavelet=results_configs.DWT_MOTHER_WAVELET,
)
time_scale_results

# %% [markdown]
### Durables (real)
comp = (ids.DIFF_LOG_EXPECTATIONS, ids.DIFF_LOG_REAL_DURABLES)
print(f"Regressing {comp[1]} on {comp[0]}")
time_scale_results = regression.time_scale_regression(
    input_coeffs=dwt_results_dict[comp[0]].coeffs,
    output_coeffs=dwt_results_dict[comp[1]].coeffs,
    levels=dwt_results_dict[comp[0]].levels,
    mother_wavelet=results_configs.DWT_MOTHER_WAVELET,
)
time_scale_results

# %% [markdown]
### Savings
comp = (ids.DIFF_LOG_EXPECTATIONS, ids.DIFF_LOG_SAVINGS)
print(f"Regressing {comp[1]} on {comp[0]}")
time_scale_results = regression.time_scale_regression(
    input_coeffs=dwt_results_dict[comp[0]].coeffs,
    output_coeffs=dwt_results_dict[comp[1]].coeffs,
    levels=dwt_results_dict[comp[0]].levels,
    mother_wavelet=results_configs.DWT_MOTHER_WAVELET,
)
time_scale_results

# %% [markdown]
### Savings (% chg)
comp = (ids.EXPECTATIONS, ids.SAVINGS_RATE)
print(f"Regressing {comp[1]} on {comp[0]}")
time_scale_results = regression.time_scale_regression(
    input_coeffs=dwt_results_dict[comp[0]].coeffs,
    output_coeffs=dwt_results_dict[comp[1]].coeffs,
    levels=dwt_results_dict[comp[0]].levels,
    mother_wavelet=results_configs.DWT_MOTHER_WAVELET,
)
time_scale_results

# %% [markdown]
### Savings (real)
comp = (ids.DIFF_LOG_EXPECTATIONS, ids.DIFF_LOG_REAL_SAVINGS)
print(f"Regressing {comp[1]} on {comp[0]}")
time_scale_results = regression.time_scale_regression(
    input_coeffs=dwt_results_dict[comp[0]].coeffs,
    output_coeffs=dwt_results_dict[comp[1]].coeffs,
    levels=dwt_results_dict[comp[0]].levels,
    mother_wavelet=results_configs.DWT_MOTHER_WAVELET,
)
time_scale_results
