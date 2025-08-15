# %%
import logging
from pathlib import Path
import sys

import matplotlib.pyplot as plt
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
    process_camme,
    regression,
    xwt,
)
from src.utils import helpers
from src.utils.logging_helpers import define_other_module_log_level
from src import retrieve_data
from scripts.utils.helpers import (
    create_cwt_dict,
    create_cwt_results_dict,
    create_dwt_dict,
    create_dwt_results_dict,
    create_xwt_dict,
    create_xwt_results_dict,
)

# * Logging settings
logger = logging.getLogger(__name__)
define_other_module_log_level("Error")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# %%
ID = "CPALTT01BRM659N"


# * Inflation rate
raw_data = retrieve_data.get_fed_data(ID)
brazil, _, _ = retrieve_data.clean_fed_data(raw_data)
brazil.rename(columns={"value": "%"}, inplace=True)

# %%
brazil.describe()


# %%
sns.lineplot(data=brazil, x="date", y="%")

# %% [markdown]
# French data
# * CPI
raw_data = retrieve_data.get_fed_data(ids.FR_CPI)
fr_cpi, _, _ = retrieve_data.clean_fed_data(raw_data)
fr_cpi.rename(columns={"value": ids.CPI}, inplace=True)

# * Measured inflation
raw_data = retrieve_data.get_fed_data(ids.FR_CPI, units="pc1", freq="m")
fr_inf, _, _ = retrieve_data.clean_fed_data(raw_data)
fr_inf.rename(columns={"value": ids.INFLATION}, inplace=True)

# * Inflation expectations
_, fr_exp = process_camme.preprocess(process_camme.camme_dir)
## Remove random lines with month as letter
fr_exp = fr_exp[fr_exp["month"].apply(isinstance, args=(int,))]
## Create date column
fr_exp[ids.DATE] = pd.to_datetime(fr_exp[["year", "month"]].assign(DAY=1))
## Use just quantitative expectations and date
fr_exp = fr_exp[[ids.DATE, "inf_exp_val_inc", "inf_exp_val_dec"]]
## Convert to negative for averaging
fr_exp["inf_exp_val_dec"] = fr_exp["inf_exp_val_dec"] * -1
## Melt then pivot to get average expectation for each month
fr_exp_melt = pd.melt(fr_exp, [ids.DATE])
fr_exp = pd.pivot_table(fr_exp_melt, index=ids.DATE, aggfunc="mean")
fr_exp.rename(columns={"value": ids.EXPECTATIONS}, inplace=True)

# %%
# * Food consumption
raw_data = retrieve_data.get_insee_data(ids.FR_FOOD_CONSUMPTION)
fr_food_cons, _, _ = retrieve_data.clean_insee_data(raw_data)
fr_food_cons.rename(columns={"value": "food"}, inplace=True)

# * Goods consumption
raw_data = retrieve_data.get_insee_data(ids.FR_GOODS_CONSUMPTION)
fr_goods_cons, _, _ = retrieve_data.clean_insee_data(raw_data)
fr_goods_cons.rename(columns={"value": "goods"}, inplace=True)

# * Durables consumption
raw_data = retrieve_data.get_insee_data(ids.FR_DURABLES_CONSUMPTION)
fr_dur_cons, _, _ = retrieve_data.clean_insee_data(raw_data)
fr_dur_cons.rename(columns={"value": "durables"}, inplace=True)

# * Expected savings
raw_data = retrieve_data.get_insee_data(ids.FR_EXPECTED_SAVINGS)
fr_exp_sav, _, _ = retrieve_data.clean_insee_data(raw_data)
fr_exp_sav.rename(columns={"value": "exp_savings"}, inplace=True)

# * Perceived savings
raw_data = retrieve_data.get_insee_data(ids.FR_PERCEIVED_SAVINGS)
fr_per_sav, _, _ = retrieve_data.clean_insee_data(raw_data)
fr_per_sav.rename(columns={"value": "per_savings"}, inplace=True)

# * Savings rate
data_path = Path(__file__).parents[1] / "data"
savings_data_path = data_path / "10_ECC-11_ECO-11E_Figure5.csv"
livret_a_data_path = (
    data_path / "WEBSTAT-observations-2025-08-06T13_57_31.393+02_00.csv"
)
fr_sav = pd.read_csv(savings_data_path)
fr_sav = fr_sav.rename(
    columns={
        "Année": ids.DATE,
        "Épargne": "savings",
        "Épargne financière": "financial_savings",
    }
)
fr_sav[ids.DATE] = fr_sav[ids.DATE].astype("int32")
fr_sav[ids.DATE] = pd.to_datetime(fr_sav[ids.DATE], format="%Y")
fr_sav[ids.DATE] = fr_sav[ids.DATE].dt.to_period("M")

fr_livret = pd.read_csv(livret_a_data_path, delimiter=";")
# fr_livret[ids.DATE] = fr_livret[ids.DATE].astype("int32")
fr_livret[ids.DATE] = pd.to_datetime(fr_livret[ids.DATE])
fr_livret[ids.DATE] = fr_livret[ids.DATE].dt.to_period("M")
fr_livret = fr_livret.rename(
    columns={
        "Livret A": "Interest Rate (Livret A)",
    }
)


# %%
dataframes = [
    # fr_cpi,
    fr_inf,
    fr_exp,
    # fr_food_cons,
    # fr_goods_cons,
    # fr_dur_cons,
    fr_exp_sav,
    fr_per_sav,
]
fr_data = helpers.combine_series(dataframes, on=[ids.DATE], how="left")
fr_data[ids.DATE] = fr_data[ids.DATE].dt.to_period("M")

# %%
fr_data = fr_data.merge(fr_sav, how="left")
fr_data = fr_data.merge(fr_livret.sort_values("date"), how="left")

# %%
fr_data["pct_exp_savings"] = fr_data["exp_savings"].pct_change(periods=12, axis=0)
fr_data["pct_per_savings"] = fr_data["per_savings"].pct_change(periods=12, axis=0)
fr_data["pct_financial_savings"] = (
    fr_data["financial_savings"].pct_change(periods=1, axis=0) * 100
)
# Drop the single extreme negative outlier to avoid distorting the chart
if fr_data["pct_financial_savings"].notna().any():
    min_idx = fr_data["pct_financial_savings"].idxmin()
    fr_data.loc[min_idx, "pct_financial_savings"] = np.nan
fr_data["pct_financial_savings_10ma"] = (
    fr_data["pct_financial_savings"].rolling(window=10, center=False).mean()
)
fr_data["Real Interest Rate (Livret A)"] = (
    fr_data["Interest Rate (Livret A)"] - fr_data[ids.INFLATION]
)

COLS = [
    ids.DATE,
    ids.INFLATION,
    # "savings",
    # "financial_savings",
    "pct_financial_savings",
    # "pct_financial_savings_10ma",
    "Real Interest Rate (Livret A)",
    ids.EXPECTATIONS,
    # "pct_exp_savings",
    # "pct_per_savings",
]

fr_sliced = pd.concat([fr_data[COLS].dropna().head(), fr_data[COLS].dropna().tail()])
# fr_sliced

# %%
left_axis_cols = [
    ids.INFLATION,
    "financial_savings",
    "Real Interest Rate (Livret A)",
]
right_axis_cols = ["pct_financial_savings_10ma"]

fig, ax = plt.subplots(figsize=(10, 8))

fr_melt = pd.melt(
    fr_data[COLS].dropna(),
    [ids.DATE],
)
fr_melt = fr_melt.rename(columns={"value": "Measured (%)"})
# Ensure numeric y and datetime-like x for plotting
fr_melt["Measured (%)"] = pd.to_numeric(fr_melt["Measured (%)"], errors="coerce")
if pd.api.types.is_period_dtype(fr_melt[ids.DATE]):
    fr_melt[ids.DATE] = fr_melt[ids.DATE].dt.to_timestamp()
else:
    fr_melt[ids.DATE] = pd.to_datetime(fr_melt[ids.DATE], errors="coerce")
# Drop rows with invalid x or y
fr_melt = fr_melt.dropna(subset=[ids.DATE, "Measured (%)"])

sns.lineplot(
    data=fr_melt, x=ids.DATE, y="Measured (%)", hue="variable", style="variable", ax=ax
)
ax.hlines(
    y=0,
    xmin=fr_melt[ids.DATE].min(),
    xmax=fr_melt[ids.DATE].max(),
    color="black",
    alpha=0.5,
)
plt.show()

# %%
# Calculate average change in savings rate during phases of negative real interest rate
PHASE_1 = ("1968-01", "1985-01")
PHASE_2 = ("2017-01", "2024-01")

fr_data[(fr_data[ids.DATE] >= PHASE_1[0]) & (fr_data[ids.DATE] <= PHASE_1[1])][
    COLS
].describe()

# %%
fr_data[(fr_data[ids.DATE] >= PHASE_2[0]) & (fr_data[ids.DATE] <= PHASE_2[1])][
    COLS
].describe()

# %%
fr_data[
    ((fr_data[ids.DATE] >= PHASE_1[0]) & (fr_data[ids.DATE] <= PHASE_1[1]))
    | (fr_data[ids.DATE] >= PHASE_2[0]) & (fr_data[ids.DATE] <= PHASE_2[1])
][COLS].describe()

# %%
# * Create measured inflation dataframe
inf_data = pd.merge(
    us_data[[ids.DATE, ids.INFLATION, ids.EXPECTATIONS]],
    fr_data[[ids.DATE, ids.INFLATION, ids.EXPECTATIONS]],
    on=ids.DATE,
    suffixes=("_us", "_fr"),
)

inf_data.columns = [
    "Date",
    "Measured (US)",
    "Expectations (US)",
    "Measured (France)",
    "Expectations (France)",
]

# %%
inf_melt = pd.melt(inf_data, ["Date"])
inf_melt.rename(columns={"value": "Measured (%)"}, inplace=True)

# %% [markdown]
##### Figure XX - Time series: Measured Inflation (US and France)
# %%
_, (ax, bx) = plt.subplots(2, 1, sharex=True)

# * US subplot
measures_to_plot = ["Measured (US)", "Expectations (US)"]
data = inf_melt[inf_melt["variable"].isin(measures_to_plot)]
ax = sns.lineplot(data=data, x="Date", y="Measured (%)", hue="variable", ax=ax)
ax.legend().set_title(None)

# * French subplot
measures_to_plot = ["Measured (France)", "Expectations (France)"]
data = inf_melt[inf_melt["variable"].isin(measures_to_plot)]
bx = sns.lineplot(data=data, x="Date", y="Measured (%)", hue="variable", ax=bx)
bx.legend().set_title(None)
plt.suptitle("Inflation Rates, US and France")
plt.tight_layout()
