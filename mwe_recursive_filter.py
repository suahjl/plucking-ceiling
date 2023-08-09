'''
These packages are required.

py -m pip install pandas
py -m pip install numpy
py -m pip install statsmodels
'''


import pandas as pd
import numpy as np
import statsmodels.tsa.api as sm

# Load data
df = pd.read_csv(
    "input.csv"
)  # there should be a time period column, followed by input columns in levels to compute gaps for

# Recursive filter
cols_levels = [
    "gdp",
    "pc",
    "gfcf",
]  # list of column labels to be filtered (change this manually)

cols_trend = [
    i + "_trend" for i in cols_levels
]  # to hold labels of columns to be created for trend components (e.g., gdp_trend)

use_hpfilter = True
use_bpfilter = False

burn_in_duration = 20  # number of time periods to estimate the initial run
for level, trend in zip(cols_levels, cols_trend):
    t_count = 0
    for t in list(df.index):
        if t_count < burn_in_duration:
            pass
        elif t_count >= burn_in_duration:
            # this is for hp filter
            if use_hpfilter:
                cycle, trend = sm.filters.hpfilter(
                    df.loc[(~df[level].isna()) & (df.index <= t), level], lamb=11200
                )
            # this should work for bandpass filter
            if use_bpfilter:
                cycle = sm.filters.bkfilter(
                    df.loc[(~df[level].isna()) & (df.index <= t), level], low=6, high=32, K=12
                )
                trend = df[level] - df[cycle]  # back out trend
            # the following block updates the trend estimate for the latest time period
            if t_count == burn_in_duration:
                df[trend] = trend
            elif t_count > burn_in_duration:
                df.loc[df[trend].isna(), trend] = trend  # fill in NA with new trend
        t_count += 1

# Compute gap
cols_gap = [i + "_gap" for i in cols_levels]
for level, trend, gap in zip(cols_levels, cols_trend, cols_gap):
    df[gap] = 100 * ((df[level] / df[trend]) - 1)

# Output data
df.to_csv('output.csv', index=False)
