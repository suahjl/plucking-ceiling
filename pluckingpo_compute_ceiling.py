# -------------- Allow PO to deviate from first guess based on K and N
# -------------- Casts the bands as 'uncertainty in timing of peaks by 1Q earlier'
# -------------- Option to show or hide confidence bands
# -------------- Option to not show any forecasts
# -------------- Open data version (est3)


import pandas as pd
import numpy as np
from datetime import date, timedelta
import statsmodels.formula.api as smf
import statsmodels.tsa.api as sm
from quantecon import hamilton_filter
import plotly.graph_objects as go
import plotly.express as px
import telegram_send
import dataframe_image as dfi
from PIL import Image
from tqdm import tqdm
import time
from dotenv import load_dotenv
import os
import ast

time_start = time.time()

# 0 --- Main settings
load_dotenv()
tel_config = os.getenv('TEL_CONFIG')
T_lb = '1995Q1'
T_lb_day = date(1995, 1, 1)
use_forecast = ast.literal_eval(os.getenv('USE_FORECAST_BOOL'))
if use_forecast:
    file_suffix_fcast = '_forecast'
elif not use_forecast:
    file_suffix_fcast = ''

# I --- Functions


def telsendimg(conf='', path='', cap=''):
    with open(path, 'rb') as f:
        telegram_send.send(conf=conf,
                           images=[f],
                           captions=[cap])


def telsendfiles(conf='', path='', cap=''):
    with open(path, 'rb') as f:
        telegram_send.send(conf=conf,
                           files=[f],
                           captions=[cap])


def telsendmsg(conf='', msg=''):
    telegram_send.send(conf=conf,
                       messages=[msg])

# II --- Load data
df = pd.read_parquet('pluckingpo_input_data' + file_suffix_fcast + '.parquet')
df['quarter'] = pd.to_datetime(df['quarter']).dt.to_period('Q')
df = df.set_index('quarter')

# III --- Initial estimate


def compute_ceilings(data, levels_labels, ref_level_label, downturn_threshold, bounds_timing_shift, hard_bound):

    # Deep copy
    d = data.copy()

    col_levels = levels_labels.copy()
    col_diff = [i + '_diff' for i in col_levels]
    col_trough = [i + '_trough' for i in col_levels]
    col_peak = [i + '_peak' for i in col_levels]
    col_epi = [i + '_epi' for i in col_levels]
    col_cepi = [i + '_cepi' for i in col_levels]  # ceiling episodes
    col_pace = [i + '_pace' for i in col_levels]
    col_ceiling = [i + '_ceiling' for i in col_levels]
    col_cpace = [i + '_cpace' for i in col_levels]

    ref_levels = ref_level_label
    ref_diff = ref_levels + '_diff'
    ref_trough = ref_levels + '_trough'
    ref_peak = ref_levels + '_peak'
    ref_epi = ref_levels + '_epi'
    ref_cepi = ref_levels + '_cepi'
    ref_pace = ref_levels + '_pace'
    ref_ceiling = ref_levels + '_ceiling'
    ref_cpace = ref_levels + '_cpace'

    # --------- Cycle over point est, and confidence bounds
    round = 1
    while round <= 2:

        print('Round: ' + str(round))

        # Peak-trough selection
        for diff, levels, trough, peak, epi, cepi, pace, ceiling, cpace \
                in \
                tqdm(zip(col_diff, col_levels,
                         col_trough, col_peak,
                         col_epi,
                         col_cepi,
                         col_pace,
                         col_ceiling,
                         col_cpace)):

            if round == 2:
                # Minus 1Q peak-peak timing
                trough = trough + '_lb'
                peak = peak + '_lb'
                epi = epi + '_lb'
                cepi = cepi + '_lb'
                pace = pace + '_lb'
                ceiling = ceiling + '_lb'
                cpace = cpace + '_lb'

            if round < 3:

                # Check unit roots in difference series
                # adf_pvalue = sm.adfuller(d[input].dropna())[1]
                # telsendmsg(conf=tel_config,
                #            msg=text_adf)

                # Check unit roots in second diff series
                # adf_pvalue = sm.adfuller((d[input] - d[input].shift(1)).dropna())[1]
                # telsendmsg(conf=tel_config,
                #            msg=text_adf)

                # Calculate standard deviation
                stdev = np.std(d[diff])
                factor_threshold = downturn_threshold  # adjust to include / exclude obvious episodes

                # if logdiff < 0, met with logdiff >= 0 in t+1, and abs(logdiff) >= stdev, then a trough
                d.loc[
                    (
                            (d[diff] < 0) &
                            (d[diff].shift(-1) >= 0) &
                            (np.abs(d[diff]) >= factor_threshold * stdev)
                    ),
                    trough
                ] = 1
                d[trough] = d[trough].fillna(0)

                # if logdiff > 0, met with logdiff < 0 in t+1, and trough within next 4 quarters (variable), then a peak
                d.loc[
                    (
                            (d[diff] >= 0) &
                            (d[diff].shift(-1) < 0) &
                            (
                                    (d[trough].shift(-1) == 1) |
                                    (d[trough].shift(-2) == 1) |
                                    (d[trough].shift(-3) == 1) |
                                    (d[trough].shift(-4) == 1)
                            )
                    )
                    , peak
                ] = 1
                d[peak] = d[peak].fillna(0)

                # Confidence bands
                if round == 2:
                    d[peak] = d[peak].shift(bounds_timing_shift)  # minus X
                    d[peak] = d[peak].fillna(0)

                # Episodes
                d[epi] = (d[trough] + d[peak]).cumsum()  # 0 = initial expansion, odd = gaps, even = expansion
                d.loc[((d[trough] == 1) | (d[peak] == 1)), epi] = d[epi] - 1  # exp start after trough, and end at peaks
                # d.loc[~((d[epi] % 2) == 0), epi] = np.nan
                # d[epi] = d[epi] / 2

                # Ceiling episodes, peak to peak
                d[cepi] = d[peak].cumsum()
                d.loc[d[peak] == 1, cepi] = d[cepi] - 1  # exp start after trough, and end at peaks

                # Calculate average episodic pace
                d[pace] = d.groupby(epi)[diff].transform('mean')
                tab = d.groupby(epi)[diff].agg('mean').reset_index()
                print(tab)
                # telsendmsg(conf=tel_config,
                #            msg=str(tab))

                # Compute 'ceiling'
                # Check if more than 1 expansion episodes
                single_exp = bool(d[epi].max() == 0)
                if not single_exp:
                    # interpolate
                    d.loc[d[peak] == 1, ceiling] = d[levels]  # peaks as joints
                    d = d.reset_index()
                    d[ceiling] = d[ceiling].interpolate(method='quadratic')  # too sparse for cubic
                    d = d.set_index('quarter')

                    # end-point extrapolation
                    cepi_minusone = d[cepi].max() - 1
                    d['_x'] = d[ceiling] - d[ceiling].shift(1)
                    ceiling_minusone_avgdiff = (d.loc[d[cepi] == cepi_minusone, '_x']).mean()
                    del d['_x']
                    nrows_na = len(d.isna())
                    for r in tqdm(range(nrows_na)):
                        d.loc[d[ceiling].isna(), ceiling] = d[ceiling].shift(1) + ceiling_minusone_avgdiff

                    # start-point extrapolation
                    d['_x'] = d[ceiling] - d[ceiling].shift(1)
                    ceiling_one_avgdiff = (d.loc[d[cepi] == 1, '_x']).mean()
                    del d['_x']
                    nrows_na = len(d.isna())
                    for r in tqdm(range(nrows_na)):
                        d.loc[d[ceiling].isna(), ceiling] = d[ceiling].shift(-1) - ceiling_one_avgdiff  # reverse

                elif single_exp:  # then follow GDP peaks and troughs
                    d[peak] = d[ref_peak].copy()
                    d[cepi] = d[ref_cepi].copy()

                    # interpolate
                    d.loc[d[peak] == 1, ceiling] = d[levels]  # peaks as joints
                    d = d.reset_index()
                    d[ceiling] = d[ceiling].interpolate(method='quadratic')  # too sparse for cubic
                    d = d.set_index('quarter')

                    # end-point extrapolation
                    cepi_minusone = d[cepi].max() - 1
                    d['_x'] = d[ceiling] - d[ceiling].shift(1)
                    ceiling_minusone_avgdiff = (d.loc[d[cepi] == cepi_minusone, '_x']).mean()
                    del d['_x']
                    nrows_na = len(d.isna())
                    for r in tqdm(range(nrows_na)):
                        d.loc[d[ceiling].isna(), ceiling] = d[ceiling].shift(1) + ceiling_minusone_avgdiff

                    # start-point extrapolation
                    d['_x'] = d[ceiling] - d[ceiling].shift(1)
                    ceiling_one_avgdiff = (d.loc[d[cepi] == 1, '_x']).mean()
                    del d['_x']
                    nrows_na = len(d.isna())
                    for r in tqdm(range(nrows_na)):
                        d.loc[d[ceiling].isna(), ceiling] = d[ceiling].shift(-1) - ceiling_one_avgdiff  # reverse

                # hard-impose definition of 'ceiling'
                if hard_bound | (levels == 'ln_lforce') | (levels == 'ln_nks'):
                    d.loc[d[ceiling] < d[levels], ceiling] = d[levels]  # replace with levels if first guess is lower

        # Left right merge + bounds
        if round == 1:
            d_consol = d.copy()  # initial copy
        elif round == 2:
            d_consol = d_consol.combine_first(d)  # minusX bound

        # indicator to trigger computation of confidence bands
        round += 1

    # Output
    return d_consol


list_ln = ['ln_gdp', 'ln_lforce', 'ln_nks']
df = compute_ceilings(
    data=df,
    levels_labels=list_ln,
    ref_level_label='ln_gdp',
    downturn_threshold=0.8,  # 0.65 to 0.8
    bounds_timing_shift=-1,
    hard_bound=True
)
df_nobound = compute_ceilings(
    data=df,
    levels_labels=list_ln,
    ref_level_label='ln_gdp',
    downturn_threshold=0.8,  # 0.65 to 0.8
    bounds_timing_shift=-1,
    hard_bound=False
)

# IV --- Allow ceiling to vary with K and N (04JAN2023: k_rev AND n_rev NEED ATTENTION)

def update_ceiling(data, option, hard_bound):
    d = data.copy()

    d['ln_gdp_ceiling_initial'] = d['ln_gdp_ceiling'].copy()  # for reference
    d['ln_gdp_ceiling_initial_lb'] = d['ln_gdp_ceiling_lb'].copy()  # for reference

    # 04jan2023: Ygap = Y0gap + alpha(deltaKgap) + (1-alpha)(deltaNgap)
    if option == 'delta_gap':
        d['ln_gdp_ceiling'] = \
            d['ln_gdp_ceiling_initial'] + \
            (d['alpha'] / 100) * (d['ln_nks_ceiling'] - d['ln_nks_ceiling'].shift(1)) + \
            (1 - d['alpha'] / 100) * (d['ln_lforce_ceiling'] - d['ln_lforce_ceiling'].shift(1))
        d['ln_gdp_ceiling_lb'] = \
            d['ln_gdp_ceiling_initial_lb'] + \
            (d['alpha'] / 100) * (d['ln_nks_ceiling_lb'] - d['ln_nks_ceiling_lb'].shift(1)) + \
            (1 - d['alpha'] / 100) * (d['ln_lforce_ceiling_lb'] - d['ln_lforce_ceiling_lb'].shift(1))
        d_short = d.copy()
        d_short['ln_k_rev'] = \
            (d['alpha'] / 100) * (d['ln_nks_ceiling'] - d['ln_nks_ceiling'].shift(1))
        d_short['ln_n_rev'] = \
            (1 - d['alpha'] / 100) * (d['ln_lforce_ceiling'] - d['ln_lforce_ceiling'].shift(1))
        d_short['k_rev'] = \
            ((np.exp(d['ln_gdp_ceiling']) / np.exp(d['ln_gdp_ceiling_initial'])) /
             (np.exp(d_short['ln_n_rev']) ** (1 - d['alpha'] / 100)))  # Y1/Y2 * 1/e^(x1)
        d_short['n_rev'] = \
            ((np.exp(d['ln_gdp_ceiling']) / np.exp(d['ln_gdp_ceiling_initial'])) /
            (np.exp(d_short['ln_k_rev']) ** (d['alpha'] / 100)))

    # 04jan2023: Ygap = Y0gap + alpha(Kgap) + (1-alpha)(Ngap)
    if option == 'gap':
        d['ln_gdp_ceiling'] = \
            d['ln_gdp_ceiling_initial'] + \
            (d['alpha'] / 100) * (d['ln_nks'] - d['ln_nks_ceiling']) + \
            (1 - d['alpha'] / 100) * (d['ln_lforce'] - d['ln_lforce_ceiling'])
        d['ln_gdp_ceiling_lb'] = \
            d['ln_gdp_ceiling_initial_lb'] + \
            (d['alpha'] / 100) * (d['ln_nks'] - d['ln_nks_ceiling_lb']) + \
            (1 - d['alpha'] / 100) * (d['ln_lforce'] - d['ln_lforce_ceiling_lb'])
        d_short = d.copy()
        d_short['ln_k_rev'] = \
            (d['alpha'] / 100) * (d['ln_nks'] - d['ln_nks_ceiling'])
        d_short['ln_n_rev'] = \
            (1 - d['alpha'] / 100) * (d['ln_lforce'] - d['ln_lforce_ceiling'])
        d_short['k_rev'] = \
            ((np.exp(d['ln_gdp_ceiling']) / np.exp(d['ln_gdp_ceiling_initial'])) /
            (np.exp(d['ln_lforce'] - d['ln_lforce_ceiling']) ** (1 - d['alpha'] / 100)))  # X2 = Y1/Y0 * 1/X1^(B1)
        d_short['n_rev'] = \
            ((np.exp(d['ln_gdp_ceiling']) / np.exp(d['ln_gdp_ceiling_initial'])) /
            (np.exp(d['ln_nks'] - d['ln_nks_ceiling']) ** (d['alpha'] / 100)))

    # Hardbound definition
    if hard_bound:
        d.loc[d['ln_gdp_ceiling'] < d['ln_gdp'], 'ln_gdp_ceiling'] = d['ln_gdp']

    # Reduced DF for plotting
    d_short = d_short[['ln_gdp_ceiling', 'ln_gdp_ceiling_initial', 'ln_k_rev', 'ln_n_rev', 'k_rev', 'n_rev']]

    # Convert reduced DF into levels
    d_short['gdp_ceiling'] = np.exp(d_short['ln_gdp_ceiling'])
    d_short['gdp_ceiling_initial'] = np.exp(d_short['ln_gdp_ceiling_initial'])

    # Output
    return d, d_short

df, df_update_ceiling = update_ceiling(data=df, option='gap', hard_bound=True)
df_nobound, df_nobound_update_ceiling = update_ceiling(data=df_nobound, option='gap', hard_bound=False)

# V --- Compute production function decomposition of ceiling and actual output


def prodfunc(data):
    d = data.copy()

    # ACTUAL
    d['implied_y'] = (d['alpha'] / 100) * d['ln_nks'] + (1 - d['alpha'] / 100) * d['ln_lforce']
    d['ln_tfp'] = d['ln_gdp'] - d['implied_y']  # ln(tfp)

    # CEILING
    # Calculate TFP
    d['ln_tfp_ceiling'] = d['ln_gdp_ceiling'] - \
                          ((d['alpha'] / 100) * d['ln_nks_ceiling']) - \
                          (1 - d['alpha'] / 100) * d['ln_lforce_ceiling']
    d['ln_tfp_ceiling_lb'] = d['ln_gdp_ceiling_lb'] - \
                          ((d['alpha'] / 100) * d['ln_nks_ceiling_lb']) - \
                          (1 - d['alpha'] / 100) * d['ln_lforce_ceiling_lb']
    # Back out levels (PO)
    d['gdp_ceiling'] = np.exp(d['ln_gdp_ceiling'])
    d['gdp_ceiling_lb'] = np.exp(d['ln_gdp_ceiling_lb'])
    d['output_gap'] = 100 * (d['gdp'] / d['gdp_ceiling'] - 1)  # % PO
    d['output_gap_lb'] = 100 * (d['gdp'] / d['gdp_ceiling_lb'] - 1) # % PO
    d['capital_ceiling'] = np.exp(d['ln_nks_ceiling'])
    d['capital_ceiling_lb'] = np.exp(d['ln_nks_ceiling_lb'])
    d['labour_ceiling'] = np.exp(d['ln_lforce_ceiling'])
    d['labour_ceiling_lb'] = np.exp(d['ln_lforce_ceiling_lb'])
    d['tfp_ceiling'] = np.exp(d['ln_tfp_ceiling'])
    d['tfp_ceiling_lb'] = np.exp(d['ln_tfp_ceiling_lb'])

    # Back out levels (observed output)
    d['capital_observed'] = np.exp(d['ln_nks'])
    d['labour_observed'] = np.exp(d['ln_lforce'])
    d['tfp_observed'] = np.exp(d['ln_tfp'])

    # trim data frame
    list_col_keep = ['output_gap', 'output_gap_lb',
                     'gdp_ceiling', 'gdp_ceiling_lb',
                     'gdp',
                     'capital_ceiling', 'capital_ceiling_lb',
                     'labour_ceiling', 'labour_ceiling_lb',
                     'tfp_ceiling', 'tfp_ceiling_lb',
                     'capital_observed', 'labour_observed', 'tfp_observed',
                     'alpha']
    d = d[list_col_keep]

    return d


def prodfunc_histdecomp(input):
    print('\n Uses output dataframe from prodfunc_po() ' +
          'to calculate the historical decomposition of potential and actual output')
    d = input.copy()

    # Calculate YoY growth
    list_levels = ['gdp_ceiling', 'gdp',
                   'capital_ceiling', 'labour_ceiling', 'tfp_ceiling',
                   'capital_observed', 'labour_observed', 'tfp_observed']
    list_yoy = [i + '_yoy' for i in list_levels]
    for i, j in zip(list_levels, list_yoy):
        d[j] = 100 * ((d[i] / d[i].shift(4)) - 1)

    # Decompose potential output growth
    d['capital_cont_ceiling'] = d['capital_ceiling_yoy'] * (d['alpha'] / 100)
    d['labour_cont_ceiling'] = d['labour_ceiling_yoy'] * (1 - (d['alpha'] / 100))
    d['tfp_cont_ceiling'] = d['gdp_ceiling_yoy'] - d['capital_cont_ceiling'] - d['labour_cont_ceiling']

    # Decompose observed output growth
    d['capital_cont_observed'] = d['capital_observed_yoy'] * (d['alpha'] / 100)
    d['labour_cont_observed'] = d['labour_observed_yoy'] * (1 - (d['alpha'] / 100))
    d['tfp_cont_observed'] = d['gdp_yoy'] - d['capital_cont_observed'] - d['labour_cont_observed']

    return d


df_pd = prodfunc(data=df)  # use labour force
df_hd = prodfunc_histdecomp(input=df_pd)


# VI --- Export data frames
# post-update frame with bounds (useful for plotting logs, and ceilings of K and N)
df = df.reset_index()
df['quarter'] = df['quarter'].astype('str')
df.to_parquet('pluckingpo_estimates' + file_suffix_fcast + '.parquet', compression='brotli')
# post_update frame without bounds (useful for plotting logs)
df_nobound = df_nobound.reset_index()
df_nobound['quarter'] = df_nobound['quarter'].astype('str')
df_nobound.to_parquet('pluckingpo_estimates_nobounds' + file_suffix_fcast + '.parquet', compression='brotli')
# update process
df_update_ceiling = df_update_ceiling.reset_index()
df_update_ceiling['quarter'] = df_update_ceiling['quarter'].astype('str')
df_update_ceiling.to_parquet('pluckingpo_updateceiling' + file_suffix_fcast + '.parquet', compression='brotli')
# production function
df_pd = df_pd.reset_index()
df_pd['quarter'] = df_pd['quarter'].astype('str')
df_pd.to_parquet('pluckingpo_estimates_pf' + file_suffix_fcast + '.parquet', compression='brotli')
# production function decomp
df_hd = df_hd.reset_index()
df_hd['quarter'] = df_hd['quarter'].astype('str')
df_hd.to_parquet('pluckingpo_estimates_pf_hd' + file_suffix_fcast + '.parquet', compression='brotli')

telsendmsg(conf=tel_config,
           msg='pluckingpo_compute_ceiling: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
