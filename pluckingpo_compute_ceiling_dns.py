# -------------- Allow PO to deviate from first guess based on K and N
# -------------- Casts the bands as 'uncertainty in timing of peaks by 1Q earlier'
# -------------- Option to show or hide confidence bands
# -------------- Option to not show any forecasts
# -------------- Open data version (est3)
# -------------- Follow dupraz nakamura steinsson


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

downturn_threshold_choice = 0.2


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
    df = data.copy()

    # Current column labels
    cols_levels = levels_labels.copy()
    cols_diff = [i + '_diff' for i in cols_levels]
    cols_trough = [i + '_trough' for i in cols_levels]
    cols_peak = [i + '_peak' for i in cols_levels]
    cols_epi = [i + '_epi' for i in cols_levels]
    cols_cepi = [i + '_cepi' for i in cols_levels]  # ceiling episodes
    cols_pace = [i + '_pace' for i in cols_levels]
    cols_ceiling = [i + '_ceiling' for i in cols_levels]
    cols_cpace = [i + '_cpace' for i in cols_levels]

    # Reference column labels
    ref_levels = ref_level_label
    ref_diff = ref_levels + '_diff'
    ref_trough = ref_levels + '_trough'
    ref_peak = ref_levels + '_peak'
    ref_epi = ref_levels + '_epi'
    ref_cepi = ref_levels + '_cepi'
    ref_pace = ref_levels + '_pace'
    ref_ceiling = ref_levels + '_ceiling'
    ref_cpace = ref_levels + '_cpace'

    # Base list of peaks and troughs
    list_peaks = []
    list_troughs = []

    # Store list of indices
    list_time = [str(i) for i in list(df.index)]  # str is fine since this is periodindex
    n_list_time = len(list_time)

    # Loop through list of timestamps
    for col_level, col_peak, col_trough, col_epi, col_cepi, col_pace, col_epi, \
        col_diff, col_ceiling \
            in \
            zip(cols_levels, cols_peak, cols_trough, cols_epi, cols_cepi, cols_pace,
                cols_epi, cols_diff, cols_ceiling):
        # Compute downturn threshold using standard deviation of logdiff
        if col_level == 'urate':
            threshold_for_this_col = np.std(df[col_level]) * downturn_threshold
        else:
            threshold_for_this_col = np.std(df[col_diff]) * downturn_threshold

        # Initialise time count
        t_position = -1  # so that we start at t=0
        still_checking_for_trough = False  # so that initial check will run
        peak_checked = False
        while t_position < (n_list_time - 1):  # T+1 does not exist
            # CHECK FOR PEAK
            if not still_checking_for_trough:
                # Move on to next time count
                t_position += 1  # as long as we're checking for peak

                # Extract time label
                current_t = list_time[t_position]
                t_plus_one = list_time[t_position + 1]
                print('checking peak in ' + current_t)

                # Check if t+1 is worse than t
                go_back = (
                        df.loc[df.index == t_plus_one, col_level].values[0]
                        <
                        df.loc[df.index == current_t, col_level].values[0]
                )  # equivalent to logdiff being negative

                # Check if t+1 is higher than threshold
                if not go_back:
                    # only sensible for rates
                    if 'rate' in col_level:
                        go_back = (
                                df.loc[df.index == t_plus_one, col_level].values[0]
                                <
                                df.loc[df.index == current_t, col_level].values[0] + threshold_for_this_col
                        )
                    # adapted to non-rate data
                    else:
                        go_back = (
                                df.loc[df.index == t_plus_one, col_diff].values[0]
                                <
                                0 + threshold_for_this_col
                        )

                # Add to list of peaks
                if not go_back:
                    list_peaks = list_peaks + [current_t]
                    peak_checked = True
                    print('peak found in ' + current_t)

            # CHECK FOR TROUGH
            if peak_checked:
                still_checking_for_trough = True
            if still_checking_for_trough:
                # Move on to next time count
                t_position += 1  # as long as we're checking for trough

                # Re-extract time label
                if not go_back:
                    current_t = list_time[t_position]
                    t_plus_one = list_time[t_position + 1]
                    print('checking trough in ' + current_t)

                # Check if t+1 is better than t
                if not go_back:
                    go_back = (
                            df.loc[df.index == t_plus_one, col_level].values[0]
                            >
                            df.loc[df.index == current_t, col_level].values[0]
                    )

                # Check if t+1 is lower than threshold
                if not go_back:
                    # only sensible for rates
                    if 'rate' in col_level:
                        go_back = (
                                df.loc[df.index == t_plus_one, col_level].values[0]
                                >
                                df.loc[df.index == current_t, col_level].values[0] - threshold_for_this_col
                            # need to redo
                        )
                    # adapted to non-rate data
                    else:
                        go_back = (
                                df.loc[df.index == t_plus_one, col_diff].values[0]
                                >
                                0 - threshold_for_this_col
                        )

                # Add to list of troughs
                if not go_back:
                    list_troughs = list_troughs + [current_t]
                    still_checking_for_trough = False
                    peak_checked = False
                    print('trough found in ' + current_t)

        # Add columns indicating peaks and troughs
        df.loc[df.index.astype('str').isin(list_peaks), col_peak] = 1
        df[col_peak] = df[col_peak].fillna(0)
        df.loc[df.index.astype('str').isin(list_troughs), col_trough] = 1
        df[col_trough] = df[col_trough].fillna(0)

        # Episodes
        df[col_epi] = (df[col_trough] + df[col_peak]).cumsum()
        df.loc[((df[col_trough] == 1) | (df[col_peak] == 1)), col_epi] = df[col_epi] - 1

        # Peak-to-peak episodes
        df[col_cepi] = df[col_peak].cumsum()
        df.loc[df[col_peak] == 1, col_cepi] = df[col_cepi] - 1  # exp start after trough, and end at peaks

        # Calculate average episodic pace
        df[col_pace] = df.groupby(col_epi)[col_diff].transform('mean')
        tab = df.groupby(col_epi)[col_diff].agg('mean').reset_index()
        print(tab)

        # Compute ceiling
        # Check if single expansion
        single_exp = bool(df[col_epi].max() == 0)
        if not single_exp:
            # interpolate
            df.loc[df[col_peak] == 1, col_ceiling] = df[col_level]  # peaks as joints
            df = df.reset_index()
            df[col_ceiling] = df[col_ceiling].interpolate(method='quadratic')  # too sparse for cubic
            df = df.set_index('quarter')

            # end-point extrapolation
            cepi_minusone = df[col_cepi].max() - 1
            df['_x'] = df[col_ceiling] - df[col_ceiling].shift(1)
            ceiling_minusone_avgdiff = (df.loc[df[col_cepi] == cepi_minusone, '_x']).mean()
            del df['_x']
            nrows_na = len(df.isna())
            for r in tqdm(range(nrows_na)):
                df.loc[df[col_ceiling].isna(), col_ceiling] = df[col_ceiling].shift(1) + ceiling_minusone_avgdiff

            # start-point extrapolation
            df['_x'] = df[col_ceiling] - df[col_ceiling].shift(1)
            ceiling_one_avgdiff = (df.loc[df[col_cepi] == 1, '_x']).mean()
            del df['_x']
            nrows_na = len(df.isna())
            for r in tqdm(range(nrows_na)):
                df.loc[df[col_ceiling].isna(), col_ceiling] = df[col_ceiling].shift(-1) - ceiling_one_avgdiff  # reverse
        if single_exp:
            df[col_peak] = df[ref_peak].copy()
            df[col_cepi] = df[ref_cepi].copy()

            # interpolate
            df.loc[df[col_peak] == 1, col_ceiling] = df[col_level]  # peaks as joints
            df = df.reset_index()
            df[col_ceiling] = df[col_ceiling].interpolate(method='quadratic')  # too sparse for cubic
            df = df.set_index('quarter')

            # end-point extrapolation
            cepi_minusone = df[col_cepi].max() - 1
            df['_x'] = df[col_ceiling] - df[col_ceiling].shift(1)
            ceiling_minusone_avgdiff = (df.loc[df[col_cepi] == cepi_minusone, '_x']).mean()
            del df['_x']
            nrows_na = len(df.isna())
            for r in tqdm(range(nrows_na)):
                df.loc[df[col_ceiling].isna(), col_ceiling] = df[col_ceiling].shift(1) + ceiling_minusone_avgdiff

            # start-point extrapolation
            df['_x'] = df[col_ceiling] - df[col_ceiling].shift(1)
            ceiling_one_avgdiff = (df.loc[df[col_cepi] == 1, '_x']).mean()
            del df['_x']
            nrows_na = len(df.isna())
            for r in tqdm(range(nrows_na)):
                df.loc[df[col_ceiling].isna(), col_ceiling] = df[col_ceiling].shift(-1) - ceiling_one_avgdiff  # reverse

            # hard-impose definition of 'ceiling'
        if hard_bound | (col_level == 'ln_lforce') | (col_level == 'ln_nks'):
            df.loc[df[col_ceiling] < df[col_level], col_ceiling] = df[
                col_level]  # replace with levels if first guess is lower

    # Output
    return df


list_ln = ['ln_gdp', 'ln_lforce', 'ln_nks']
df = compute_ceilings(
    data=df,
    levels_labels=list_ln,
    ref_level_label='ln_gdp',
    downturn_threshold=downturn_threshold_choice,
    bounds_timing_shift=-1,
    hard_bound=True
)
df_nobound = compute_ceilings(
    data=df,
    levels_labels=list_ln,
    ref_level_label='ln_gdp',
    downturn_threshold=downturn_threshold_choice,
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
    d['output_gap_lb'] = 100 * (d['gdp'] / d['gdp_ceiling_lb'] - 1)  # % PO
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
df.to_parquet('pluckingpo_dns_estimates' + file_suffix_fcast + '.parquet', compression='brotli')
# post_update frame without bounds (useful for plotting logs)
df_nobound = df_nobound.reset_index()
df_nobound['quarter'] = df_nobound['quarter'].astype('str')
df_nobound.to_parquet('pluckingpo_dns_estimates_nobounds' + file_suffix_fcast + '.parquet', compression='brotli')
# update process
df_update_ceiling = df_update_ceiling.reset_index()
df_update_ceiling['quarter'] = df_update_ceiling['quarter'].astype('str')
df_update_ceiling.to_parquet('pluckingpo_dns_updateceiling' + file_suffix_fcast + '.parquet', compression='brotli')
# production function
df_pd = df_pd.reset_index()
df_pd['quarter'] = df_pd['quarter'].astype('str')
df_pd.to_parquet('pluckingpo_dns_estimates_pf' + file_suffix_fcast + '.parquet', compression='brotli')
# production function decomp
df_hd = df_hd.reset_index()
df_hd['quarter'] = df_hd['quarter'].astype('str')
df_hd.to_parquet('pluckingpo_dns_estimates_pf_hd' + file_suffix_fcast + '.parquet', compression='brotli')

telsendmsg(conf=tel_config,
           msg='pluckingpo_compute_ceiling_dns: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
