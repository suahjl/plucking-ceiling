# -------------- Allow PO to deviate from first guess based on K and N
# -------------- Casts the bands as 'uncertainty in timing of peaks by 1Q earlier'
# -------------- Option to show or hide confidence bands
# -------------- Option to not show any forecasts
# -------------- Open data version (est3)
# -------------- Follow dupraz nakamura steinsson
# -------------- Application to US data


import pandas as pd
import numpy as np
from datetime import date, timedelta
import statsmodels.formula.api as smf
import statsmodels.tsa.api as sm
from quantecon import hamilton_filter
import plotly.graph_objects as go
import plotly.express as px
import telegram_send
from ceic_api_client.pyceic import Ceic
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
T_lb = '1947Q1'
T_lb_day = date(1947, 1, 1)
use_forecast = ast.literal_eval(os.getenv('USE_FORECAST_BOOL'))
if use_forecast:
    file_suffix_fcast = '_forecast'
elif not use_forecast:
    file_suffix_fcast = ''

# gets 1997Q4, 2008Q1, 2019Q3; and 1998Q3, 2009Q1, 2020Q2 correct
downturn_threshold_choice = 0.23


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


def ceic2pandas_ts(input, start_date):  # input should be a list of CEIC Series IDs
    for m in range(len(input)):
        try:
            # brute force remove all np.nans from series ID list
            input.remove(np.nan)
        except:
            print('no more np.nan')
    k = 1
    for i in tqdm(input):
        series_result = Ceic.series(
            i, start_date=start_date)  # retrieves ceicseries
        y = series_result.data
        series_name = y[0].metadata.name  # retrieves name of series
        # this is a list of 1 dictionary,
        time_points_dict = dict((tp._date, tp.value)
                                for tp in y[0].time_points)
        # convert into pandas series indexed to timepoints
        series = pd.Series(time_points_dict)
        if k == 1:
            frame_consol = pd.DataFrame(series)
            frame_consol = frame_consol.rename(columns={0: series_name})
        elif k > 1:
            frame = pd.DataFrame(series)
            frame = frame.rename(columns={0: series_name})
            # left-right concat on index (time)
            frame_consol = pd.concat([frame_consol, frame], axis=1)
        elif k < 1:
            raise NotImplementedError
        k += 1
        frame_consol = frame_consol.sort_index()
    return frame_consol


# II --- Load data
# Pull from CEIC
Ceic.login(os.getenv('CEIC_USERNAME'), os.getenv('CEIC_PASSWORD'))
seriesids = [459402067]  # only GDP
df = ceic2pandas_ts(input=seriesids, start_date=T_lb_day)  # US data
df = df.rename(columns={'Gross Domestic Product: Real: sa': 'gdp'})
df = df.reset_index().rename(columns={'index': 'quarter'})
df['quarter'] = pd.to_datetime(df['quarter']).dt.to_period('Q')
df['quarter'] = df['quarter'].astype('str')
# Transform
# Take logs
list_col = ['gdp']  # , 'lforce', 'nks'
list_col_ln = ['ln_' + i for i in list_col]
for i, j in zip(list_col, list_col_ln):
    df.loc[:, j] = np.log(df[i])

# Take log-difference
list_col_ln_diff = [i + '_diff' for i in list_col_ln]
for i, j in zip(list_col_ln, list_col_ln_diff):
    df.loc[:, j] = df[i] - df[i].shift(1)

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
    cols_is_rate = [False for i in cols_levels]

    # Reference column labels
    ref_levels = ref_level_label
    ref_diff = ref_levels + '_diff'
    ref_trough = ref_levels + '_trough'
    ref_peak = ref_levels + '_peak'
    ref_epi = ref_levels + '_epi'
    ref_cepi = ref_levels + '_cepi'
    ref_pace = ref_levels + '_pace'
    ref_ceiling = ref_levels + '_ceiling'

    # Base list of peaks and troughs
    list_peaks = []
    list_troughs = []

    # Store list of quarters
    list_time = [str(i) for i in list(df['quarter'])]
    list_indices = [i for i in list(df.index)]

    # Counter for if lower bound or point estimate is being calculated
    count_xbound_now = 0

    # Loop through list of timestamps
    while count_xbound_now <= 1:
        for col_level, \
            col_peak, col_trough, \
            col_epi, col_cepi, col_pace, \
            col_diff, col_ceiling, col_is_rate \
                in \
                zip(
                    cols_levels,
                    cols_peak, cols_trough,
                    cols_epi, cols_cepi, cols_pace,
                    cols_diff, cols_ceiling, cols_is_rate
                ):

            # Add lb as suffix if estimating xbound
            if count_xbound_now == 1:
                col_trough = col_trough + '_lb'
                col_peak = col_peak + '_lb'
                col_epi = col_epi + '_lb'
                col_cepi = col_cepi + '_lb'
                col_pace = col_pace + '_lb'
                col_ceiling = col_ceiling + '_lb'

            # Compute downturn threshold using standard deviation of logdiff
            if col_is_rate:
                threshold_for_this_col = np.std(
                    df[col_level]) * downturn_threshold
            elif not col_is_rate:
                threshold_for_this_col = np.std(
                    df[col_diff]) * downturn_threshold

            # Initialise parameters
            t_cp = 0  # peak candidate
            t_ct = 0  # trough candidate
            t_next = t_cp + 1  # initial time stamp
            just_found_peak = False
            just_found_trough = False
            stuck_in_step_one = True
            stuck_in_step_two = True
            stuck_in_step_five = True
            stuck_in_step_six = True

            # Define all requisite steps as functions

            def step_one(just_found_trough: bool,
                         t_cp, t_ct, t_next):
                if just_found_trough:
                    t_cp = t_ct + 1
                    t_next = t_cp + 1
                elif not just_found_trough:
                    t_cp = t_cp + 1
                    t_next = t_next + 1
                return t_cp, t_next

            def step_two(df: pd.DataFrame, t_cp, t_next):
                go_back = (
                    df.loc[df.index == t_cp, col_level].values[0]
                    <
                    df.loc[df.index == t_next, col_level].values[0]
                )
                return go_back

            def step_three(df: pd.DataFrame, is_rate, t_next):
                # only sensible for rates
                if is_rate:
                    go_back = (
                        df.loc[df.index == t_cp, col_level].values[0]
                        <
                        df.loc[df.index == t_next, col_level].values[0] +
                        threshold_for_this_col
                    )
                    t_next = t_next + 1  # without changing t_cp
                # adapted to non-rate data
                elif not is_rate:
                    go_back = (
                        df.loc[df.index == t_next, col_diff].values[0]
                        <
                        0 + threshold_for_this_col
                    )
                    t_next = t_next + 1  # without changing t_cp
                return go_back, t_next

            def step_four(t_cp, list_peaks):
                list_peaks = list_peaks + [list_time[t_cp]]
                just_found_peak = True
                return list_peaks, just_found_peak

            def step_five(
                    just_found_peak: bool,
                    t_cp, t_ct, t_next
            ):
                if just_found_peak:
                    t_ct = t_cp + 1
                    t_next = t_ct + 1
                elif not just_found_peak:
                    t_ct = t_ct + 1
                    t_next = t_next + 1
                return t_ct, t_next

            def step_six(df: pd.DataFrame, t_ct, t_next):
                go_back = (
                    df.loc[df.index == t_ct, col_level].values[0]
                    >
                    df.loc[df.index == t_next, col_level].values[0]
                )
                return go_back

            def step_seven(df: pd.DataFrame, is_rate, t_next):
                # only sensible for rates
                if is_rate:
                    go_back = (
                        df.loc[df.index == t_ct, col_level].values[0]
                        >
                        df.loc[df.index == t_next, col_level].values[0] -
                        threshold_for_this_col
                    )
                    t_next = t_next + 1  # without changing t_cp
                # adapted to non-rate data
                elif not is_rate:
                    go_back = (
                        df.loc[df.index == t_next, col_diff].values[0]
                        >
                        0 - threshold_for_this_col
                    )
                    t_next = t_next + 1  # without changing t_cp
                return go_back, t_next

            def step_eight(t_ct, list_troughs):
                list_troughs = list_troughs + [list_time[t_ct]]
                just_found_trough = True
                return list_troughs, just_found_trough

            while t_next <= list_indices[-1]:
                try:
                    # FIND PEAK
                    # step one and two
                    while stuck_in_step_one:
                        # print('Step 1: t_next = ' + list_time[t_next] + ' for ' + col_level)
                        t_cp, t_next = step_one(just_found_trough=just_found_trough, t_cp=t_cp, t_ct=t_ct,
                                                t_next=t_next)
                        just_found_trough = False  # only allow just_found_trough to be true once per loop
                        stuck_in_step_one = step_two(
                            df=df, t_cp=t_cp, t_next=t_next)
                    stuck_in_step_one = True  # reset so loop will run again
                    # step three
                    while stuck_in_step_two:
                        # print('Step 2-3: t_next = ' + list_time[t_next] + ' for ' + col_level)
                        stuck_in_step_two, t_next = step_three(df=df, is_rate=col_is_rate,
                                                               t_next=t_next)  # if true, skips next line
                        restuck_in_step_one = step_two(
                            df=df, t_cp=t_cp, t_next=t_next)
                        while restuck_in_step_one:  # if step 3 is executed, but then fails step 2, so back to step 1
                            # print('Back to step 1-2: t_next = ' + list_time[t_next] + ' for ' + col_level)
                            t_cp, t_next = step_one(just_found_trough=just_found_trough, t_cp=t_cp, t_ct=t_ct,
                                                    t_next=t_next)
                            restuck_in_step_one = step_two(
                                df=df, t_cp=t_cp, t_next=t_next)
                    stuck_in_step_two = True  # reset so loop will run again
                    # step four
                    # print('Step 4: t_cp = ' + list_time[t_cp] + ' for ' + col_level)
                    list_peaks, just_found_peak = step_four(
                        t_cp=t_cp, list_peaks=list_peaks)  # we have a peak
                    just_found_peak = True  # voila

                    # FIND TROUGH
                    # step five and six (equivalent to one and two)
                    while stuck_in_step_five:
                        # print('Step 5: t_next = ' + list_time[t_next] + ' for ' + col_level)
                        t_ct, t_next = step_five(
                            just_found_peak=just_found_peak, t_cp=t_cp, t_ct=t_ct, t_next=t_next)
                        just_found_peak = False  # only allow just_found_peak to be true once per loop
                        stuck_in_step_five = step_six(
                            df=df, t_ct=t_ct, t_next=t_next)
                    stuck_in_step_five = True  # reset so loop will run again
                    # step seven (equivalent to three)
                    while stuck_in_step_six:
                        # print('Step 6-7: t_next = ' + list_time[t_next] + ' for ' + col_level)
                        stuck_in_step_six, t_next = step_seven(df=df, is_rate=col_is_rate,
                                                               t_next=t_next)  # if true, skips next line
                        restuck_in_step_five = step_six(
                            df=df, t_ct=t_ct, t_next=t_next)
                        while restuck_in_step_five:
                            # print('Back to step 5-6: t_next = ' + list_time[t_next] + ' for ' + col_level)
                            t_ct, t_next = step_five(just_found_peak=just_found_peak, t_cp=t_cp, t_ct=t_ct,
                                                     t_next=t_next)
                            restuck_in_step_five = step_six(
                                df=df, t_ct=t_ct, t_next=t_next)
                    stuck_in_step_six = True  # reset so loop will run again
                    # step eight (equivalent to four)
                    # print('Step 8: t_ct = ' + list_time[t_ct] + ' for ' + col_level)
                    list_troughs, just_found_trough = step_eight(t_ct=t_ct,
                                                                 list_troughs=list_troughs)  # we have a trough
                except:
                    pass

            # Check
            print('Peaks in ' + col_level + ': ' + ', '.join(list_peaks))
            print('Troughs in ' + col_level + ': ' + ', '.join(list_troughs))

            # Add columns indicating peaks and troughs
            df.loc[df['quarter'].isin(list_peaks), col_peak] = 1
            df[col_peak] = df[col_peak].fillna(0)
            df.loc[df['quarter'].isin(list_troughs), col_trough] = 1
            df[col_trough] = df[col_trough].fillna(0)

            # For Xbounds
            if count_xbound_now == 1:
                df[col_peak] = df[col_peak].shift(
                    bounds_timing_shift)  # move back by h horizons
                df[col_peak] = df[col_peak].fillna(0)

            # Episodes
            df[col_epi] = (df[col_trough] + df[col_peak]).cumsum()
            df.loc[((df[col_trough] == 1) | (df[col_peak] == 1)),
                   col_epi] = df[col_epi] - 1

            # Peak-to-peak episodes
            df[col_cepi] = df[col_peak].cumsum()
            df.loc[df[col_peak] == 1, col_cepi] = df[col_cepi] - \
                1  # exp start after trough, and end at peaks

            # Calculate average episodic pace
            df[col_pace] = df.groupby(col_epi)[col_diff].transform('mean')
            tab = df.groupby(col_epi)[col_diff].agg('mean').reset_index()
            print(tab)

            # Compute ceiling
            # Check if single expansion
            single_exp = bool(df[col_epi].max() == 0)
            if not single_exp:
                # interpolate
                # peaks as joints
                df.loc[df[col_peak] == 1, col_ceiling] = df[col_level]
                df[col_ceiling] = df[col_ceiling].interpolate(
                    method='quadratic')  # too sparse for cubic

                # end-point extrapolation
                cepi_minusone = df[col_cepi].max() - 1
                df['_x'] = df[col_ceiling] - df[col_ceiling].shift(1)
                ceiling_minusone_avgdiff = (
                    df.loc[df[col_cepi] == cepi_minusone, '_x']).mean()
                del df['_x']
                nrows_na = len(df.isna())
                for r in tqdm(range(nrows_na)):
                    df.loc[df[col_ceiling].isna(), col_ceiling] = df[col_ceiling].shift(
                        1) + ceiling_minusone_avgdiff

                # start-point extrapolation
                df['_x'] = df[col_ceiling] - df[col_ceiling].shift(1)
                ceiling_one_avgdiff = (df.loc[df[col_cepi] == 1, '_x']).mean()
                del df['_x']
                nrows_na = len(df.isna())
                for r in tqdm(range(nrows_na)):
                    df.loc[df[col_ceiling].isna(), col_ceiling] = df[col_ceiling].shift(
                        -1) - ceiling_one_avgdiff  # reverse
            if single_exp:
                df[col_peak] = df[ref_peak].copy()
                df[col_cepi] = df[ref_cepi].copy()

                # interpolate
                # peaks as joints
                df.loc[df[col_peak] == 1, col_ceiling] = df[col_level]
                df[col_ceiling] = df[col_ceiling].interpolate(
                    method='quadratic')  # too sparse for cubic

                # end-point extrapolation
                cepi_minusone = df[col_cepi].max() - 1
                df['_x'] = df[col_ceiling] - df[col_ceiling].shift(1)
                ceiling_minusone_avgdiff = (
                    df.loc[df[col_cepi] == cepi_minusone, '_x']).mean()
                del df['_x']
                nrows_na = len(df.isna())
                for r in tqdm(range(nrows_na)):
                    df.loc[df[col_ceiling].isna(), col_ceiling] = df[col_ceiling].shift(
                        1) + ceiling_minusone_avgdiff

                # start-point extrapolation
                df['_x'] = df[col_ceiling] - df[col_ceiling].shift(1)
                ceiling_one_avgdiff = (df.loc[df[col_cepi] == 1, '_x']).mean()
                del df['_x']
                nrows_na = len(df.isna())
                for r in tqdm(range(nrows_na)):
                    df.loc[df[col_ceiling].isna(), col_ceiling] = df[col_ceiling].shift(
                        -1) - ceiling_one_avgdiff  # reverse

                # hard-impose definition of 'ceiling'
            if hard_bound | (col_level == 'ln_lforce') | (col_level == 'ln_nks'):
                df.loc[df[col_ceiling] < df[col_level], col_ceiling] = df[
                    col_level]  # replace with levels if first guess is lower

        # Left-right merge + bounds
        if count_xbound_now == 0:
            df_consol = df.copy()
        elif count_xbound_now == 1:
            df_consol = df_consol.combine_first(df)  # with xbounds

        # Indicator to trigger computation of xbounds
        count_xbound_now += 1

    # Output
    return df


list_ln = ['ln_gdp']  # 'ln_lforce', 'ln_nks'
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


# IV --- Allow ceiling to vary with K and N

def update_ceiling(data, option, hard_bound):
    d = data.copy()

    d['ln_gdp_ceiling_initial'] = d['ln_gdp_ceiling'].copy()  # for reference
    d['ln_gdp_ceiling_initial_lb'] = d['ln_gdp_ceiling_lb'].copy()  # for reference

    # 04jan2023: Ygap = Y0gap + alpha(deltaKgap) + (1-alpha)(deltaNgap)
    # if option == 'delta_gap':
    #     d['ln_gdp_ceiling'] = \
    #         d['ln_gdp_ceiling_initial'] + \
    #         (d['alpha'] / 100) * (d['ln_nks_ceiling'] - d['ln_nks_ceiling'].shift(1)) + \
    #         (1 - d['alpha'] / 100) * (d['ln_lforce_ceiling'] - d['ln_lforce_ceiling'].shift(1))
    #     d['ln_gdp_ceiling_lb'] = \
    #         d['ln_gdp_ceiling_initial_lb'] + \
    #         (d['alpha'] / 100) * (d['ln_nks_ceiling_lb'] - d['ln_nks_ceiling_lb'].shift(1)) + \
    #         (1 - d['alpha'] / 100) * (d['ln_lforce_ceiling_lb'] - d['ln_lforce_ceiling_lb'].shift(1))
    #     d_short = d.copy()
    #     d_short['ln_k_rev'] = \
    #         (d['alpha'] / 100) * (d['ln_nks_ceiling'] - d['ln_nks_ceiling'].shift(1))
    #     d_short['ln_n_rev'] = \
    #         (1 - d['alpha'] / 100) * (d['ln_lforce_ceiling'] - d['ln_lforce_ceiling'].shift(1))
    #     d_short['k_rev'] = \
    #         ((np.exp(d['ln_gdp_ceiling']) / np.exp(d['ln_gdp_ceiling_initial'])) /
    #          (np.exp(d_short['ln_n_rev']) ** (1 - d['alpha'] / 100)))  # Y1/Y2 * 1/e^(x1)
    #     d_short['n_rev'] = \
    #         ((np.exp(d['ln_gdp_ceiling']) / np.exp(d['ln_gdp_ceiling_initial'])) /
    #          (np.exp(d_short['ln_k_rev']) ** (d['alpha'] / 100)))

    # 04jan2023: Ygap = Y0gap + alpha(Kgap) + (1-alpha)(Ngap)
    # if option == 'gap':
    #     d['ln_gdp_ceiling'] = \
    #         d['ln_gdp_ceiling_initial'] + \
    #         (d['alpha'] / 100) * (d['ln_nks'] - d['ln_nks_ceiling']) + \
    #         (1 - d['alpha'] / 100) * (d['ln_lforce'] - d['ln_lforce_ceiling'])
    #     d['ln_gdp_ceiling_lb'] = \
    #         d['ln_gdp_ceiling_initial_lb'] + \
    #         (d['alpha'] / 100) * (d['ln_nks'] - d['ln_nks_ceiling_lb']) + \
    #         (1 - d['alpha'] / 100) * (d['ln_lforce'] - d['ln_lforce_ceiling_lb'])
    #     d_short = d.copy()
    #     d_short['ln_k_rev'] = \
    #         (d['alpha'] / 100) * (d['ln_nks'] - d['ln_nks_ceiling'])
    #     d_short['ln_n_rev'] = \
    #         (1 - d['alpha'] / 100) * (d['ln_lforce'] - d['ln_lforce_ceiling'])
    #     d_short['k_rev'] = \
    #         ((np.exp(d['ln_gdp_ceiling']) / np.exp(d['ln_gdp_ceiling_initial'])) /
    #          (np.exp(d['ln_lforce'] - d['ln_lforce_ceiling']) ** (1 - d['alpha'] / 100)))  # X2 = Y1/Y0 * 1/X1^(B1)
    #     d_short['n_rev'] = \
    #         ((np.exp(d['ln_gdp_ceiling']) / np.exp(d['ln_gdp_ceiling_initial'])) /
    #          (np.exp(d['ln_nks'] - d['ln_nks_ceiling']) ** (d['alpha'] / 100)))

    # Hardbound definition
    if hard_bound:
        d.loc[d['ln_gdp_ceiling'] < d['ln_gdp'],
              'ln_gdp_ceiling'] = d['ln_gdp']

    # Reduced DF for plotting
    d_short = d.copy()
    # d_short = d_short[['quarter', 'ln_gdp_ceiling', 'ln_gdp_ceiling_initial', 'ln_k_rev', 'ln_n_rev', 'k_rev', 'n_rev']]
    d_short = d_short[['quarter', 'ln_gdp_ceiling', 'ln_gdp_ceiling_initial']]

    # Convert reduced DF into levels
    # d_short['gdp_ceiling'] = np.exp(d_short['ln_gdp_ceiling'])
    d_short['gdp_ceiling_initial'] = np.exp(d_short['ln_gdp_ceiling_initial'])

    # Output
    return d, d_short


df, df_update_ceiling = update_ceiling(data=df, option='gap', hard_bound=True)
df_nobound, df_nobound_update_ceiling = update_ceiling(
    data=df_nobound, option='gap', hard_bound=False)


# V --- Compute production function decomposition of ceiling and actual output

def prodfunc(data):
    d = data.copy()

    # ACTUAL
    # d['implied_y'] = (d['alpha'] / 100) * d['ln_nks'] + (1 - d['alpha'] / 100) * d['ln_lforce']
    # d['ln_tfp'] = d['ln_gdp'] - d['implied_y']  # ln(tfp)

    # CEILING
    # Calculate TFP
    # d['ln_tfp_ceiling'] = d['ln_gdp_ceiling'] - \
    #                       ((d['alpha'] / 100) * d['ln_nks_ceiling']) - \
    #                       (1 - d['alpha'] / 100) * d['ln_lforce_ceiling']
    # d['ln_tfp_ceiling_lb'] = d['ln_gdp_ceiling_lb'] - \
    #                          ((d['alpha'] / 100) * d['ln_nks_ceiling_lb']) - \
    #                          (1 - d['alpha'] / 100) * d['ln_lforce_ceiling_lb']
    # Back out levels (PO)
    d['gdp_ceiling'] = np.exp(d['ln_gdp_ceiling'])
    d['gdp_ceiling_lb'] = np.exp(d['ln_gdp_ceiling_lb'])
    d['output_gap'] = 100 * (d['gdp'] / d['gdp_ceiling'] - 1)  # % PO
    d['output_gap_lb'] = 100 * (d['gdp'] / d['gdp_ceiling_lb'] - 1)  # % PO
    # d['capital_ceiling'] = np.exp(d['ln_nks_ceiling'])
    # d['capital_ceiling_lb'] = np.exp(d['ln_nks_ceiling_lb'])
    # d['labour_ceiling'] = np.exp(d['ln_lforce_ceiling'])
    # d['labour_ceiling_lb'] = np.exp(d['ln_lforce_ceiling_lb'])
    # d['tfp_ceiling'] = np.exp(d['ln_tfp_ceiling'])
    # d['tfp_ceiling_lb'] = np.exp(d['ln_tfp_ceiling_lb'])

    # Back out levels (observed output)
    # d['capital_observed'] = np.exp(d['ln_nks'])
    # d['labour_observed'] = np.exp(d['ln_lforce'])
    # d['tfp_observed'] = np.exp(d['ln_tfp'])

    # trim data frame
    list_col_keep = [
        'quarter',
        'output_gap',
        'output_gap_lb',
        'gdp_ceiling',
        'gdp_ceiling_lb',
        'gdp',
        # 'capital_ceiling',
        # 'capital_ceiling_lb',
        # 'labour_ceiling',
        # 'labour_ceiling_lb',
        # 'tfp_ceiling',
        # 'tfp_ceiling_lb',
        # 'capital_observed',
        # 'labour_observed',
        # 'tfp_observed',
        # 'alpha'
    ]
    d = d[list_col_keep]

    return d


# def prodfunc_histdecomp(input):
#     print('\n Uses output dataframe from prodfunc_po() ' +
#           'to calculate the historical decomposition of potential and actual output')
#     d = input.copy()

#     # Calculate YoY growth
#     list_levels = ['gdp_ceiling', 'gdp',
#                    'capital_ceiling', 'labour_ceiling', 'tfp_ceiling',
#                    'capital_observed', 'labour_observed', 'tfp_observed']
#     list_yoy = [i + '_yoy' for i in list_levels]
#     for i, j in zip(list_levels, list_yoy):
#         d[j] = 100 * ((d[i] / d[i].shift(4)) - 1)

#     # Decompose potential output growth
#     d['capital_cont_ceiling'] = d['capital_ceiling_yoy'] * (d['alpha'] / 100)
#     d['labour_cont_ceiling'] = d['labour_ceiling_yoy'] * (1 - (d['alpha'] / 100))
#     d['tfp_cont_ceiling'] = d['gdp_ceiling_yoy'] - d['capital_cont_ceiling'] - d['labour_cont_ceiling']

#     # Decompose observed output growth
#     d['capital_cont_observed'] = d['capital_observed_yoy'] * (d['alpha'] / 100)
#     d['labour_cont_observed'] = d['labour_observed_yoy'] * (1 - (d['alpha'] / 100))
#     d['tfp_cont_observed'] = d['gdp_yoy'] - d['capital_cont_observed'] - d['labour_cont_observed']

#     return d


df_pd = prodfunc(data=df)  # use labour force
# df_hd = prodfunc_histdecomp(input=df_pd)

# VI --- Export data frames
# post-update frame with bounds (useful for plotting logs, and ceilings of K and N)
df['quarter'] = df['quarter'].astype('str')
df.to_parquet('pluckingpo_dns_usa_estimates' +
              file_suffix_fcast + '.parquet', compression='brotli')
# post_update frame without bounds (useful for plotting logs)
df_nobound['quarter'] = df_nobound['quarter'].astype('str')
df_nobound.to_parquet('pluckingpo_dns_usa_estimates_nobounds' +
                      file_suffix_fcast + '.parquet', compression='brotli')
# update process
df_update_ceiling['quarter'] = df_update_ceiling['quarter'].astype('str')
df_update_ceiling.to_parquet('pluckingpo_dns_usa_updateceiling' + file_suffix_fcast +
                             '.parquet', compression='brotli')
# production function
df_pd['quarter'] = df_pd['quarter'].astype('str')
df_pd.to_parquet('pluckingpo_dns_usa_estimates_pf' + file_suffix_fcast + '.parquet', compression='brotli')  # take this to next script
# production function decomp
# df_hd['quarter'] = df_hd['quarter'].astype('str')
# df_hd.to_parquet('pluckingpo_dns_usa_estimates_pf_hd' + file_suffix_fcast + '.parquet', compression='brotli')

telsendmsg(conf=tel_config,
           msg='pluckingpo_compute_ceiling_dns_usa: COMPLETED')

# End
print('\n----- Ran in ' +
      "{:.0f}".format(time.time() - time_start) + ' seconds -----')
