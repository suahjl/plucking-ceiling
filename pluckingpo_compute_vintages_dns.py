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

list_T_ub = [
    '2007Q2', '2008Q2',
    '2009Q3', '2015Q4',
    '2019Q4', '2022Q4'
]
list_colours = [
    'lightcoral', 'crimson',
    'red', 'steelblue',
    'darkblue', 'gray'
]
list_dash_styles = [
    'solid', 'solid',
    'solid', 'solid',
    'solid', 'solid'
]
dict_revision_pairs = {
    '2009Q3': '2007Q2',
    '2019Q4': '2015Q4',
    '2022Q4': '2019Q4'
}
list_threshold = [
    0.23, 0.23,
    0.23, 0.23,
    0.23, 0.23
]  # 1, 1, 1, 1, 1, 0.8
list_interpolate_method = [
    'slinear', 'slinear',
    'slinear', 'slinear',
    'slinear', 'quadratic'
]  # harmonise for ease


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
df_full = pd.read_parquet('pluckingpo_input_data.parquet')
df_full['quarter'] = pd.to_datetime(df_full['quarter']).dt.to_period('Q')
df_full = df_full.reset_index(drop=True)  # so that we can work on numeric indices directly


# III --- Define functions for computation (only initial estimate + update + output gap; decomposition not needed)


def compute_ceilings(
        data,
        levels_labels, ref_level_label, downturn_threshold, bounds_timing_shift, hard_bound,
        interpolation_method):
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
                threshold_for_this_col = np.std(df[col_level]) * downturn_threshold
            elif not col_is_rate:
                threshold_for_this_col = np.std(df[col_diff]) * downturn_threshold

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
                            df.loc[df.index == t_next, col_level].values[0] + threshold_for_this_col
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
                            df.loc[df.index == t_next, col_level].values[0] - threshold_for_this_col
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
                        stuck_in_step_one = step_two(df=df, t_cp=t_cp, t_next=t_next)
                    stuck_in_step_one = True  # reset so loop will run again
                    # step three
                    while stuck_in_step_two:
                        # print('Step 2-3: t_next = ' + list_time[t_next] + ' for ' + col_level)
                        stuck_in_step_two, t_next = step_three(df=df, is_rate=col_is_rate,
                                                               t_next=t_next)  # if true, skips next line
                        restuck_in_step_one = step_two(df=df, t_cp=t_cp, t_next=t_next)
                        while restuck_in_step_one:  # if step 3 is executed, but then fails step 2, so back to step 1
                            # print('Back to step 1-2: t_next = ' + list_time[t_next] + ' for ' + col_level)
                            t_cp, t_next = step_one(just_found_trough=just_found_trough, t_cp=t_cp, t_ct=t_ct,
                                                    t_next=t_next)
                            restuck_in_step_one = step_two(df=df, t_cp=t_cp, t_next=t_next)
                    stuck_in_step_two = True  # reset so loop will run again
                    # step four
                    # print('Step 4: t_cp = ' + list_time[t_cp] + ' for ' + col_level)
                    list_peaks, just_found_peak = step_four(t_cp=t_cp, list_peaks=list_peaks)  # we have a peak
                    just_found_peak = True  # voila

                    # FIND TROUGH
                    # step five and six (equivalent to one and two)
                    while stuck_in_step_five:
                        # print('Step 5: t_next = ' + list_time[t_next] + ' for ' + col_level)
                        t_ct, t_next = step_five(just_found_peak=just_found_peak, t_cp=t_cp, t_ct=t_ct, t_next=t_next)
                        just_found_peak = False  # only allow just_found_peak to be true once per loop
                        stuck_in_step_five = step_six(df=df, t_ct=t_ct, t_next=t_next)
                    stuck_in_step_five = True  # reset so loop will run again
                    # step seven (equivalent to three)
                    while stuck_in_step_six:
                        # print('Step 6-7: t_next = ' + list_time[t_next] + ' for ' + col_level)
                        stuck_in_step_six, t_next = step_seven(df=df, is_rate=col_is_rate,
                                                               t_next=t_next)  # if true, skips next line
                        restuck_in_step_five = step_six(df=df, t_ct=t_ct, t_next=t_next)
                        while restuck_in_step_five:
                            # print('Back to step 5-6: t_next = ' + list_time[t_next] + ' for ' + col_level)
                            t_ct, t_next = step_five(just_found_peak=just_found_peak, t_cp=t_cp, t_ct=t_ct,
                                                     t_next=t_next)
                            restuck_in_step_five = step_six(df=df, t_ct=t_ct, t_next=t_next)
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
                df[col_peak] = df[col_peak].shift(bounds_timing_shift)  # move back by h horizons
                df[col_peak] = df[col_peak].fillna(0)

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
                if df[col_peak].sum() == 1:  # if only 1 peak is identified, force last obs to be another peak
                    df.iloc[-1, df.columns.get_loc(col_peak)] = 1  # only for vintage computation
                df.loc[df[col_peak] == 1, col_ceiling] = df[col_level]  # peaks as joints
                df[col_ceiling] = df[col_ceiling].interpolate(method=interpolation_method)  # too sparse for cubic

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
                    df.loc[df[col_ceiling].isna(), col_ceiling] = df[col_ceiling].shift(
                        -1) - ceiling_one_avgdiff  # reverse
            if single_exp:
                df[col_peak] = df[ref_peak].copy()
                df[col_cepi] = df[ref_cepi].copy()

                # interpolate
                df.loc[df[col_peak] == 1, col_ceiling] = df[col_level]  # peaks as joints
                df[col_ceiling] = df[col_ceiling].interpolate(method=interpolation_method)  # too sparse for cubic

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
    d_short = d_short[['quarter', 'ln_gdp_ceiling', 'ln_gdp_ceiling_initial', 'ln_k_rev', 'ln_n_rev', 'k_rev', 'n_rev']]

    # Convert reduced DF into levels
    d_short['gdp_ceiling'] = np.exp(d_short['ln_gdp_ceiling'])
    d_short['gdp_ceiling_initial'] = np.exp(d_short['ln_gdp_ceiling_initial'])

    # Output
    return d, d_short


def output_gap(data):
    d = data.copy()

    d['gdp_ceiling'] = np.exp(d['ln_gdp_ceiling'])
    d['gdp_ceiling_lb'] = np.exp(d['ln_gdp_ceiling_lb'])

    d['output_gap'] = 100 * (d['gdp'] / d['gdp_ceiling'] - 1)  # % PO
    d['output_gap_lb'] = 100 * (d['gdp'] / d['gdp_ceiling_lb'] - 1)  # % PO

    # trim data frame
    list_col_keep = ['quarter',
                     'output_gap', 'output_gap_lb',
                     'gdp_ceiling', 'gdp_ceiling_lb',
                     'gdp']
    d = d[list_col_keep]

    return d


# IV --- Computation
round = 1
for T_ub, interpolate_method, threshold in zip(list_T_ub, list_interpolate_method, list_threshold):
    # For tracking
    print('Currently ' + T_ub + ' vintage')

    # Generate vintage
    df = df_full[df_full['quarter'] <= T_ub]
    df['quarter'] = df['quarter'].astype('str')

    # Compute ceiling

    list_ln = ['ln_gdp', 'ln_lforce', 'ln_nks']
    df = compute_ceilings(
        data=df,
        levels_labels=list_ln,
        ref_level_label='ln_gdp',
        downturn_threshold=threshold,  # 0.65 to 0.8
        bounds_timing_shift=-1,
        hard_bound=True,
        interpolation_method=interpolate_method
    )

    df, df_rev = update_ceiling(
        data=df,
        option='gap',
        hard_bound=True
    )

    # Compute output gap

    df_og = output_gap(data=df)
    df_og = pd.DataFrame(df_og).rename(columns={'output_gap': T_ub})

    # Consolidate vintages
    if round == 1:
        df_final = df_og.copy()
    elif round > 1:
        df_final = df_final.merge(df_og, on='quarter', how='outer')
    round += 1

df_final = df_final[['quarter'] + list_T_ub]

# V --- Export data

df_final['quarter'] = df_final['quarter'].astype('str')
df_final.to_parquet('pluckingpo_dns_estimates_vintages.parquet', compression='brotli')

# VI --- Notify
telsendmsg(conf=tel_config,
           msg='pluckingpo_compute_vintages_dns: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
