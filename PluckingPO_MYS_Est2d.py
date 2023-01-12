# -------------- Allow PO to deviate from first guess based on K and N
# -------------- Casts the bands as 'uncertainty in timing of peaks by 1Q earlier'
# -------------- Option to show or hide confidence bands
# -------------- Option to not show any forecasts

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
from ceic_api_client.pyceic import Ceic

time_start = time.time()

# 0 --- Main settings
tel_config = 'EcMetrics_Config_GeneralFlow.conf'
T_lb = '1995Q1'
T_ub = '2022Q2'
T_lb_day = date(1995, 1, 1)
T_forecast_start = '2022Q3'  # Start of forecast period
T_forecast_end = '2023Q4'  # End of forecast period
show_conf_bands = False
use_forecast = False  # public or internal use
Ceic.login("suahjinglian@bnm.gov.my", "dream1234")  # suahjinglian@bnm.gov.my

# I --- Functions


def ceic2pandas_ts(input, start_date):  # input should be a list of CEIC Series IDs
    for m in range(len(input)):
        try: input.remove(np.nan)  # brute force remove all np.nans from series ID list
        except: print('no more np.nan')
    k = 1
    for i in tqdm(input):
        series_result = Ceic.series(i, start_date=start_date)  # retrieves ceicseries
        y = series_result.data
        series_name = y[0].metadata.name  # retrieves name of series
        time_points_dict = dict((tp.date, tp.value) for tp in y[0].time_points)  # this is a list of 1 dictionary,
        series = pd.Series(time_points_dict)  # convert into pandas series indexed to timepoints
        if k == 1:
            frame_consol = pd.DataFrame(series)
            frame_consol = frame_consol.rename(columns={0: series_name})
        elif k > 1:
            frame = pd.DataFrame(series)
            frame = frame.rename(columns={0: series_name})
            frame_consol = pd.concat([frame_consol, frame], axis=1)  # left-right concat on index (time)
        elif k < 1:
            raise NotImplementedError
        k += 1
        frame_consol = frame_consol.sort_index()
    return frame_consol


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


def wrangle(
        data,
        trim_start=None,
        trim_end=None,
        seasonal_adj=True,
        log_transform=True,
        filter_using_hamilton=False
):
    # Trimming and renaming columns
    d = data.copy()
    d['quarter'] = pd.to_datetime(d['quarter']).dt.to_period('Q')

    # Timebound
    if trim_start is not None:
        d = d[d['quarter'] >= trim_start]
    if trim_end is not None:
        d = d[d['quarter'] <= trim_end]

    d = d.set_index('quarter')

    # Generate YoY growth on unseasonal-adjusted GDP
    d['gdp15_yoy'] = 100 * ((d['gdp15'] / d['gdp15'].shift(-4)) - 1)

    # Seasonal adjustment: only GDP, labour, and employment
    list_col = ['gdp15', 'labour', 'employment']
    if seasonal_adj:
        for i in list_col:
            sadj_res = sm.x13_arima_analysis(d[i])
            sadj_seasadj = sadj_res.seasadj
            d[i] = sadj_seasadj  # Ideally, use MYS-specific calendar effects

    # Take logs post-seasonal adjustment: now including capital stock
    list_col = list_col + ['k_stock15']
    list_col_ln = ['ln_' + i for i in list_col]
    if log_transform:
        for i, j in zip(list_col, list_col_ln):
            d[j] = np.log(d[i])

    # Take log-difference
    list_col_ln_diff = [i + '_diff' for i in list_col_ln]
    for i, j in zip(list_col_ln, list_col_ln_diff):
        d[j] = d[i] - d[i].shift(1)

    # filter trend
    list_col_ln_trend = [i + '_trend' for i in list_col_ln]
    list_col_ln_cycle = [i + '_cycle' for i in list_col_ln]
    if not filter_using_hamilton:
        for i, j, k in zip(list_col_ln, list_col_ln_trend, list_col_ln_cycle):
            cycle, trend = sm.filters.hpfilter(d[i], lamb=1600)
            d[j] = trend  # don't replace original with trend component
            d[k] = cycle  # don't replace original with cycle component
    elif filter_using_hamilton:
        for i, j, k in zip(list_col_ln, list_col_ln_trend, list_col_ln_cycle):
            cycle, trend = hamilton_filter(d[i], h=8, p=4)  # 2 years (2 for annual, 8 for quarter, 24 for monthly, ...)
            d[j] = trend  # don't replace original with trend component
            d[k] = cycle  # don't replace original with cycle component

    return d


# II --- Wrangling
# Base data
df_raw = pd.read_csv('testdata.txt', sep='|')
df = wrangle(
    data=df_raw,
    trim_start=T_lb,
    trim_end=T_forecast_end,
    seasonal_adj=True,
    log_transform=True,
    filter_using_hamilton=False
)
del df['labour']  # drop the NAIRU-adjusted labour stock
# CEIC data
df_ceic = ceic2pandas_ts(input=['357222087'], start_date=T_lb_day)
df_ceic = df_ceic.rename(columns={'Labour Force: Person th: Malaysia': 'lforce'})
df_ceic['quarter'] = pd.to_datetime(df_ceic.index).to_period('Q')
df_ceic = df_ceic.set_index('quarter')

# labour force
lforce_old = pd.read_csv('old_static_lforce.txt')
lforce_old['quarter'] = pd.to_datetime(lforce_old['quarter']).dt.to_period('Q')
lforce_old = lforce_old.set_index('quarter')
lforce_old = lforce_old.rename(columns={'lforce': 'lforce_old'})

# merge
df = pd.concat([df, df_ceic, lforce_old], axis=1)
df.loc[df['lforce'].isna(), 'lforce'] = df['lforce_old']
del df['lforce_old']
df = df.sort_index()

# forecasts
# labour force
lforce_lt_qoq_growth = 0.46  # 2017-22
forecast_run = len(df)
run = 1  # labour force (to be converted directly into labour input)
while run <= forecast_run:
    df.loc[df['lforce'].isna(), 'lforce'] = df['lforce'].shift(1) * (1 + (lforce_lt_qoq_growth / 100))
    run += 1
else:
    pass

# Additional wrangling
# logs, logdiff
for i in ['lforce']:
    df['ln_' + i] = np.log(df[i])
    df['ln_' + i + '_diff'] = df['ln_' + i] - df['ln_' + i].shift(1)


# III --- Compute ceiling (initial guess)


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
                if hard_bound | (levels == 'ln_lforce') | (levels == 'ln_employment') | (levels == 'ln_k_stock15'):
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


list_ln = ['ln_gdp15', 'ln_lforce', 'ln_employment', 'ln_k_stock15']
df = compute_ceilings(
    data=df,
    levels_labels=list_ln,
    ref_level_label='ln_gdp15',
    downturn_threshold=0.8,  # 0.65 to 0.8
    bounds_timing_shift=-1,
    hard_bound=True
)
df_nobound = compute_ceilings(
    data=df,
    levels_labels=list_ln,
    ref_level_label='ln_gdp15',
    downturn_threshold=0.8,  # 0.65 to 0.8
    bounds_timing_shift=-1,
    hard_bound=False
)


# IV.A --- Allow ceiling to vary with K and N (04JAN2023: k_rev AND n_rev NEED ATTENTION)

def update_ceiling(data, option, use_lforce=True):
    d = data.copy()

    d['ln_gdp15_ceiling_initial'] = d['ln_gdp15_ceiling'].copy()  # for reference
    d['ln_gdp15_ceiling_initial_lb'] = d['ln_gdp15_ceiling_lb'].copy()  # for reference

    # 04jan2023: Ygap = Y0gap + alpha(deltaKgap) + (1-alpha)(deltaNgap)
    if option == 'delta_gap':
        if use_lforce:
            d['ln_gdp15_ceiling'] = \
                d['ln_gdp15_ceiling_initial'] + \
                (d['alpha'] / 100) * (d['ln_k_stock15_ceiling'] - d['ln_k_stock15_ceiling'].shift(1)) + \
                (1 - d['alpha'] / 100) * (d['ln_lforce_ceiling'] - d['ln_lforce_ceiling'].shift(1))
            d['ln_gdp15_ceiling_lb'] = \
                d['ln_gdp15_ceiling_initial_lb'] + \
                (d['alpha'] / 100) * (d['ln_k_stock15_ceiling_lb'] - d['ln_k_stock15_ceiling_lb'].shift(1)) + \
                (1 - d['alpha'] / 100) * (d['ln_lforce_ceiling_lb'] - d['ln_lforce_ceiling_lb'].shift(1))
            d_short = d.copy()
            d_short['ln_k_rev'] = \
                (d['alpha'] / 100) * (d['ln_k_stock15_ceiling'] - d['ln_k_stock15_ceiling'].shift(1))
            d_short['ln_n_rev'] = \
                (1 - d['alpha'] / 100) * (d['ln_lforce_ceiling'] - d['ln_lforce_ceiling'].shift(1))
            d_short['k_rev'] = \
                ((np.exp(d['ln_gdp15_ceiling']) / np.exp(d['ln_gdp15_ceiling_initial'])) /
                 (np.exp(d_short['ln_n_rev']) ** (1 - d['alpha'] / 100)))  # Y1/Y2 * 1/e^(x1)
            d_short['n_rev'] = \
                ((np.exp(d['ln_gdp15_ceiling']) / np.exp(d['ln_gdp15_ceiling_initial'])) /
                (np.exp(d_short['ln_k_rev']) ** (d['alpha'] / 100)))


        elif not use_lforce:
            d['ln_gdp15_ceiling'] = \
                d['ln_gdp15_ceiling_initial'] + \
                (d['alpha'] / 100) * (d['ln_k_stock15_ceiling'] - d['ln_k_stock15_ceiling'].shift(1)) + \
                (1 - d['alpha'] / 100) * (d['ln_employment_ceiling'] - d['ln_employment_ceiling'].shift(1))
            d['ln_gdp15_ceiling_lb'] = \
                d['ln_gdp15_ceiling_initial_lb'] + \
                (d['alpha'] / 100) * (d['ln_k_stock15_ceiling_lb'] - d['ln_k_stock15_ceiling_lb'].shift(1)) + \
                (1 - d['alpha'] / 100) * (d['ln_employment_ceiling_lb'] - d['ln_employment_ceiling_lb'].shift(1))
            d_short = d.copy()
            d_short['ln_k_rev'] = \
                (d['alpha'] / 100) * (d['ln_k_stock15_ceiling'] - d['ln_k_stock15_ceiling'].shift(1))
            d_short['ln_n_rev'] = \
                (1 - d['alpha'] / 100) * (d['ln_employment_ceiling'] - d['ln_employment_ceiling'].shift(1))
            d_short['k_rev'] = \
                ((np.exp(d['ln_gdp15_ceiling']) / np.exp(d['ln_gdp15_ceiling_initial'])) /
                 (np.exp(d['ln_employment_ceiling'] - d['ln_employment_ceiling'].shift(1)) ** (1 - d['alpha'] / 100)))  # Y1/Y2 * 1/e^(x1)
            d_short['n_rev'] = \
                ((np.exp(d['ln_gdp15_ceiling']) / np.exp(d['ln_gdp15_ceiling_initial'])) /
                (np.exp(d['ln_k_stock15_ceiling'] - d['ln_k_stock15_ceiling'].shift(1)) ** (d['alpha'] / 100)))

    # 04jan2023: Ygap = Y0gap + alpha(Kgap) + (1-alpha)(Ngap)
    if option == 'gap':
        if use_lforce:
            d['ln_gdp15_ceiling'] = \
                d['ln_gdp15_ceiling_initial'] + \
                (d['alpha'] / 100) * (d['ln_k_stock15'] - d['ln_k_stock15_ceiling']) + \
                (1 - d['alpha'] / 100) * (d['ln_lforce'] - d['ln_lforce_ceiling'])
            d['ln_gdp15_ceiling_lb'] = \
                d['ln_gdp15_ceiling_initial_lb'] + \
                (d['alpha'] / 100) * (d['ln_k_stock15'] - d['ln_k_stock15_ceiling_lb']) + \
                (1 - d['alpha'] / 100) * (d['ln_lforce'] - d['ln_lforce_ceiling_lb'])
            d_short = d.copy()
            d_short['ln_k_rev'] = \
                (d['alpha'] / 100) * (d['ln_k_stock15'] - d['ln_k_stock15_ceiling'])
            d_short['ln_n_rev'] = \
                (1 - d['alpha'] / 100) * (d['ln_lforce'] - d['ln_lforce_ceiling'])
            d_short['k_rev'] = \
                ((np.exp(d['ln_gdp15_ceiling']) / np.exp(d['ln_gdp15_ceiling_initial'])) /
                (np.exp(d['ln_lforce'] - d['ln_lforce_ceiling']) ** (1 - d['alpha'] / 100)))  # X2 = Y1/Y0 * 1/X1^(B1)
            d_short['n_rev'] = \
                ((np.exp(d['ln_gdp15_ceiling']) / np.exp(d['ln_gdp15_ceiling_initial'])) /
                (np.exp(d['ln_k_stock15'] - d['ln_k_stock15_ceiling']) ** (d['alpha'] / 100)))

        elif not use_lforce:
            d['ln_gdp15_ceiling'] = \
                d['ln_gdp15_ceiling_initial'] + \
                (d['alpha'] / 100) * (d['ln_k_stock15'] - d['ln_k_stock15_ceiling']) + \
                (1 - d['alpha'] / 100) * (d['ln_employment'] - d['ln_employment_ceiling'])
            d['ln_gdp15_ceiling_lb'] = \
                d['ln_gdp15_ceiling_initial_lb'] + \
                (d['alpha'] / 100) * (d['ln_k_stock15'] - d['ln_k_stock15_ceiling_lb']) + \
                (1 - d['alpha'] / 100) * (d['ln_employment'] - d['ln_employment_ceiling_lb'])
            d_short = d.copy()
            d_short['ln_k_rev'] = \
                (d['alpha'] / 100) * (d['ln_k_stock15'] - d['ln_k_stock15_ceiling'])
            d_short['ln_n_rev'] = \
                (1 - d['alpha'] / 100) * (d['ln_employment'] - d['ln_employment_ceiling'])
            d_short['k_rev'] = \
                ((np.exp(d['ln_gdp15_ceiling']) / np.exp(d['ln_gdp15_ceiling_initial'])) /
                (np.exp(d['ln_employment'] - d['ln_employment_ceiling']) ** (1 - d['alpha'] / 100)))  # Y1/Y2 * 1/e^(x1)
            d_short['n_rev'] = \
                ((np.exp(d['ln_gdp15_ceiling']) / np.exp(d['ln_gdp15_ceiling_initial'])) /
                (np.exp(d['ln_k_stock15'] - d['ln_k_stock15_ceiling']) ** (d['alpha'] / 100)))

    # Reduced DF for plotting
    d_short = d_short[['ln_gdp15_ceiling', 'ln_gdp15_ceiling_initial', 'ln_k_rev', 'ln_n_rev', 'k_rev', 'n_rev']]

    # Convert reduced DF into levels
    d_short['gdp15_ceiling'] = np.exp(d_short['ln_gdp15_ceiling'])
    d_short['gdp15_ceiling_initial'] = np.exp(d_short['ln_gdp15_ceiling_initial'])

    # Output
    return d, d_short

df, df_update_ceiling = update_ceiling(data=df, option='gap', use_lforce=True)
df_nobound, df_nobound_update_ceiling = update_ceiling(data=df, option='gap', use_lforce=True)

# IV.B --- Compute production function decomposition of ceiling and actual output


def prodfunc(
        data,
        use_lforce=True
):
    d = data.copy()

    # ACTUAL
    if use_lforce:
        d['implied_y'] = (d['alpha'] / 100) * d['ln_k_stock15'] + (1 - d['alpha'] / 100) * d['ln_lforce']
    elif not use_lforce:
        d['implied_y'] = (d['alpha'] / 100) * d['ln_k_stock15'] + (1 - d['alpha'] / 100) * d['ln_employment']
    d['ln_tfp'] = d['ln_gdp15'] - d['implied_y']  # ln(tfp)

    # CEILING
    # Calculate TFP
    if use_lforce:
        d['ln_tfp_ceiling'] = d['ln_gdp15_ceiling'] - \
                              ((d['alpha'] / 100) * d['ln_k_stock15_ceiling']) - \
                              (1 - d['alpha'] / 100) * d['ln_lforce_ceiling']
        d['ln_tfp_ceiling_lb'] = d['ln_gdp15_ceiling_lb'] - \
                              ((d['alpha'] / 100) * d['ln_k_stock15_ceiling_lb']) - \
                              (1 - d['alpha'] / 100) * d['ln_lforce_ceiling_lb']
    elif not use_lforce:
        d['ln_tfp_ceiling'] = d['ln_gdp15_ceiling'] - \
                              ((d['alpha'] / 100) * d['ln_k_stock15_ceiling']) - \
                              (1 - d['alpha'] / 100) * d['ln_employment_ceiling']
        d['ln_tfp_ceiling_lb'] = d['ln_gdp15_ceiling_lb'] - \
                              ((d['alpha'] / 100) * d['ln_k_stock15_ceiling_lb']) - \
                              (1 - d['alpha'] / 100) * d['ln_employment_ceiling_lb']
    # Back out levels (PO)
    d['gdp15_ceiling'] = np.exp(d['ln_gdp15_ceiling'])
    d['gdp15_ceiling_lb'] = np.exp(d['ln_gdp15_ceiling_lb'])
    d['output_gap'] = 100 * (d['gdp15'] / d['gdp15_ceiling'] - 1)  # % PO
    d['output_gap_lb'] = 100 * (d['gdp15'] / d['gdp15_ceiling_lb'] - 1) # % PO
    d['capital_ceiling'] = np.exp(d['ln_k_stock15_ceiling'])
    d['capital_ceiling_lb'] = np.exp(d['ln_k_stock15_ceiling_lb'])
    if use_lforce:
        d['labour_ceiling'] = np.exp(d['ln_lforce_ceiling'])
        d['labour_ceiling_lb'] = np.exp(d['ln_lforce_ceiling_lb'])
    elif not use_lforce:
        d['labour_ceiling'] = np.exp(d['ln_employment_ceiling'])
        d['labour_ceiling_lb'] = np.exp(d['ln_employment_ceiling_lb'])
    d['tfp_ceiling'] = np.exp(d['ln_tfp_ceiling'])
    d['tfp_ceiling_lb'] = np.exp(d['ln_tfp_ceiling_lb'])

    # Back out levels (observed output)
    d['capital_observed'] = np.exp(d['ln_k_stock15'])
    if use_lforce: d['labour_observed'] = np.exp(d['ln_labour'])
    elif not use_lforce: d['labour_observed'] = np.exp(d['ln_employment'])
    d['tfp_observed'] = np.exp(d['ln_tfp'])

    # trim data frame
    list_col_keep = ['output_gap', 'output_gap_lb',
                     'gdp15_ceiling', 'gdp15_ceiling_lb',
                     'gdp15',
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
    list_levels = ['gdp15_ceiling', 'gdp15',
                   'capital_ceiling', 'labour_ceiling', 'tfp_ceiling',
                   'capital_observed', 'labour_observed', 'tfp_observed']
    list_yoy = [i + '_yoy' for i in list_levels]
    for i, j in zip(list_levels, list_yoy):
        d[j] = 100 * ((d[i] / d[i].shift(4)) - 1)

    # Decompose potential output growth
    d['capital_cont_ceiling'] = d['capital_ceiling_yoy'] * (d['alpha'] / 100)
    d['labour_cont_ceiling'] = d['labour_ceiling_yoy'] * (1 - (d['alpha'] / 100))
    d['tfp_cont_ceiling'] = d['gdp15_ceiling_yoy'] - d['capital_cont_ceiling'] - d['labour_cont_ceiling']

    # Decompose observed output growth
    d['capital_cont_observed'] = d['capital_observed_yoy'] * (d['alpha'] / 100)
    d['labour_cont_observed'] = d['labour_observed_yoy'] * (1 - (d['alpha'] / 100))
    d['tfp_cont_observed'] = d['gdp15_yoy'] - d['capital_cont_observed'] - d['labour_cont_observed']

    return d

df_pd = prodfunc(data=df, use_lforce=False)  # use employment, so no assumptions on NAIRU is required
df_hd = prodfunc_histdecomp(input=df_pd)

# V --- Charts
# Bounded and unbounded ceiling estimates
fig_bounds = go.Figure()
fig_bounds.add_trace(
        go.Scatter(
            x=df_nobound.index.astype('str'),
            y=df_nobound['ln_gdp15_ceiling'],
            name='Without',
            mode='lines',
            line=dict(color='black', width=1)
        )
    )
fig_bounds.add_trace(
        go.Scatter(
            x=df.index.astype('str'),
            y=df['ln_gdp15_ceiling'],
            name='With',
            mode='lines',
            line=dict(color='crimson', width=1)
        )
    )
fig_bounds.update_layout(title='Estimates of Output Ceiling With and Without Hard Upper Bound',
                  yaxis_title='Natural Logs',
                  plot_bgcolor='white',
                  hovermode='x',
                  barmode='relative',
                  font=dict(size=20, color='black'))
fig_bounds.write_image('Output/' + 'PluckingPO_HardAndNoBound' + '.png', height=768, width=1366)
fig_bounds.write_html('Output/' + 'PluckingPO_HardAndNoBound' + '.html')
telsendimg(conf=tel_config,
           path='Output/PluckingPO_HardAndNoBound.png',
           cap='Estimates of Output Ceiling With and Without Hard Upper Bound')

# Difference between bounded and unbounded ceiling estimates
fig_bounds = go.Figure()
fig_bounds.add_trace(
    go.Scatter(
        x=df_nobound.index.astype('str'),
        y=df_nobound['ln_gdp15_ceiling'] - df['ln_gdp15_ceiling'],
        name='Without - With',
        mode='lines',
        line=dict(color='lightcoral', width=1),
        fill='tozeroy'
    )
)
fig_bounds.update_layout(
    title='Difference Between Estimates of Output Ceiling With and Without Hard Upper Bound',
    yaxis_title='Natural Logs',
    plot_bgcolor='white',
    hovermode='x',
    barmode='relative',
    showlegend=False,
    font=dict(size=20, color='black')
)
fig_bounds.write_image('Output/' + 'PluckingPO_HardAndNoBound_Diff' + '.png', height=768, width=1366)
fig_bounds.write_html('Output/' + 'PluckingPO_HardAndNoBound_Diff' + '.html')
telsendimg(conf=tel_config,
           path='Output/PluckingPO_HardAndNoBound_Diff.png',
           cap='Difference Between Estimates of Output Ceiling With and Without Hard Upper Bound')

# Update of output ceiling using K and N
fig_update_ceiling = go.Figure()
fig_update_ceiling.add_trace(
        go.Scatter(
            x=df_update_ceiling.index.astype('str'),
            y=df_update_ceiling['gdp15_ceiling_initial'],
            name='Initial Estimate',
            mode='lines',
            line=dict(color='black', width=1)
        )
    )
fig_update_ceiling.add_trace(
        go.Scatter(
            x=df_update_ceiling.index.astype('str'),
            y=df_update_ceiling['gdp15_ceiling'],
            name='Final Estimate',
            mode='lines',
            line=dict(color='crimson', width=1)
        )
    )
fig_update_ceiling.update_layout(title='Initial and Final Estimates of the Output Ceiling',
                  yaxis_title='MYR mn',
                  plot_bgcolor='white',
                  hovermode='x',
                  barmode='relative',
                  font=dict(size=20, color='black'))
fig_update_ceiling.write_image('Output/' + 'PluckingPO_UpdateCeiling' + '.png', height=768, width=1366)
fig_update_ceiling.write_html('Output/' + 'PluckingPO_UpdateCeiling' + '.html')
telsendimg(conf=tel_config,
           path='Output/PluckingPO_UpdateCeiling.png',
           cap='Initial and Final Estimates of the Output Ceiling')

# Update of output ceiling using K and N
fig_update_ceiling_decomp = go.Figure()
fig_update_ceiling_decomp.add_trace(
        go.Scatter(
            x=df_update_ceiling.index.astype('str'),
            y=df_update_ceiling['ln_gdp15_ceiling_initial'],
            name='Initial Estimate',
            mode='lines',
            line=dict(color='black', width=1)
        )
    )
fig_update_ceiling_decomp.add_trace(
    go.Scatter(
        x=df_update_ceiling.index.astype('str'),
        y=df_update_ceiling['ln_k_rev'] + df_update_ceiling['ln_gdp15_ceiling_initial'],
        name='Capital Gap',
        fill='tonexty',
        mode='lines',
        line=dict(color='lightblue', width=0),
        fillcolor='lightblue'
    )
)
fig_update_ceiling_decomp.add_trace(
    go.Scatter(
        x=df_update_ceiling.index.astype('str'),
        y=df_update_ceiling['ln_n_rev'] + df_update_ceiling['ln_k_rev'] + df_update_ceiling['ln_gdp15_ceiling_initial'],
        name='Labour Gap',
        fill='tonexty',
        mode='lines',
        line=dict(color='lightpink', width=0),
        fillcolor='lightpink'
    )
)
fig_update_ceiling_decomp.add_trace(
        go.Scatter(
            x=df_update_ceiling.index.astype('str'),
            y=df_update_ceiling['ln_gdp15_ceiling'],
            name='Final Estimate',
            mode='lines',
            line=dict(color='crimson', width=1)
        )
    )
fig_update_ceiling_decomp.update_layout(title='Initial and Final Estimates of Log of Output Ceiling with Decomposition',
                  yaxis_title='Natural Logs',
                  plot_bgcolor='white',
                  hovermode='x',
                  barmode='relative',
                  font=dict(size=20, color='black'))
fig_update_ceiling_decomp.write_image('Output/' + 'PluckingPO_UpdateCeiling_Decomp' + '.png', height=768, width=1366)
fig_update_ceiling_decomp.write_html('Output/' + 'PluckingPO_UpdateCeiling_Decomp' + '.html')
telsendimg(conf=tel_config,
           path='Output/PluckingPO_UpdateCeiling_Decomp.png',
           cap='Initial and Final Estimates of the Log of Output Ceiling with Decomposition')

# Ceilings and observed + boom-bust version of output gap


def plot_linechart(data, cols, nice_names, colours, dash_styles, y_axis_title, main_title, output_suffix, fcast_start):
    d = data.copy()
    fig = go.Figure()
    for col, nice_name, colour, dash_style in tqdm(zip(cols, nice_names, colours, dash_styles)):
        fig.add_trace(
            go.Scatter(
                x=d.index.astype('str'),
                y=d[col],
                name=nice_name,
                mode='lines',
                line=dict(color=colour, dash=dash_style)
            )
        )
    d['_shadetop'] = d[cols].max().max()  # max of entire dataframe
    d.loc[d.index < fcast_start, '_shadetop'] = 0
    fig.add_trace(
        go.Bar(
            x=d.index.astype('str'),
            y=d['_shadetop'],
            name='Forecast',
            width=1,
            marker=dict(color='lightgrey', opacity=0.3)
        )
    )
    if bool(d[cols].min().min() < 0):  # To avoid double shades
        d['_shadebtm'] = d[cols].min().min()  # min of entire dataframe
        d.loc[d.index < fcast_start, '_shadebtm'] = 0
        fig.add_trace(
            go.Bar(
                x=d.index.astype('str'),
                y=d['_shadebtm'],
                name='Forecast',
                width=1,
                marker=dict(color='lightgrey', opacity=0.3)
            )
        )
    fig.update_layout(title=main_title,
                      yaxis_title=y_axis_title,
                      plot_bgcolor='white',
                      hovermode='x',
                      barmode='relative',
                      font=dict(size=20, color='black'))
    fig.write_image('Output/PluckingPO_ObsCeiling_' + output_suffix + '.png', height=768, width=1366)
    fig.write_html('Output/PluckingPO_ObsCeiling_' + output_suffix + '.html')
    return fig

# Original boom-bust output gap
df_bb = pd.read_csv('D:/Users/ECSUAH/OneDrive - Bank Negara Malaysia/output_for_po_estimation/2022-10-17_KFilter_Estimates.txt', sep='|')
df_bb['quarter'] = pd.to_datetime(df_bb['quarter']).dt.to_period('Q')
df_bb = df_bb[(df_bb['quarter'] >= T_lb) & (df_bb['quarter'] <= T_forecast_end)]
df_bb = df_bb.set_index('quarter')
df_bb = df_bb.sort_index()
df_og = pd.concat([df_bb['output_gap_avg'], df_pd[['output_gap', 'output_gap_lb']]], axis=1)
df_og = df_og.rename(columns={'output_gap_avg': 'boom_bust_og',
                               'output_gap': 'pluck_og',
                               'output_gap_lb': 'pluck_og_lb'})
df_og = df_og.sort_index()
df_og['boom_bust_og_norm'] = \
    (df_og['boom_bust_og'] - df_og['boom_bust_og'].min()) / (df_og['boom_bust_og'].max() - df_og['boom_bust_og'].min())
df_og['pluck_og_norm'] = \
    (df_og['pluck_og'] - df_og['pluck_og'].min()) / (df_og['pluck_og'].max() - df_og['pluck_og'].min())
df_og['ref'] = 0

# Charts
if show_conf_bands:
    col_gdp = ['gdp15', 'gdp15_ceiling', 'gdp15_ceiling_lb']
    col_lab = ['labour_observed', 'labour_ceiling', 'labour_ceiling_lb']
    col_cap = ['capital_observed', 'capital_ceiling', 'capital_ceiling_lb']
    col_tfp = ['tfp_observed', 'tfp_ceiling', 'tfp_ceiling_lb']
    col_og = ['pluck_og', 'pluck_og_lb', 'boom_bust_og', 'ref']

    col_gdp_nice = ['Observed Real GDP', 'Output Ceiling UB', 'Output Ceiling LB']
    col_lab_nice = ['Observed Labour', 'Labour Ceiling UB', 'Labour Ceiling LB']
    col_cap_nice = ['Observed Capital', 'OutCapitalput Ceiling UB', 'Capital Ceiling LB']
    col_tfp_nice = ['Observed TFP', 'TFP Ceiling UB', 'TFP Ceiling LB']
    col_og_nice = ['Plucking UB', 'Plucking LB', 'Current', 'Reference (Y=0)']

    colour1 = ['crimson', 'black', 'black']
    colour2 = ['darkblue', 'darkblue', 'lightcoral', 'black']

    dash1 = ['solid', 'longdash', 'longdash']
    dash2 = ['solid', 'solid', 'longdash', 'dash']

elif not show_conf_bands:
    col_gdp = ['gdp15', 'gdp15_ceiling']
    col_lab = ['labour_observed', 'labour_ceiling']
    col_cap = ['capital_observed', 'capital_ceiling']
    col_tfp = ['tfp_observed', 'tfp_ceiling']
    col_og = ['pluck_og', 'boom_bust_og', 'ref']

    col_gdp_nice = ['Observed Real GDP', 'Output Ceiling']
    col_lab_nice = ['Observed Labour', 'Labour Ceiling']
    col_cap_nice = ['Observed Capital', 'OutCapitalput Ceiling']
    col_tfp_nice = ['Observed TFP', 'TFP Ceiling']
    col_og_nice = ['Plucking', 'Current', 'Reference (Y=0)']

    colour1 = ['crimson', 'black']
    colour2 = ['darkblue', 'lightcoral', 'black']

    dash1 = ['solid', 'longdash']
    dash2 = ['solid', 'longdash', 'dash']

col_og_norm = ['pluck_og_norm', 'boom_bust_og_norm']
col_og_norm_nice = ['Plucking UB', 'Current']
colour3 = ['darkblue', 'lightcoral']
dash3 = ['solid', 'solid']

fig_gdp = plot_linechart(data=df_pd,
                         cols=col_gdp,
                         nice_names=col_gdp_nice,
                         colours=colour1,
                         dash_styles=dash1,
                         y_axis_title='ppt',
                         main_title='Real GDP: Observed and Ceiling',
                         output_suffix='GDP',
                         fcast_start=T_forecast_start)
fig_lab = plot_linechart(data=df_pd,
                         cols=col_lab,
                         nice_names=col_lab_nice,
                         colours=colour1,
                         dash_styles=dash1,
                         y_axis_title='ppt',
                         main_title='Labour: Observed and Ceiling',
                         output_suffix='Labour',
                         fcast_start=T_forecast_start)
fig_cap = plot_linechart(data=df_pd,
                         cols=col_cap,
                         nice_names=col_cap_nice,
                         colours=colour1,
                         dash_styles=dash1,
                         y_axis_title='ppt',
                         main_title='Capital: Observed and Ceiling',
                         output_suffix='Capital',
                         fcast_start=T_forecast_start)
fig_tfp = plot_linechart(data=df_pd,
                         cols=col_tfp,
                         nice_names=col_tfp_nice,
                         colours=colour1,
                         dash_styles=dash1,
                         y_axis_title='ppt',
                         main_title='TFP: Observed and Ceiling',
                         output_suffix='TFP',
                         fcast_start=T_forecast_start)
fig_og = plot_linechart(data=df_og,
                         cols=col_og,
                         nice_names=col_og_nice,
                         colours=colour2,
                         dash_styles=dash2,
                        y_axis_title='% Potential Output',
                        main_title='Output Gap: Current and Plucking',
                        output_suffix='OG',
                        fcast_start=T_forecast_start)
fig_og_norm = plot_linechart(data=df_og,
                             cols=col_og_norm,
                             nice_names=col_og_norm_nice,
                             colours=colour3,
                             dash_styles=dash3,
                             y_axis_title='Index',
                             main_title='Normalised Output Gap: Current and Plucking',
                             output_suffix='OG_Norm',
                             fcast_start=T_forecast_start)

suffix_figs = ['GDP', 'Labour', 'Capital', 'TFP', 'OG', 'OG_Norm']
for i in suffix_figs:
    telsendimg(conf=tel_config,
               path='Output/PluckingPO_ObsCeiling_' + i + '.png',
               cap=i + ' (observed and ceiling)')

# Contribution


def plot_histdecomp(input):
    d = input.copy()
    list_col_keep = ['gdp15_yoy', 'gdp15_ceiling_yoy',
                     'capital_cont_ceiling', 'labour_cont_ceiling', 'tfp_cont_ceiling',
                     'capital_cont_observed', 'labour_cont_observed', 'tfp_cont_observed',]
    d = d[list_col_keep]

    # Potential output
    fig_ceiling = go.Figure()
    fig_ceiling.add_trace(go.Scatter(x=d.index.astype('str'),
                                     y=d['gdp15_ceiling_yoy'],
                                     name='Ceiling Output Growth',
                                     mode='lines',
                                     line=dict(width=3, color='black')))
    fig_ceiling.add_trace(go.Bar(x=d.index.astype('str'),
                                 y=d['capital_cont_ceiling'],
                                 name='Ceiling Capital Growth',
                                 marker=dict(color='lightblue')))
    fig_ceiling.add_trace(go.Bar(x=d.index.astype('str'),
                                 y=d['labour_cont_ceiling'],
                                 name='Ceiling Labour Growth',
                                 marker=dict(color='lightpink')))
    fig_ceiling.add_trace(go.Bar(x=d.index.astype('str'),
                                 y=d['tfp_cont_ceiling'],
                                 name='Ceiling TFP Growth',
                                 marker=dict(color='palegreen')))
    fig_ceiling.update_layout(title='Historical Decomposition of Ceiling Output YoY Growth',
                              yaxis_title='Percentage Point, %YoY growth',
                              barmode='relative',
                              plot_bgcolor='white',
                              hovermode='x',
                              font=dict(size=20, color='black'))
    fig_ceiling.write_html('Output/PluckingPO_HistDecomp_Ceiling.html')
    fig_ceiling.write_image('Output/PluckingPO_HistDecomp_Ceiling.png', height=768, width=1366)

    # Observed output
    fig_ob = go.Figure()
    fig_ob.add_trace(go.Scatter(x=d.index.astype('str'),
                                y=d['gdp15_yoy'],
                                name='Observed Output Growth',
                                mode='lines',
                                line=dict(width=3, color='black')))
    fig_ob.add_trace(go.Bar(x=d.index.astype('str'),
                            y=d['capital_cont_observed'],
                            name='Observed Capital Growth',
                            marker=dict(color='lightblue')))
    fig_ob.add_trace(go.Bar(x=d.index.astype('str'),
                            y=d['labour_cont_observed'],
                            name='Observed Labour Growth',
                            marker=dict(color='lightpink')))
    fig_ob.add_trace(go.Bar(x=d.index.astype('str'),
                            y=d['tfp_cont_observed'],
                            name='Observed TFP Growth',
                            marker=dict(color='palegreen')))
    fig_ob.update_layout(title='Historical Decomposition of Observed Output YoY Growth',
                         yaxis_title='Percentage Point, %YoY growth',
                         barmode='relative',
                         plot_bgcolor='white',
                         hovermode='x',
                         font=dict(size=20, color='black'))
    fig_ob.write_html('Output/PluckingPO_HistDecomp_Obs.html')
    fig_ob.write_image('Output/PluckingPO_HistDecomp_Obs.png', height=768, width=1366)

    return fig_ceiling, fig_ob


fig_ceiling, fig_ob = plot_histdecomp(input=df_hd)
for i in ['Output/PluckingPO_HistDecomp_Ceiling.png', 'Output/PluckingPO_HistDecomp_Obs.png']:
    telsendimg(conf=tel_config,
               path=i)


# -------------------- OVERLAYING RECOVERIES --------------------

# Setup
df_recov = pd.concat([pd.DataFrame(df['ln_gdp15_cepi']).rename(columns={'ln_gdp15_cepi': 'cepi'}),
                      pd.DataFrame(df_pd['output_gap'])],
                     axis=1)
df_recov = df_recov[df_recov['cepi'] >= 1]
# move cepi up by 1 row
df_recov['cepi'] = df_recov['cepi'].shift(-1)
df_recov['cepi'] = df_recov['cepi'].fillna(method='ffill')
df_recov['cepi'] = df_recov['cepi'].astype('int')
# Setup data frame
list_crises=['AFC', 'GFC', 'COVID-19']
df_recov_consol = pd.DataFrame(columns=list_crises)
# Wrangle
r = 1
for cepi, crises in zip(range(df_recov['cepi'].min(), df_recov['cepi'].max() + 1), list_crises):
    d = df_recov[df_recov['cepi'] == cepi].reset_index(drop=True)
    d = pd.DataFrame(d['output_gap']).rename(columns={'output_gap': crises})
    if r == 1:
        df_recov_consol = d.copy()
    elif r > 1:
        df_recov_consol = pd.concat([df_recov_consol, d], axis=1)
    r += 1
for crises in list_crises:
    df_recov_consol[crises] = df_recov_consol[crises] - df_recov_consol.loc[df_recov_consol.index == 0, crises][0]
df_recov_consol['ref'] = 0
df_recov_short = df_recov_consol.head(28)
# Calculate when to start shading as 'forecast'
df_recov_latest = df_recov[df_recov['cepi'] == df_recov['cepi'].max()]
df_recov_latest = df_recov_latest.reset_index()
recov_fcast_start = df_recov_latest[df_recov_latest['quarter'] == T_forecast_start].reset_index()['index'][0]
# Split latest vintage into 'estimate' and 'forecast', and plot both separately
df_recov_short['COVID-19_forecast'] = df_recov_short.loc[df_recov_short.index >= recov_fcast_start, 'COVID-19']
df_recov_short.loc[df_recov_short.index > recov_fcast_start, 'COVID-19'] = np.nan
# Plot
fig_recov = plot_linechart(
    data=df_recov_short,
    cols=list_crises + ['COVID-19_forecast', 'ref'],
    nice_names=list_crises + ['COVID-19 (Forecast)', 'Reference (Pre-Crisis Peak)'],
    colours=['crimson', 'darkblue', 'black', 'black', 'black'],
    dash_styles=['solid', 'solid', 'solid', 'longdash', 'dash'],
    y_axis_title='Percentage Points (0 = Pre-Crisis Peak)',
    main_title='Evolution of Plucking Output Gap Post-Crises',
    output_suffix='CrisisRecoveries',
    fcast_start=recov_fcast_start
)
telsendimg(conf=tel_config,
           path='Output/PluckingPO_ObsCeiling_CrisisRecoveries.png',
           cap='Evolution of Plucking Output Gap Post-Crises')

# -------------------- ANALYSIS OF PLUCKING PROPERTY --------------------

run_scatter = 0
if run_scatter == 1:


    def plot_scatter(data, y_col, x_col, colour, chart_title, output_suffix):
        fig = px.scatter(data, x=x_col, y=y_col, trendline='ols', color_discrete_sequence=[colour])
        fig.update_traces(marker=dict(size=20),
                          selector=dict(mode='markers'))
        fig.update_layout(title=chart_title,
                          plot_bgcolor='white',
                          hovermode='x',
                          font=dict(size=20, color='black'))
        fig.write_image('Output/PluckingPO_Scatter_' + output_suffix + '.png', height=768, width=1366)
        fig.write_html('Output/PluckingPO_Scatter_' + output_suffix + '.html')
        return fig


    list_pace = [i + '_pace' for i in list_ln]
    list_epi = [i + '_epi' for i in list_ln]
    for levels, pace, epi in zip(list_ln, list_pace, list_epi):
        if df[epi].max() == 0:
            pass
        else:
            d = df[[pace, epi]]
            d = d.groupby(epi).agg('mean')

            expansions = d.iloc[::2].reset_index(drop=True)
            subsequent_contractions = d.iloc[1::2].reset_index(drop=True)
            d_exp_con = pd.concat([expansions, subsequent_contractions], axis=1).dropna()
            d_exp_con.columns = [levels + '_expansion_pace', levels + '_subsequent_contraction_pace']
            dfi.export(d_exp_con, 'Output/PluckingPO_ExpConTable_' + levels + '.png')
            fig = plot_scatter(data=d_exp_con,
                               y_col=levels + '_subsequent_contraction_pace',
                               x_col=levels + '_expansion_pace',
                               colour='black',
                               chart_title=levels + ': Expansion Pace vs. Subsequent Contraction Pace',
                               output_suffix=levels + '_ExpCon')
            telsendimg(conf=tel_config,
                       path='Output/PluckingPO_Scatter_' + levels + '_ExpCon' + '.png')


            contractions = d.iloc[1::2].reset_index(drop=True)
            subsequent_expansions = d.iloc[2::2].reset_index(drop=True)
            d_con_exp = pd.concat([contractions, subsequent_expansions], axis=1).dropna()
            d_con_exp.columns = [levels + '_contraction_pace', levels + '_subsequent_expansion_pace']
            dfi.export(d_con_exp, 'Output/PluckingPO_ConExpTable_' + levels + '.png')
            fig = plot_scatter(data=d_con_exp,
                               y_col=levels + '_subsequent_expansion_pace',
                               x_col=levels + '_contraction_pace',
                               colour='crimson',
                               chart_title=levels + ': Contraction Pace vs. Subsequent Expansion Pace',
                               output_suffix=levels + '_ConExp')
            telsendimg(conf=tel_config,
                       path='Output/PluckingPO_Scatter_' + levels + '_ConExp' + '.png')

# -------------------- COMPILE ALL CHARTS --------------------


def pil_img2pdf(list_images, extension='png', img_path='Output/', pdf_name='PluckingPO_AllCharts'):
    seq = list_images.copy()  # deep copy
    list_img = []
    file_pdf = img_path + pdf_name + '.pdf'
    run = 0
    for i in seq:
        img = Image.open(img_path + i + '.' + extension)
        img = img.convert('RGB')  # PIL cannot save RGBA files as pdf
        if run == 0:
            first_img = img.copy()
        elif run > 0:
            list_img = list_img + [img]
        run += 1
    first_img.save(img_path + pdf_name + '.pdf',
                   'PDF',
                   resolution=100.0,
                   save_all=True,
                   append_images=list_img)


seq_output = [
    'PluckingPO_UpdateCeiling', 'PluckingPO_HardAndNoBound', 'PluckingPO_HardAndNoBound_Diff',
    'PluckingPO_ObsCeiling_GDP', 'PluckingPO_ObsCeiling_Labour', 'PluckingPO_ObsCeiling_Capital',
    'PluckingPO_ObsCeiling_TFP', 'PluckingPO_ObsCeiling_OG', 'PluckingPO_ObsCeiling_OG_Norm',
    'PluckingPO_HistDecomp_Ceiling', 'PluckingPO_HistDecomp_Obs',
    'PluckingPO_ObsCeiling_CrisisRecoveries',
    # 'PluckingPO_Scatter_ln_gdp15_ExpCon', 'PluckingPO_Scatter_ln_gdp15_ConExp',
    # 'PluckingPO_Scatter_ln_labour_ExpCon', 'PluckingPO_Scatter_ln_labour_ConExp',
    # 'PluckingPO_Scatter_ln_employment_ExpCon', 'PluckingPO_Scatter_ln_employment_ConExp',
    # 'PluckingPO_ExpConTable_ln_gdp15', 'PluckingPO_ConExpTable_ln_gdp15',
    # 'PluckingPO_ExpConTable_ln_labour', 'PluckingPO_ConExpTable_ln_labour',
    # 'PluckingPO_ExpConTable_ln_employment', 'PluckingPO_ConExpTable_ln_employment'
]
pil_img2pdf(list_images=seq_output,
            img_path='Output/',
            extension='png',
            pdf_name='PluckingPO_AllCharts')
telsendfiles(conf=tel_config,
             path='Output/PluckingPO_AllCharts.pdf',
             cap='All charts from the PluckingPO flow')

# -------------------- EXPORT DATA FRAMES --------------------
df_pd = df_pd.reset_index()
df_pd.to_csv('Output/PluckingPO_Estimates.txt', sep='|', index=False)
telsendfiles(conf=tel_config,
             path='Output/PluckingPO_Estimates.txt',
             cap='Estimates from the PluckingPO flow')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
