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
from ceic_api_client.pyceic import Ceic
from dotenv import load_dotenv
import os
import ast

time_start = time.time()

# 0 --- Main settings
load_dotenv()
tel_config = os.getenv('TEL_CONFIG')
T_lb = '1995Q1'
T_lb_day = date(1995, 1, 1)
show_conf_bands = False
use_forecast = ast.literal_eval(os.getenv('USE_FORECAST_BOOL'))
manual_data = ast.literal_eval(os.getenv('MANUAL_DATA'))
if not manual_data:
    Ceic.login(os.getenv('CEIC_USERNAME'), os.getenv('CEIC_PASSWORD'))

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
        time_points_dict = dict((tp._date, tp.value) for tp in y[0].time_points)  # this is a list of 1 dictionary,
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


# II --- Wrangling
# Base CEIC data --- output (quarterly), labour force (quarterly), net capital stock (annual)
if not manual_data:
    series_ids = pd.read_csv('ceic_seriesid_pluckingpo.txt', sep='|')
    series_ids = list(series_ids['series_id'])
    df_ceic = ceic2pandas_ts(input=series_ids, start_date=T_lb_day)
    df_ceic = df_ceic.rename(columns={'Real GDP 2014p: Quarterly: Malaysia':'gdp',
                                      'Labour Force: Person th: Malaysia': 'lforce',
                                      'Net Capital Stock (NKS)': 'nks15',
                                      '(DC)Net Capital Stock (NKS): 2010p': 'nks10',
                                      '(DC)Net Capital Stock (NKS): 2005p': 'nks05',
                                      'Net Capital Stock (NKS): 2000p': 'nks00',
                                      'GDP: 2015p: GCF: Gross Fixed Capital Formation (GFCF)': 'gfcf'})
    df_ceic = df_ceic.reset_index().rename(columns={'index': 'quarter'})
    df_ceic['quarter'] = pd.to_datetime(df_ceic['quarter']).dt.to_period('Q')
    df_ceic = df_ceic.set_index('quarter')  # harmonise index

if manual_data:
    df_ceic = pd.read_excel('ceic_pluckingpo.xlsx', sheet_name='Sheet1')

# Static nks (annual)
nks_old = pd.read_csv('old_static_nks.txt')
nks_old['year'] = nks_old['year'].astype('str') + '-12-01'  # make everything december 1 (last Q)
nks_old['year'] = pd.to_datetime(nks_old['year'].astype('str')).dt.date
nks_old = nks_old.rename(columns={'nks': 'nks_old'})
nks_old = nks_old.set_index('year')

# Static labour force
lforce_old = pd.read_csv('old_static_lforce.txt')
lforce_old['quarter'] = pd.to_datetime(lforce_old['quarter']).dt.to_period('Q')
lforce_old = lforce_old.rename(columns={'lforce': 'lforce_old'})
lforce_old = lforce_old.set_index('quarter')

# Splice nks into single annual series + merge with static series + interpolate into quarterly
# prelims
col_nks = ['nks00', 'nks05', 'nks10', 'nks15']
nks = df_ceic[col_nks].reset_index().rename(columns={'quarter': 'year'})
nks['year'] = pd.to_datetime(nks['year'].astype('str')).dt.date # harmonise index type
nks = nks.set_index('year')
nks = pd.concat([nks, nks_old], axis=1)
nks = nks.dropna(axis=0, how='all')  # drop rows where all values are NA
nks = nks.sort_index()
nks.loc[nks['nks15'].isna(), 'nks15'] = nks['nks_old']
# splice
col_cap_tosplice = ['nks00', 'nks05', 'nks10']
for i in col_cap_tosplice:
    nks[i] = nks[i] / nks[i].shift(1)  # G_t = (1 + g_t) = (Y_t / Y_{t-1})
col_cap_tosplice.reverse()  # reverse order, so that the most recent series takes precedence
cap_splice_run = len(nks)
for i in col_cap_tosplice:
    run = 1
    while run <= cap_splice_run:
        nks.loc[nks['nks15'].isna(), 'nks15'] = \
            nks['nks15'].shift(-1) / nks[i].shift(-1)  # Y_t = Y_{t+1} / G_{t+1}
        run += 1
    else:
        pass
nks = nks.rename(columns={'nks15': 'nks'})
for i in ['nks00', 'nks05', 'nks10']:
    del nks[i]  # clear old columns
    del df_ceic[i]
del df_ceic['nks15']
# interpolate
nks = nks.reset_index()
nks = nks.rename(columns={'year': 'quarter'})
nks['quarter'] = pd.to_datetime(nks['quarter']).dt.to_period('Q')
nks = nks.set_index('quarter')
last_nks_obsq = nks.index[-1] + 1  # range is [a, b)
qrange_series = pd.DataFrame(pd.date_range(start=T_lb_day, end=str(last_nks_obsq) , freq='Q').to_period('Q'))
qrange_series['placeholder'] = 1
qrange_series = qrange_series.rename(columns={0: 'quarter'})
qrange_series = qrange_series.set_index('quarter')
nks = pd.concat([qrange_series, nks], axis=1)  # left right
nks = nks.sort_index()
del nks['placeholder']
nks = nks.interpolate(method='linear')  # can consider cubic spline too

# Merge base data sets
df = pd.concat([df_ceic, lforce_old, nks], axis=1)

# Extend NKS series with GFCF and fixed capital consumption
icfc = 1.76255291769226 / 100
while run <= len(df):
    df.loc[df['nks'].isna(), 'nks'] = df['nks'].shift(1) + \
                                      df['gfcf'] - \
                                      (icfc * df['nks'].shift(1))
    run += 1
else:
    pass
del df['gfcf']

# Harmonise lforce
df.loc[df['lforce'].isna(), 'lforce'] = df['lforce_old']

# Alpha
df['alpha'] = 58.4226639119982  # average of 2010 - T adjusted GOS share of output

# Cleaning
# Trim
for i in ['lforce_old', 'nks_old']:
    del df[i]

# Seasonal adjustment
list_col = ['gdp']  # only GDP is raw
for i in list_col:
    sadj_res = sm.x13_arima_analysis(df[i])
    sadj_seasadj = sadj_res.seasadj
    df.loc[:, i] = sadj_seasadj  # Ideally, use MYS-specific calendar effects

# FORECAST
if use_forecast:
    # gdp fcast
    gdp_fcast = pd.read_csv('Forecast/gdp_qoqsa_forecast.csv')
    gdp_fcast['quarter'] = pd.to_datetime(gdp_fcast['quarter']).dt.to_period('q')
    gdp_fcast = gdp_fcast.set_index('quarter')
    fcast_iteration = len(gdp_fcast)
    df = pd.concat([df, gdp_fcast], axis=1)
    round = 1
    while round <= fcast_iteration:
        df.loc[df['gdp'].isna(), 'gdp'] = df['gdp'].shift(1) * (1 + (df['gdp_qoqsa_forecast'] / 100)) # back out SA levels
        round += 1
    # nks fcast
    gfcf_fcast = pd.read_csv('Forecast/gfcf_sa_forecast.csv')
    gfcf_fcast['quarter'] = pd.to_datetime(gfcf_fcast['quarter']).dt.to_period('q')
    gfcf_fcast = gfcf_fcast.set_index('quarter')
    df = pd.concat([df, gfcf_fcast], axis=1)
    round = 1
    while round <= fcast_iteration:
        df.loc[df['nks'].isna(), 'nks'] = \
            df['nks'].shift(1) + df['gfcf_sa_forecast'] - \
            ((df['icfc_assumption'] / 100 )* df['nks'].shift(1))
        round += 1
    # lforce fcast
    lforce_fcast = pd.read_csv('Forecast/lforce_forecast.csv')
    lforce_fcast['quarter'] = pd.to_datetime(lforce_fcast['quarter']).dt.to_period('q')
    lforce_fcast = lforce_fcast.set_index('quarter')
    df = pd.concat([df, lforce_fcast], axis=1)
    df.loc[df['lforce'].isna(), 'lforce'] = df['lforce_forecast'].copy()
    # ffill alpha
    df['alpha'] = df['alpha'].fillna(method='ffill')
    # clear columns
    for i in ['lforce_forecast', 'gfcf_sa_forecast', 'icfc_assumption', 'gdp_qoqsa_forecast']:
        del df[i]

# Take logs
list_col = ['gdp', 'lforce', 'nks']
list_col_ln = ['ln_' + i for i in list_col]
for i, j in zip(list_col, list_col_ln):
    df.loc[:, j] = np.log(df[i])

# Take log-difference
list_col_ln_diff = [i + '_diff' for i in list_col_ln]
for i, j in zip(list_col_ln, list_col_ln_diff):
    df.loc[:, j] = df[i] - df[i].shift(1)


# III --- save interim data set
if not use_forecast:
    df = df.reset_index()
    df['quarter'] = df['quarter'].astype('str')
    df.to_parquet('pluckingpo_input_data.parquet', compression='brotli')
    telsendmsg(conf=tel_config,
               msg='pluckingpo-compile-input-data: COMPLETED')

if use_forecast:
    df = df.reset_index()
    df['quarter'] = df['quarter'].astype('str')
    df.to_parquet('pluckingpo_input_data_forecast.parquet', compression='brotli')
    telsendmsg(conf=tel_config,
               msg='pluckingpo-compile-input-data: COMPLETED (WITH FORECAST)')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
