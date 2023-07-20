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
from ceic_api_client.pyceic import Ceic
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
show_conf_bands = False
use_forecast = ast.literal_eval(os.getenv('USE_FORECAST_BOOL'))
if use_forecast:
    file_suffix_fcast = '_forecast'
    fcast_start = '2023Q1'
elif not use_forecast:
    file_suffix_fcast = ''

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


# II --- Load data
# Data for NKPC
Ceic.login(os.getenv('CEIC_USERNAME'), os.getenv('CEIC_PASSWORD'))
series_id = pd.read_csv('ceic_seriesid_forkf.txt', sep='|')
series_id = list(series_id['series_id'])
df_ceic = ceic2pandas_ts(series_id, start_date=T_lb_day).fillna(method='ffill')
# Cleaning CEIC downloads: merging old + new brent
dict_ceic_rename = {'Crude Oil: Spot Price: Europe Brent': 'com',
                    'FX Spot Rate: FRB: Malaysian Ringgit': 'usdmyr',
                    'Consumer Price Index (CPI): Core': 'cpi_core'}
df_ceic = df_ceic.rename(columns=dict_ceic_rename)
df_ceic = df_ceic.reset_index()
df_ceic = df_ceic.rename(columns={'index': 'quarter'})
df_ceic['quarter'] = pd.to_datetime(df_ceic['quarter']).dt.to_period('Q')
df_ceic = df_ceic.groupby('quarter').agg('mean')

# Static historical core cpi
core_cpi_old = pd.read_csv('old_static_cpi_core.txt', sep=',')
core_cpi_old['quarter'] = pd.to_datetime(core_cpi_old['quarter']).dt.to_period('Q')
core_cpi_old = core_cpi_old.set_index('quarter')
core_cpi_old = core_cpi_old.rename(columns={'cpi_core': 'cpi_core_old'})
core_cpi_old = core_cpi_old[core_cpi_old.index >= T_lb]  # timebound

# Load PF estimates
output_pf = pd.read_parquet('boombustpo_estimates_pf' + file_suffix_fcast + '.parquet')  # if includes forecast
output_pf['quarter'] = pd.to_datetime(output_pf['quarter']).dt.to_period('Q')
output_pf = output_pf.set_index('quarter')
output_pf = output_pf.rename(columns={'output_gap': 'output_gap_pf',
                                      'po': 'po_pf'})
output_pf = output_pf[['gdp', 'output_gap_pf', 'po_pf']]

# Merge
df = pd.concat([df_ceic, core_cpi_old, output_pf], axis=1)
df.loc[df.index < '2015Q1', 'cpi_core'] = df['cpi_core_old']  # merge old and new cpi_core
del df['cpi_core_old']  # delete old cpi_core

# Delete redundant data frames
del df_ceic
del core_cpi_old
del output_pf

# III --- Data prep
# Seasonally adjust core cpi and com prices (2QMA)
list_col = ['cpi_core', 'com']
for i in tqdm(list_col):
    sadj_res = sm.x13_arima_analysis(df.loc[df[i].notna(), i])  # handles NAs
    sadj_seasadj = sadj_res.seasadj
    df[i] = sadj_seasadj

# New column
df['com_2q'] = ((df['com'] + df['com'].shift(1)) / 2)  # 2QMA of seasonally adjusted commodity prices

# Take logs
list_col = list_col + ['com_2q', 'gdp', 'po_pf', 'usdmyr']
list_col_ln = ['ln_' + i for i in list_col]
for i, j in tqdm(zip(list_col, list_col_ln)):
    df[j] = np.log(df[i])  # log(x)

# Take diff
list_col_ln_d = [i + '_d' for i in list_col_ln]
for i, j in tqdm(zip(list_col_ln, list_col_ln_d)):
    df[j] = df[i] - df[i].shift(1)  # logdiff(x)

# Gaps (relative to sample avg)
list_col_ln_d = ['ln_' + i + '_d' for i in ['usdmyr', 'com', 'com_2q']]
list_col_ln_d_gap = [i + '_gap' for i in ['usdmyr', 'com', 'com_2q']]
for i, j in tqdm(zip(list_col_ln_d, list_col_ln_d_gap)):
    df[j] = df[i] - df[i].mean()  # logdiff(x) - mean(logdiff(x))

# Rescale variables
list_col = ['output_gap_pf']
for i in tqdm(list_col):
    df[i] = df[i] / 100

# D. Additional
# Assumed trend growth for ProdFunc PO
df['ln_po_pf_d_trend'] = np.power(1.0479, 0.25) - 1
# HP-filtered PO (for initial states later)
trend, cycle = sm.filters.hpfilter(df['ln_gdp'].dropna(), lamb=11200)
df['po_hp'] = trend
df['output_gap_hp'] = cycle

# IV --- Export data frames
df = df.reset_index()
df['quarter'] = df['quarter'].astype('str')
df.to_parquet('boombustpo_input_data_kf' + file_suffix_fcast + '.parquet', compression='brotli')

# V --- Notify
if not use_forecast:
    telsendmsg(conf=tel_config,
               msg='boombustpo_compile_input_data_kf: COMPLETED')
if use_forecast:
    telsendmsg(conf=tel_config,
               msg='boombustpo_compile_input_data_kf: COMPLETED (WITH FORECAST)')
# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')