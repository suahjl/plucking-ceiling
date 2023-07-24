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


# II --- Load original
df = pd.read_parquet('boombustpo_input_data_kf' + file_suffix_fcast + '.parquet')

# III --- Redo HP filtered PO with recursive version
del df['po_hp']
del df['output_gap_hp']
burn_in_duration = 20
t_count = 0
for t in tqdm(list(df.index)):
    if t_count < burn_in_duration:
        pass
    elif t_count >= burn_in_duration:
        cycle, trend = \
            sm.filters.hpfilter(
                df.loc[(~df['ln_gdp'].isna()) & (df.index <= t), 'ln_gdp'],
                lamb=11200
            )
        if t_count == burn_in_duration:
            df['po_hp'] = trend
            df['output_gap_hp'] = cycle
        elif t_count > burn_in_duration:
            df['po_hp'] = df['po_hp'].combine_first(trend)  # fill in NA with new trend
            df['output_gap_hp'] = df['output_gap_hp'].combine_first(cycle)  # fill in NA with new trend
    t_count += 1

# IV --- Bring in one-sided HP filter PF estimates
# Load
df_hp = pd.read_parquet('boombustpo_estimates_pf_onesided' + file_suffix_fcast + '.parquet')
# df_hp['quarter'] = pd.to_datetime(df_hp['quarter']).dt.to_period('Q')
# df_hp = df_hp.set_index('quarter')
# Replace
df['output_gap_pf'] = df_hp['output_gap'] / 100  # rescaled
df['po_pf'] = df_hp['po']
df['ln_po_pf'] = np.log(df['po_pf'])
df['ln_po_pf_d'] = df['ln_po_pf'] - df['ln_po_pf'].shift(1)

# IV --- Export data frames
df.to_parquet('boombustpo_input_data_kf_onesided' + file_suffix_fcast + '.parquet', compression='brotli')

# V --- Notify
if not use_forecast:
    telsendmsg(conf=tel_config,
               msg='boombustpo_compile_input_data_kf_onesided: COMPLETED')
if use_forecast:
    telsendmsg(conf=tel_config,
               msg='boombustpo_compile_input_data_kf_onesided: COMPLETED (WITH FORECAST)')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
