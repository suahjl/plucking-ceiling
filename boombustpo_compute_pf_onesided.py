# ------ WITH ONE-SIDED TIME SERIES FILTER
# (https://www.bis.org/publ/work1033.pdf -- HW filter)
# (https://www.bis.org/publ/work744.pdf -- recursive HP filter)
# (https://www.bis.org/publ/work114.pdf -- recursive HP filter)

import pandas as pd
import numpy as np
from datetime import date, timedelta
import statsmodels.formula.api as smf
import statsmodels.tsa.api as sm
# from quantecon import hamilton_filter
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


# II --- Load data
df = pd.read_parquet('pluckingpo_input_data' + file_suffix_fcast + '.parquet')  # use same data input as for plucking
df['quarter'] = pd.to_datetime(df['quarter']).dt.to_period('Q')
df = df.set_index('quarter')

# III --- RECURSIVE Holy Puki (HP) Filter
list_col_ln = ['ln_gdp', 'ln_lforce', 'ln_nks']
list_col_ln_trend = [i + '_trend' for i in list_col_ln]
burn_in_duration = 20
for i, j in zip(list_col_ln, list_col_ln_trend):
    t_count = 0
    for t in tqdm(list(df.index)):
        if t_count < burn_in_duration:
            pass
        elif t_count >= burn_in_duration:
            cycle, trend = \
                sm.filters.hpfilter(
                    df.loc[(~df[i].isna()) & (df.index <= t), i],
                    lamb=11200
                )
            if t_count == burn_in_duration:
                df[j] = trend
            elif t_count > burn_in_duration:
                df[j] = df[j].combine_first(trend)  # fill in NA with new trend
        t_count += 1

# IV --- Production function estimation


def prodfunc_po(data):
    d = data.copy()

    # TFP: a*ln(k) + (1-a)*ln(l)
    d['implied_y'] = (d['alpha'] / 100) * d['ln_nks'] + (1 - d['alpha'] / 100) * d['ln_lforce']
    d['ln_tfp'] = d['ln_gdp'] - d['implied_y']  # ln(tfp)

    # TFP trend
    cycle, trend = sm.filters.hpfilter(d.loc[~d['ln_tfp'].isna(), 'ln_tfp'], lamb=11200)  # deals with NA
    d['ln_tfp_trend'] = trend  # don't replace original with trend component

    # Calculate potential output
    d['ln_po'] = d['ln_tfp_trend'] + \
                 ((d['alpha'] / 100) * d['ln_nks_trend']) + \
                 (1 - d['alpha'] / 100) * d['ln_lforce_trend']

    # Back out levels (PO)
    d['po'] = np.exp(d['ln_po'])
    d['output_gap'] = 100 * (d['gdp'] / d['po'] - 1)  # % PO
    d['capital_input'] = np.exp(d['ln_nks_trend'])
    d['labour_input'] = np.exp(d['ln_lforce_trend'])
    d['tfp_input'] = np.exp(d['ln_tfp_trend'])

    # Back out levels (observed output)
    d['capital_observed'] = np.exp(d['ln_nks'])
    d['labour_observed'] = np.exp(d['ln_lforce'])
    d['tfp_observed'] = np.exp(d['ln_tfp'])

    # trim data frame
    list_col_keep = ['output_gap', 'po', 'gdp',
                     'capital_input', 'labour_input', 'tfp_input',
                     'capital_observed', 'labour_observed', 'tfp_observed',
                     'alpha']
    d = d[list_col_keep]

    return d


def prodfunc_histdecomp(input):
    print('\n Uses output dataframe from prodfunc_po() ' +
          'to calculate the historical decomposition of potential and actual output')
    d = input.copy()

    # Calculate YoY growth
    list_levels = ['po', 'gdp',
                   'capital_input', 'labour_input', 'tfp_input',
                   'capital_observed', 'labour_observed', 'tfp_observed']
    list_yoy = [i + '_yoy' for i in list_levels]
    for i, j in zip(list_levels, list_yoy):
        d[j] = 100 * ((d[i] / d[i].shift(4)) - 1)

    # Decompose potential output
    d['capital_cont_po'] = d['capital_input_yoy'] * (d['alpha'] / 100)
    d['labour_cont_po'] = d['labour_input_yoy'] * (1 - (d['alpha'] / 100))
    d['tfp_cont_po'] = d['po_yoy'] - d['capital_cont_po'] - d['labour_cont_po']

    # Decompose observed output
    d['capital_cont_observed'] = d['capital_observed_yoy'] * (d['alpha'] / 100)
    d['labour_cont_observed'] = d['labour_observed_yoy'] * (1 - (d['alpha'] / 100))
    d['tfp_cont_observed'] = d['gdp_yoy'] - d['capital_cont_observed'] - d['labour_cont_observed']

    return d


df_pf = prodfunc_po(data=df)
df_pf_histdecomp = prodfunc_histdecomp(df_pf)

# V --- Export data frames

df_pf = df_pf.reset_index()
df_pf['quarter'] = df_pf['quarter'].astype('str')
df_pf.to_parquet('boombustpo_estimates_pf_onesided' + file_suffix_fcast + '.parquet', compression='brotli')

df_pf_histdecomp = df_pf_histdecomp.reset_index()
df_pf_histdecomp['quarter'] = df_pf_histdecomp['quarter'].astype('str')
df_pf_histdecomp.to_parquet('boombustpo_estimates_pf_onesided_hd' + file_suffix_fcast + '.parquet',
                            compression='brotli')

# VI --- Notify
telsendmsg(conf=tel_config,
           msg='boombustpo_compute_pf_onesided: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
