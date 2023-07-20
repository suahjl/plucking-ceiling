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
df_update_ceiling = pd.read_parquet('pluckingpo_updateceiling' + file_suffix_fcast + '.parquet')
df_update_ceiling['quarter'] = pd.to_datetime(df_update_ceiling['quarter']).dt.to_period('Q')
df_update_ceiling = df_update_ceiling.set_index('quarter')

# III --- Charts
# Update of output ceiling using K and N
fig_update_ceiling_decomp = go.Figure()
fig_update_ceiling_decomp.add_trace(
        go.Scatter(
            x=df_update_ceiling.index.astype('str'),
            y=df_update_ceiling['ln_gdp_ceiling_initial'],
            name='Initial Estimate',
            mode='lines',
            line=dict(color='black', width=1)
        )
    )
fig_update_ceiling_decomp.add_trace(
    go.Scatter(
        x=df_update_ceiling.index.astype('str'),
        y=df_update_ceiling['ln_k_rev'] + df_update_ceiling['ln_gdp_ceiling_initial'],
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
        y=df_update_ceiling['ln_n_rev'] + df_update_ceiling['ln_k_rev'] + df_update_ceiling['ln_gdp_ceiling_initial'],
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
            y=df_update_ceiling['ln_gdp_ceiling'],
            name='Final Estimate',
            mode='lines',
            line=dict(color='crimson', width=1)
        )
    )
if use_forecast:
    max_everything = pd.concat([df_update_ceiling['ln_gdp_ceiling'],
                                df_update_ceiling['ln_gdp_ceiling_initial']]).max().max()
    min_everything = pd.concat([df_update_ceiling['ln_gdp_ceiling'],
                                df_update_ceiling['ln_gdp_ceiling_initial']]).min().min()
    df_update_ceiling['_shadetop'] = max_everything  # max of entire dataframe
    df_update_ceiling.loc[df_update_ceiling.index < fcast_start, '_shadetop'] = 0
    fig_update_ceiling_decomp.add_trace(
        go.Bar(
            x=df_update_ceiling.index.astype('str'),
            y=df_update_ceiling['_shadetop'],
            name='Forecast',
            width=1,
            marker=dict(color='lightgrey', opacity=0.3)
        )
    )
    if bool(min_everything < 0):  # To avoid double shades
        df_update_ceiling['_shadebtm'] = min_everything.min()  # min of entire dataframe
        df_update_ceiling.loc[df_update_ceiling.index < fcast_start, '_shadebtm'] = 0
        fig_update_ceiling_decomp.add_trace(
            go.Bar(
                x=df_update_ceiling.index.astype('str'),
                y=df_update_ceiling['_shadebtm'],
                name='Forecast',
                width=1,
                marker=dict(color='lightgrey', opacity=0.3)
            )
        )
    fig_update_ceiling_decomp.update_yaxes(range=[min_everything, max_everything])
fig_update_ceiling_decomp.update_layout(title='Initial and Final Estimates of Log of Output Ceiling with Decomposition',
                  yaxis_title='Natural Logs',
                  plot_bgcolor='white',
                  hovermode='x',
                  barmode='relative',
                  font=dict(size=20, color='black'))
fig_update_ceiling_decomp.write_image('Output/' + 'PluckingPO_UpdateCeiling_Decomp' + file_suffix_fcast + '.png', height=768, width=1366)
fig_update_ceiling_decomp.write_html('Output/' + 'PluckingPO_UpdateCeiling_Decomp' + file_suffix_fcast + '.html')
telsendimg(conf=tel_config,
           path='Output/PluckingPO_UpdateCeiling_Decomp' + file_suffix_fcast + '.png',
           cap='Initial and Final Estimates of the Log of Output Ceiling with Decomposition')


# IV --- Notify
telsendmsg(conf=tel_config,
           msg='pluckingpo_plot_updateceiling: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
