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
df = pd.read_parquet('pluckingpo_estimates' + file_suffix_fcast + '.parquet')
df['quarter'] = pd.to_datetime(df['quarter']).dt.to_period('Q')
df = df.set_index('quarter')

df_nobound = pd.read_parquet('pluckingpo_estimates_nobounds' + file_suffix_fcast + '.parquet')
df_nobound['quarter'] = pd.to_datetime(df_nobound['quarter']).dt.to_period('Q')
df_nobound = df_nobound.set_index('quarter')


# III --- Charts
# Bounded and unbounded ceiling estimates
fig_bounds = go.Figure()
fig_bounds.add_trace(
        go.Scatter(
            x=df_nobound.index.astype('str'),
            y=df_nobound['ln_gdp_ceiling'],
            name='Without',
            mode='lines',
            line=dict(color='black', width=1)
        )
    )
fig_bounds.add_trace(
        go.Scatter(
            x=df.index.astype('str'),
            y=df['ln_gdp_ceiling'],
            name='With',
            mode='lines',
            line=dict(color='crimson', width=1)
        )
    )
if use_forecast:
    max_everything = pd.concat([df['ln_gdp_ceiling'], df_nobound['ln_gdp_ceiling']]).max().max()
    min_everything = pd.concat([df['ln_gdp_ceiling'], df_nobound['ln_gdp_ceiling']]).min().min()
    df['_shadetop'] = max_everything  # max of entire dataframe
    df.loc[df.index < fcast_start, '_shadetop'] = 0
    fig_bounds.add_trace(
        go.Bar(
            x=df.index.astype('str'),
            y=df['_shadetop'],
            name='Forecast',
            width=1,
            marker=dict(color='lightgrey', opacity=0.3)
        )
    )
    if bool(min_everything < 0):  # To avoid double shades
        df['_shadebtm'] = min_everything.min()  # min of entire dataframe
        df.loc[df.index < fcast_start, '_shadebtm'] = 0
        fig_bounds.add_trace(
            go.Bar(
                x=df.index.astype('str'),
                y=df['_shadebtm'],
                name='Forecast',
                width=1,
                marker=dict(color='lightgrey', opacity=0.3)
            )
        )
    fig_bounds.update_yaxes(range=[min_everything, max_everything])
fig_bounds.update_layout(title='Estimates of Output Ceiling With and Without Hard Upper Bound',
                  yaxis_title='Natural Logs',
                  plot_bgcolor='white',
                  hovermode='x',
                  barmode='relative',
                  font=dict(size=20, color='black'))
fig_bounds.write_image('Output/' + 'PluckingPO_HardAndNoBound' + file_suffix_fcast + '.png', height=768, width=1366)
fig_bounds.write_html('Output/' + 'PluckingPO_HardAndNoBound' + file_suffix_fcast + '.html')
telsendimg(conf=tel_config,
           path='Output/PluckingPO_HardAndNoBound' + file_suffix_fcast + '.png',
           cap='Estimates of Output Ceiling With and Without Hard Upper Bound')

# Difference between bounded and unbounded ceiling estimates
df_nobound['bound_nobound_ceiling_diff'] = df_nobound['ln_gdp_ceiling'] - df['ln_gdp_ceiling']
fig_bounds = go.Figure()
fig_bounds.add_trace(
    go.Scatter(
        x=df_nobound.index.astype('str'),
        y=df_nobound['bound_nobound_ceiling_diff'],
        name='Without - With',
        mode='lines',
        line=dict(color='lightcoral', width=1),
        fill='tozeroy'
    )
)
if use_forecast:
    max_everything = df_nobound['bound_nobound_ceiling_diff'].max()
    min_everything = df_nobound['bound_nobound_ceiling_diff'].min()
    df['_shadetop'] = max_everything  # max of entire dataframe
    df.loc[df.index < fcast_start, '_shadetop'] = 0
    fig_bounds.add_trace(
        go.Bar(
            x=df.index.astype('str'),
            y=df['_shadetop'],
            name='Forecast',
            width=1,
            marker=dict(color='lightgrey', opacity=0.3)
        )
    )
    if bool(min_everything < 0):  # To avoid double shades
        df['_shadebtm'] = min_everything.min()  # min of entire dataframe
        df.loc[df.index < fcast_start, '_shadebtm'] = 0
        fig_bounds.add_trace(
            go.Bar(
                x=df.index.astype('str'),
                y=df['_shadebtm'],
                name='Forecast',
                width=1,
                marker=dict(color='lightgrey', opacity=0.3)
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
fig_bounds.write_image('Output/' + 'PluckingPO_HardAndNoBound_Diff' + file_suffix_fcast + '.png', height=768, width=1366)
fig_bounds.write_html('Output/' + 'PluckingPO_HardAndNoBound_Diff' + file_suffix_fcast + '.html')
telsendimg(conf=tel_config,
           path='Output/PluckingPO_HardAndNoBound_Diff' + file_suffix_fcast + '.png',
           cap='Difference Between Estimates of Output Ceiling With and Without Hard Upper Bound')

# IV --- Notify
telsendmsg(conf=tel_config,
           msg='pluckingpo_plot_bound_versus_nobound: COMPLETED')


# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
