# -------------- Allow PO to deviate from first guess based on K and N
# -------------- Casts the bands as 'uncertainty in timing of peaks by 1Q earlier'
# -------------- Option to show or hide confidence bands
# -------------- Option to not show any forecasts
# -------------- Open data version (est3)
# -------------- Only for monthly unemployment rate
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
df = pd.read_parquet('pluckingpo_dns_urate_usa_estimates' + file_suffix_fcast + '.parquet')
df['month'] = pd.to_datetime(df['month']).dt.to_period('M')
df = df.set_index('month')

# III --- Charts
if show_conf_bands:
    col_urate = ['urate', 'urate_ceiling', 'urate_ceiling_lb']
    col_urate_nice = ['Observed U-Rate', 'U-Rate Floor UB', 'U-Rate Floor LB']
    colour1 = ['crimson', 'black', 'black']
    dash1 = ['solid', 'longdash', 'longdash']
elif not show_conf_bands:
    col_urate = ['urate', 'urate_ceiling']
    col_urate_nice = ['Observed U-Rate', 'U-Rate Floor']
    colour1 = ['crimson', 'black']
    dash1 = ['solid', 'longdash']


# Ceilings / floors and observed


def plot_linechart(
        data, cols, nice_names, colours, dash_styles,
        y_axis_title, main_title,
        output_suffix, use_forecast_choice
):
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
    if use_forecast_choice:
        max_everything = d[cols].max().max()
        min_everything = d[cols].min().min()
        d['_shadetop'] = max_everything  # max of entire dataframe
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
        if bool(min_everything < 0):  # To avoid double shades
            d['_shadebtm'] = min_everything.min()  # min of entire dataframe
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
        fig.update_yaxes(range=[min_everything, max_everything])
    fig.update_layout(title=main_title,
                      yaxis_title=y_axis_title,
                      plot_bgcolor='white',
                      hovermode='x',
                      barmode='relative',
                      font=dict(size=20, color='black'))
    fig.write_image('Output/PluckingPO_ObsCeiling_' + output_suffix + file_suffix_fcast + '.png',
                    height=768, width=1366)
    fig.write_html('Output/PluckingPO_ObsCeiling_' + output_suffix + file_suffix_fcast + '.html')
    return fig


fig_gdp = plot_linechart(data=df,
                         cols=col_urate,
                         nice_names=col_urate_nice,
                         colours=colour1,
                         dash_styles=dash1,
                         y_axis_title='ppt',
                         main_title='[US] U-Rate: Observed and Floor',
                         output_suffix='URate_USA',
                         use_forecast_choice=use_forecast)

suffix_figs = ['URate_USA']
for i in suffix_figs:
    telsendimg(conf=tel_config,
               path='Output/PluckingPO_ObsCeiling_' + i + file_suffix_fcast + '.png',
               cap=i + ' (observed and floor)')

# IV --- Notify
telsendmsg(conf=tel_config,
           msg='pluckingpo_plot_ceiling_urate_usa: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
