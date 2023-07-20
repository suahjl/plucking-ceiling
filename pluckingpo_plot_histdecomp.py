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
df_hd = pd.read_parquet('pluckingpo_estimates_pf_hd' + file_suffix_fcast + '.parquet')
df_hd['quarter'] = pd.to_datetime(df_hd['quarter']).dt.to_period('Q')
df_hd = df_hd.set_index('quarter')

# III --- Charts


def plot_histdecomp(input, use_forecast_choice):
    d = input.copy()
    list_col_keep = ['gdp_yoy', 'gdp_ceiling_yoy',
                     'capital_cont_ceiling', 'labour_cont_ceiling', 'tfp_cont_ceiling',
                     'capital_cont_observed', 'labour_cont_observed', 'tfp_cont_observed']
    d = d[list_col_keep]

    # Potential output
    fig_ceiling = go.Figure()
    fig_ceiling.add_trace(go.Scatter(x=d.index.astype('str'),
                                     y=d['gdp_ceiling_yoy'],
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
    if use_forecast_choice:
        max_everything = d[['gdp_ceiling_yoy', 'capital_cont_ceiling', 'labour_cont_ceiling', 'tfp_cont_ceiling']].max().max()
        min_everything = d[['gdp_ceiling_yoy', 'capital_cont_ceiling', 'labour_cont_ceiling', 'tfp_cont_ceiling']].min().min()
        d['_shadetop'] = max_everything  # max of entire dataframe
        d.loc[d.index < fcast_start, '_shadetop'] = 0
        fig_ceiling.add_trace(
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
            fig_ceiling.add_trace(
                go.Bar(
                    x=d.index.astype('str'),
                    y=d['_shadebtm'],
                    name='Forecast',
                    width=1,
                    marker=dict(color='lightgrey', opacity=0.3)
                )
            )
        fig_ceiling.update_yaxes(range=[min_everything, max_everything])
    fig_ceiling.update_layout(title='Historical Decomposition of Ceiling Output YoY Growth',
                              yaxis_title='Percentage Point, %YoY growth',
                              barmode='relative',
                              plot_bgcolor='white',
                              hovermode='x',
                              font=dict(size=20, color='black'))
    fig_ceiling.write_html('Output/PluckingPO_HistDecomp_Ceiling' + file_suffix_fcast + '.html')
    fig_ceiling.write_image('Output/PluckingPO_HistDecomp_Ceiling' + file_suffix_fcast + '.png', height=768, width=1366)

    # Observed output
    fig_ob = go.Figure()
    fig_ob.add_trace(go.Scatter(x=d.index.astype('str'),
                                y=d['gdp_yoy'],
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
    if use_forecast_choice:
        max_everything = d[['gdp_yoy', 'capital_cont_observed', 'labour_cont_observed', 'tfp_cont_observed']].max().max()
        min_everything = d[['gdp_yoy', 'capital_cont_observed', 'labour_cont_observed', 'tfp_cont_observed']].min().min()
        d['_shadetop'] = max_everything  # max of entire dataframe
        d.loc[d.index < fcast_start, '_shadetop'] = 0
        fig_ob.add_trace(
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
            fig_ob.add_trace(
                go.Bar(
                    x=d.index.astype('str'),
                    y=d['_shadebtm'],
                    name='Forecast',
                    width=1,
                    marker=dict(color='lightgrey', opacity=0.3)
                )
            )
        fig_ob.update_yaxes(range=[min_everything, max_everything])
    fig_ob.update_layout(title='Historical Decomposition of Observed Output YoY Growth',
                         yaxis_title='Percentage Point, %YoY growth',
                         barmode='relative',
                         plot_bgcolor='white',
                         hovermode='x',
                         font=dict(size=20, color='black'))
    fig_ob.write_html('Output/PluckingPO_HistDecomp_Obs' + file_suffix_fcast + '.html')
    fig_ob.write_image('Output/PluckingPO_HistDecomp_Obs' + file_suffix_fcast + '.png', height=768, width=1366)

    return fig_ceiling, fig_ob


fig_ceiling, fig_ob = plot_histdecomp(input=df_hd, use_forecast_choice=use_forecast)
for i in ['Output/PluckingPO_HistDecomp_Ceiling' + file_suffix_fcast + '.png',
          'Output/PluckingPO_HistDecomp_Obs' + file_suffix_fcast + '.png']:
    telsendimg(conf=tel_config,
               path=i)


# IV --- Notify
telsendmsg(conf=tel_config,
           msg='pluckingpo_plot_histdecomp: COMPLETED')


# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
