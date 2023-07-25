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
df_pd = pd.read_parquet('pluckingpo_dns_estimates_pf' + file_suffix_fcast + '.parquet')
df_pd['quarter'] = pd.to_datetime(df_pd['quarter']).dt.to_period('Q')
df_pd = df_pd.set_index('quarter')

df_bb = pd.read_parquet('boombustpo_estimates_kf_onesided' + file_suffix_fcast + '.parquet')
df_bb['quarter'] = pd.to_datetime(df_bb['quarter']).dt.to_period('Q')
df_bb = df_bb.set_index('quarter')

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

# III --- Charts
if show_conf_bands:
    col_gdp = ['gdp', 'gdp_ceiling', 'gdp_ceiling_lb']
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
    col_gdp = ['gdp', 'gdp_ceiling']
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


# Ceilings and observed + boom-bust version of output gap


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


fig_gdp = plot_linechart(data=df_pd,
                         cols=col_gdp,
                         nice_names=col_gdp_nice,
                         colours=colour1,
                         dash_styles=dash1,
                         y_axis_title='ppt',
                         main_title='Real GDP: Observed and Ceiling',
                         output_suffix='GDP',
                         use_forecast_choice=use_forecast)
fig_lab = plot_linechart(data=df_pd,
                         cols=col_lab,
                         nice_names=col_lab_nice,
                         colours=colour1,
                         dash_styles=dash1,
                         y_axis_title='ppt',
                         main_title='Labour: Observed and Ceiling',
                         output_suffix='Labour',
                         use_forecast_choice=use_forecast)
fig_cap = plot_linechart(data=df_pd,
                         cols=col_cap,
                         nice_names=col_cap_nice,
                         colours=colour1,
                         dash_styles=dash1,
                         y_axis_title='ppt',
                         main_title='Capital: Observed and Ceiling',
                         output_suffix='Capital',
                         use_forecast_choice=use_forecast)
fig_tfp = plot_linechart(data=df_pd,
                         cols=col_tfp,
                         nice_names=col_tfp_nice,
                         colours=colour1,
                         dash_styles=dash1,
                         y_axis_title='ppt',
                         main_title='TFP: Observed and Ceiling',
                         output_suffix='TFP',
                         use_forecast_choice=use_forecast)
fig_og = plot_linechart(data=df_og,
                        cols=col_og,
                        nice_names=col_og_nice,
                        colours=colour2,
                        dash_styles=dash2,
                        y_axis_title='% Potential Output',
                        main_title='Output Gap: Boom-Bust and Plucking',
                        output_suffix='OG',
                        use_forecast_choice=use_forecast)
fig_og_norm = plot_linechart(data=df_og,
                             cols=col_og_norm,
                             nice_names=col_og_norm_nice,
                             colours=colour3,
                             dash_styles=dash3,
                             y_axis_title='Index',
                             main_title='Normalised Output Gap: Boom-Bust and Plucking',
                             output_suffix='OG_Norm',
                             use_forecast_choice=use_forecast)

suffix_figs = ['GDP', 'Labour', 'Capital', 'TFP', 'OG', 'OG_Norm']
for i in suffix_figs:
    telsendimg(conf=tel_config,
               path='Output/PluckingPO_ObsCeiling_' + i + file_suffix_fcast + '.png',
               cap=i + ' (observed and ceiling)')

# IV --- Notify
telsendmsg(conf=tel_config,
           msg='pluckingpo_plot_ceiling_po_og: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
