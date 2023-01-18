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

time_start = time.time()

# 0 --- Main settings
tel_config = 'EcMetrics_Config_GeneralFlow.conf'
T_lb = '1995Q1'
T_lb_day = date(1995, 1, 1)
show_conf_bands = False
use_forecast = False  # public or internal use

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
df_hd = pd.read_parquet('pluckingpo_estimates_pf_hd.parquet')
df_hd['quarter'] = pd.to_datetime(df_hd['quarter']).dt.to_period('Q')
df_hd = df_hd.set_index('quarter')

# III --- Charts


def plot_histdecomp(input):
    d = input.copy()
    list_col_keep = ['gdp_yoy', 'gdp_ceiling_yoy',
                     'capital_cont_ceiling', 'labour_cont_ceiling', 'tfp_cont_ceiling',
                     'capital_cont_observed', 'labour_cont_observed', 'tfp_cont_observed',]
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


# IV --- Notify
telsendmsg(conf=tel_config,
           msg='pluckingpo_plot_histdecomp: COMPLETED')


# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
