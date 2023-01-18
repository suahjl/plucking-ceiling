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
df = pd.read_parquet('pluckingpo_estimates.parquet')
df['quarter'] = pd.to_datetime(df['quarter']).dt.to_period('Q')
df = df.set_index('quarter')

df_nobound = pd.read_parquet('pluckingpo_estimates_nobounds.parquet')
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
fig_bounds.update_layout(title='Estimates of Output Ceiling With and Without Hard Upper Bound',
                  yaxis_title='Natural Logs',
                  plot_bgcolor='white',
                  hovermode='x',
                  barmode='relative',
                  font=dict(size=20, color='black'))
fig_bounds.write_image('Output/' + 'PluckingPO_HardAndNoBound' + '.png', height=768, width=1366)
fig_bounds.write_html('Output/' + 'PluckingPO_HardAndNoBound' + '.html')
telsendimg(conf=tel_config,
           path='Output/PluckingPO_HardAndNoBound.png',
           cap='Estimates of Output Ceiling With and Without Hard Upper Bound')

# Difference between bounded and unbounded ceiling estimates
fig_bounds = go.Figure()
fig_bounds.add_trace(
    go.Scatter(
        x=df_nobound.index.astype('str'),
        y=df_nobound['ln_gdp_ceiling'] - df['ln_gdp_ceiling'],
        name='Without - With',
        mode='lines',
        line=dict(color='lightcoral', width=1),
        fill='tozeroy'
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
fig_bounds.write_image('Output/' + 'PluckingPO_HardAndNoBound_Diff' + '.png', height=768, width=1366)
fig_bounds.write_html('Output/' + 'PluckingPO_HardAndNoBound_Diff' + '.html')
telsendimg(conf=tel_config,
           path='Output/PluckingPO_HardAndNoBound_Diff.png',
           cap='Difference Between Estimates of Output Ceiling With and Without Hard Upper Bound')

# IV --- Notify
telsendmsg(conf=tel_config,
           msg='pluckingpo_plot_bound_versus_nobound: COMPLETED')


# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
