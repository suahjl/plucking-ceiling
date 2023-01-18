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

df_pd = pd.read_parquet('pluckingpo_estimates_pf.parquet')
df_pd['quarter'] = pd.to_datetime(df_pd['quarter']).dt.to_period('Q')
df_pd = df_pd.set_index('quarter')

# setup
df_recov = pd.concat([pd.DataFrame(df['ln_gdp_cepi']).rename(columns={'ln_gdp_cepi': 'cepi'}),
                      pd.DataFrame(df_pd['output_gap'])],
                     axis=1)
df_recov = df_recov[df_recov['cepi'] >= 1]
# move cepi up by 1 row
df_recov['cepi'] = df_recov['cepi'].shift(-1)
df_recov['cepi'] = df_recov['cepi'].fillna(method='ffill')
df_recov['cepi'] = df_recov['cepi'].astype('int')
# Setup data frame
list_crises=['AFC', 'GFC', 'COVID-19']
df_recov_consol = pd.DataFrame(columns=list_crises)
# Wrangle
r = 1
for cepi, crises in zip(range(df_recov['cepi'].min(), df_recov['cepi'].max() + 1), list_crises):
    d = df_recov[df_recov['cepi'] == cepi].reset_index(drop=True)
    d = pd.DataFrame(d['output_gap']).rename(columns={'output_gap': crises})
    if r == 1:
        df_recov_consol = d.copy()
    elif r > 1:
        df_recov_consol = pd.concat([df_recov_consol, d], axis=1)
    r += 1
for crises in list_crises:
    df_recov_consol[crises] = df_recov_consol[crises] - df_recov_consol.loc[df_recov_consol.index == 0, crises][0]
df_recov_consol['ref'] = 0
df_recov_short = df_recov_consol.head(28)

# III --- Charts


def plot_linechart(data, cols, nice_names, colours, dash_styles, y_axis_title, main_title, output_suffix):
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
    fig.update_layout(title=main_title,
                      yaxis_title=y_axis_title,
                      plot_bgcolor='white',
                      hovermode='x',
                      barmode='relative',
                      font=dict(size=20, color='black'))
    fig.write_image('Output/PluckingPO_ObsCeiling_' + output_suffix + '.png', height=768, width=1366)
    fig.write_html('Output/PluckingPO_ObsCeiling_' + output_suffix + '.html')
    return fig


# Plot
fig_recov = plot_linechart(
    data=df_recov_short,
    cols=list_crises + ['ref'],
    nice_names=list_crises + ['Reference (Pre-Crisis Peak)'],
    colours=['crimson', 'darkblue', 'black', 'black'],
    dash_styles=['solid', 'solid', 'solid', 'dash'],
    y_axis_title='Percentage Points (0 = Pre-Crisis Peak)',
    main_title='Evolution of Plucking Output Gap Post-Crises',
    output_suffix='CrisisRecoveries'
)
telsendimg(conf=tel_config,
           path='Output/PluckingPO_ObsCeiling_CrisisRecoveries.png',
           cap='Evolution of Plucking Output Gap Post-Crises')


# IV --- Notify
telsendmsg(conf=tel_config,
           msg='pluckingpo_plot_crisisrecoveries: COMPLETED')


# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
