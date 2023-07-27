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
t_start = '1995Q4'
t_output_start = str(pd.to_datetime(t_start).to_period('Q') + 26)  # 26Q burn-in
t_start_plus1 = str(pd.to_datetime(t_start).to_period('Q') + 1)  # 1Q after start of time series
t_now = str(pd.to_datetime(str(date.today())).to_period('Q'))
list_t_ends = ['2007Q2', '2008Q2', '2009Q3', '2015Q4', '2019Q4', '2022Q4']
list_colours = ['lightcoral', 'crimson', 'red', 'steelblue', 'darkblue', 'gray']
list_dash_styles = ['solid', 'solid', 'solid', 'solid', 'solid', 'solid']
dict_revision_pairs = {'2009Q3': '2007Q2',
                       '2019Q4': '2015Q4',
                       '2022Q4': '2019Q4'}

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
df = pd.read_parquet('boombustpo_estimates_vintages_onesided_kfonly.parquet')  # PF ONLY
df['quarter'] = pd.to_datetime(df['quarter']).dt.to_period('Q')
df = df[df['quarter'] >= t_output_start]  # avoid burn-in period for kf
df = df.set_index('quarter')

df_rev = pd.DataFrame(columns=['rev_consol'])
round = 1
for post, pre in tqdm(dict_revision_pairs.items()):
    df_rev['rev_' + post + '_' + pre] = df[post] - df[pre]
    if round == 1:
        df_rev['rev_consol'] = df_rev['rev_' + post + '_' + pre].copy()
    elif round > 1:
        df_rev.loc[df_rev['rev_consol'].isna(), 'rev_consol'] = df_rev['rev_' + post + '_' + pre]
    round += 1
df_rev = df_rev[df_rev.index <= list_t_ends[-2]]

# III --- Charts
# vintages

def plot_linechart(data, cols, nice_names, colours, dash_styles, y_axis_title, main_title, output_suffix):
    fig = go.Figure()
    for col, nice_name, colour, dash_style in tqdm(zip(cols, nice_names, colours, dash_styles)):
        fig.add_trace(
            go.Scatter(
                x=data.index.astype('str'),
                y=data[col],
                name=nice_name,
                mode='lines',
                line=dict(color=colour, dash=dash_style)
            )
        )
    fig.update_layout(title=main_title,
                      yaxis_title=y_axis_title,
                      plot_bgcolor='white',
                      hovermode='x',
                      font=dict(size=20, color='black'))
    fig.write_image('Output/BoomBustPO_Vintage_OneSided_KFOnly_' + output_suffix + '.png', height=768, width=1366)
    fig.write_html('Output/BoomBustPO_Vintage_OneSided_KFOnly_' + output_suffix + '.html')
    return fig


df['ref'] = 0

fig_vintages = plot_linechart(
    data=df,
    cols=list_t_ends + ['ref'],
    nice_names=list_t_ends + ['Reference (Y=0)'],
    colours=list_colours + ['black'],
    dash_styles=list_dash_styles + ['solid'],
    y_axis_title='% Potential Output',
    main_title='Vintages of Current Output Gap Estimates (KF Only (One-Sided))',
    output_suffix='OutputGap'
)
telsendimg(conf=tel_config,
           path='Output/BoomBustPO_Vintage_OneSided_KFOnly_OutputGap.png',
           cap='Vintages of Current Output Gap Estimates (KF Only (One-Sided))')

# revisions


def plot_areachart(data, cols, nice_names, colours, y_axis_title, main_title, show_legend, ymin, ymax, output_suffix):
    fig = go.Figure()
    for col, nice_name, colour in tqdm(zip(cols, nice_names, colours)):
        fig.add_trace(
            go.Scatter(
                x=data.index.astype('str'),
                y=data[col],
                name=nice_name,
                mode='none',
                fill='tonexty',  # tozeroy
                fillcolor=colour,
            )
        )
    fig.update_yaxes(range=[ymin, ymax])
    fig.update_layout(title=main_title,
                      yaxis_title=y_axis_title,
                      plot_bgcolor='white',
                      hovermode='x',
                      font=dict(size=20, color='black'),
                      showlegend=show_legend)
    fig.write_image('Output/BoomBustPO_Vintage_OneSided_KFOnly_' + output_suffix + '.png', height=768, width=1366)
    fig.write_html('Output/BoomBustPO_Vintage_OneSided_KFOnly_' + output_suffix + '.html')
    return fig


fig_rev = plot_areachart(
    data=df_rev,
    cols=['rev_consol'],
    nice_names=['Revisions'],
    colours=['lightcoral'],
    y_axis_title='Percentage Points (% Potential Output)',
    main_title='Revisions in Current Output Gap Across Consecutive Vintages (KF Only (One-Sided))',
    show_legend=False,
    ymin=-5,
    ymax=5,
    output_suffix='OutputGap_Revisions'
)
telsendimg(conf=tel_config,
           path='Output/BoomBustPO_Vintage_OneSided_KFOnly_OutputGap_Revisions.png',
           cap='Revisions in Current Output Gap Across Consecutive Vintages (KF Only (One-Sided))')


# IV --- Notify
telsendmsg(conf=tel_config,
           msg='boombustpo_plot_vintages_onesided_kfonly: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
