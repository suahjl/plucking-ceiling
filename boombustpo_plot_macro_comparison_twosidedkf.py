# ---------------- Compares the boom-bust output gap against a selection of macro indicators ----------------

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
from ceic_api_client.pyceic import Ceic
from dotenv import load_dotenv
import os
import ast

time_start = time.time()

# 0 --- Main settings
load_dotenv()
tel_config = os.getenv('TEL_CONFIG')
T_lb = '1995Q4'
T_ub = '2022Q4'
burn_in_count = 26
t_burnin = str(pd.to_datetime(T_lb).to_period('Q') + burn_in_count)  # 26Q burn-in
list_T_outliers = ['1997Q4', '1998Q1', '1998Q2', '1998Q3', '1998Q4',
                   '1999Q1', '1999Q2', '1999Q3', '1999Q4',
                   '2008Q3', '2008Q4', '2009Q1', '2009Q2', '2009Q3', '2009Q4',
                   '2020Q1', '2020Q2', '2020Q3', '2020Q4',
                   '2021Q1', '2021Q2', '2021Q3', '2021Q4']


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


def ceic2pandas_ts(input, start_date):  # input should be a list of CEIC Series IDs
    for m in range(len(input)):
        try:
            input.remove(np.nan)  # brute force remove all np.nans from series ID list
        except:
            print('no more np.nan')
    k = 1
    for i in tqdm(input):
        series_result = Ceic.series(i, start_date=start_date)  # retrieves ceicseries
        y = series_result.data
        series_name = y[0].metadata.name  # retrieves name of series
        time_points_dict = dict((tp._date, tp.value) for tp in y[0].time_points)  # this is a list of 1 dictionary,
        series = pd.Series(time_points_dict)  # convert into pandas series indexed to timepoints
        if k == 1:
            frame_consol = pd.DataFrame(series)
            frame_consol = frame_consol.rename(columns={0: series_name})
        elif k > 1:
            frame = pd.DataFrame(series)
            frame = frame.rename(columns={0: series_name})
            frame_consol = pd.concat([frame_consol, frame], axis=1)  # left-right concat on index (time)
        elif k < 1:
            raise NotImplementedError
        k += 1
        frame_consol = frame_consol.sort_index()
    return frame_consol


# II --- Data
# CEIC
Ceic.login(os.getenv('CEIC_USERNAME'), os.getenv('CEIC_PASSWORD'))  # login to CEIC
series_id = pd.read_csv('seriesids_macrocomparison.txt', sep=',')
series_id = list(series_id['series_id'])
df_ceic = ceic2pandas_ts(series_id, start_date=date(1991, 1, 1)).fillna(method='ffill')
dict_ceic_rename = {'Consumer Price Index: YoY: Quarterly: Malaysia': 'cpi_yoy',
                    'Core CPI: YoY: Quarterly: Malaysia': 'cpi_core_yoy',
                    'Producer Price Index: YoY: Quarterly: Malaysia': 'ppi_yoy',
                    '(DC)Producer Price Index: YoY': 'ppi_old_yoy',
                    'Unemployment Rate': 'ur',
                    'MIER: Capacity Utilization Rate: Month Average': 'caputil',
                    'Industrial Production Index: YoY: Quarterly: Malaysia': 'ipi_yoy',
                    'M2: YoY: Quarterly: Malaysia': 'm2_yoy',
                    'Real GDP: YoY: Malaysia': 'rgdp_yoy'}
df_ceic = df_ceic.rename(columns=dict_ceic_rename)
df_ceic = df_ceic.reset_index()
df_ceic = df_ceic.rename(columns={'index': 'quarter'})
df_ceic['quarter'] = pd.to_datetime(df_ceic['quarter']).dt.to_period('Q')
df_ceic = df_ceic.set_index('quarter')
# df_ceic = df_ceic.groupby('quarter').agg('mean')  # Don't need if all series are already quarterly

# Unemployment YoY
df_ceic['ur_yoy'] = df_ceic['ur'] - df_ceic['ur'].shift(4)  # year-on-year difference

# Old CPI Core series
df_cpi_core_old = pd.read_csv('old_static_cpi_core.txt', sep=',')
df_cpi_core_old['quarter'] = pd.to_datetime(df_cpi_core_old['quarter']).dt.to_period('Q')
df_cpi_core_old = df_cpi_core_old.set_index('quarter')
df_cpi_core_old['cpi_core_old_yoy'] = ((df_cpi_core_old['cpi_core'] / df_cpi_core_old['cpi_core'].shift(4)) - 1) * 100
del df_cpi_core_old['cpi_core']

# Output gap
df_bb = pd.read_parquet('boombustpo_estimates_kf.parquet')  # twosided
df_bb['quarter'] = pd.to_datetime(df_bb['quarter']).dt.to_period('Q')
df_bb = df_bb[(df_bb['quarter'] >= T_lb) & (df_bb['quarter'] <= T_ub)]
df_bb = df_bb.set_index('quarter')
df_bb = df_bb.sort_index()
df_bb = pd.DataFrame(df_bb['output_gap_kf']).rename(columns={'output_gap_kf': 'output_gap'})

# Clear burn-in period
df_bb.loc[df_bb.index <= t_burnin, 'output_gap'] = np.nan

# Merging
df = pd.concat([df_bb, df_ceic, df_cpi_core_old], axis=1)
df = df.sort_index()

# Splicing / filling backseries
df.loc[df['ppi_yoy'].isna(), 'ppi_yoy'] = df['ppi_old_yoy']
df.loc[df['cpi_core_yoy'].isna(), 'cpi_core_yoy'] = df['cpi_core_old_yoy']
del df['ppi_old_yoy']
del df['cpi_core_old_yoy']

# Timebound
df = df[(df.index >= T_lb) & (df.index <= T_ub)]

# Generate Lags
# Define contemporaneous
list_y_cols = ['cpi_yoy', 'cpi_core_yoy', 'ppi_yoy', 'ur', 'caputil', 'ipi_yoy', 'm2_yoy']
list_y_nice_names = ['CPI YoY', 'Core CPI YoY', 'PPI YoY',
                     'Unemployment Rate', 'Capacity Utilisation', 'IPI YoY', 'M2 YoY']
# Lags
list_y_cols_lag1 = [i + '_lag1' for i in list_y_cols]
list_y_nice_names_lag1 = [i + ' (1Q Lag)' for i in list_y_nice_names]
for contemp, lag in tqdm(zip(list_y_cols, list_y_cols_lag1)):
    df[lag] = df[contemp].shift(1)

list_y_cols_lag2 = [i + '_lag2' for i in list_y_cols]
list_y_nice_names_lag2 = [i + ' (2Q Lag)' for i in list_y_nice_names]
for contemp, lag in tqdm(zip(list_y_cols, list_y_cols_lag2)):
    df[lag] = df[contemp].shift(2)

# Outliers
df_exoutliers = df[~(df.index.astype('str').isin(list_T_outliers))]  # separate df for plotting


# III --- Plots


def scatterplots(data_full,
                 data_ex_outliers,
                 x_cols, y_cols,
                 x_nice_names,
                 y_nice_names,
                 colours_core,
                 colours_outliers,
                 marker_sizes,
                 line_widths,
                 main_titles,
                 output_suffixes,
                 show_ols):
    for x_col, y_col, x_nice_name, y_nice_name, colour_core, colour_outlier, marker_size, line_width, main_title, output_suffix \
            in \
            tqdm(zip(x_cols, y_cols, x_nice_names, y_nice_names,
                     colours_core, colours_outliers,
                     marker_sizes, line_widths,
                     main_titles, output_suffixes)):

        # Preliminaries
        d_full = data_full.copy()
        d_full = d_full.dropna(subset=[y_col, x_col])  # drop rows where there are missing values

        d_exoutliers = data_ex_outliers.copy()
        d_exoutliers = d_exoutliers.dropna(subset=[y_col, x_col])  # drop rows where there are missing values

        # Reset figure
        fig = go.Figure()

        # Plot full data (plot this first so the outliers are shown)
        fig.add_trace(
            go.Scatter(
                x=d_full[x_col],
                y=d_full[y_col],
                name=y_nice_name,
                mode='markers',
                marker=dict(color=colour_outlier, size=marker_size)
            )
        )
        if show_ols:
            eqn_bfit = y_col + ' ~ ' + x_col
            est_bfit = smf.ols(eqn_bfit, data=d_full).fit()
            pred_bfit = est_bfit.predict()  # check if arguments are needed
            d_full['_pred_full'] = pred_bfit
            fig.add_trace(
                go.Scatter(
                    x=d_full[x_col],
                    y=d_full['_pred_full'],
                    mode='lines',
                    line=dict(color=colour_outlier, width=line_width),
                    showlegend=False
                )
            )

        # Plot subset (overlay this, so that the core obs are shown on top)
        fig.add_trace(
            go.Scatter(
                x=d_exoutliers[x_col],
                y=d_exoutliers[y_col],
                name=y_nice_name,
                mode='markers',
                marker=dict(color=colour_core, size=marker_size)
            )
        )
        if show_ols:
            eqn_bfit = y_col + ' ~ ' + x_col
            est_bfit = smf.ols(eqn_bfit, data=d_exoutliers).fit()
            pred_bfit = est_bfit.predict()  # check if arguments are needed
            d_exoutliers['_pred_core'] = pred_bfit
            fig.add_trace(
                go.Scatter(
                    x=d_exoutliers[x_col],
                    y=d_exoutliers['_pred_core'],
                    mode='lines',
                    line=dict(color=colour_core, width=line_width),
                    showlegend=False
                )
            )

        # Layout
        fig.update_layout(title=main_title,
                          yaxis_title=y_nice_name,
                          xaxis_title=x_nice_name,
                          plot_bgcolor='white',
                          hovermode='x',
                          font=dict(size=30, color='black'),
                          showlegend=False)
        fig.write_image('Output/BoomBustPO_TwoSidedKF_MacroComparison_' + output_suffix + '.png',
                        height=768, width=1366)
        fig.write_html('Output/BoomBustPO_TwoSidedKF_MacroComparison_' + output_suffix + '.html')


# Plot contemporaneous
list_main_titles = ['Boom-Bust Output Gap versus ' + i for i in list_y_nice_names]
list_suffixes = list_y_cols.copy()
scatterplots(
    data_full=df,
    data_ex_outliers=df_exoutliers,
    x_cols=['output_gap'] * 7,
    y_cols=list_y_cols,
    x_nice_names=['Boom-Bust Output Gap %'] * 7,
    y_nice_names=list_y_nice_names,
    colours_core=['darkblue'] * 7,
    colours_outliers=['grey'] * 7,
    marker_sizes=[16] * 7,
    line_widths=[3] * 7,
    main_titles=list_main_titles,
    output_suffixes=list_suffixes,
    show_ols=True
)
for i, j in tqdm(zip(list_suffixes, list_main_titles)):
    telsendimg(conf=tel_config,
               path='Output/BoomBustPO_TwoSidedKF_MacroComparison_' + i + '.png',
               cap=j)

# Plot lag1
time.sleep(15)
list_main_titles = ['Boom-Bust Output Gap versus ' + i for i in list_y_nice_names_lag1]
list_suffixes = list_y_cols_lag1.copy()
scatterplots(
    data_full=df,
    data_ex_outliers=df_exoutliers,
    x_cols=['output_gap'] * 7,
    y_cols=list_y_cols,
    x_nice_names=['Boom-Bust Output Gap %'] * 7,
    y_nice_names=list_y_nice_names,
    colours_core=['darkblue'] * 7,
    colours_outliers=['grey'] * 7,
    marker_sizes=[16] * 7,
    line_widths=[3] * 7,
    main_titles=list_main_titles,
    output_suffixes=list_suffixes,
    show_ols=True
)
for i, j in tqdm(zip(list_suffixes, list_main_titles)):
    telsendimg(conf=tel_config,
               path='Output/BoomBustPO_TwoSidedKF_MacroComparison_' + i + '.png',
               cap=j)

# Plot lag2
time.sleep(15)
list_main_titles = ['Boom-Bust Output Gap versus ' + i for i in list_y_nice_names_lag2]
list_suffixes = list_y_cols_lag2.copy()
scatterplots(
    data_full=df,
    data_ex_outliers=df_exoutliers,
    x_cols=['output_gap'] * 7,
    y_cols=list_y_cols,
    x_nice_names=['Boom-Bust Output Gap %'] * 7,
    y_nice_names=list_y_nice_names,
    colours_core=['darkblue'] * 7,
    colours_outliers=['grey'] * 7,
    marker_sizes=[16] * 7,
    line_widths=[3] * 7,
    main_titles=list_main_titles,
    output_suffixes=list_suffixes,
    show_ols=True
)
for i, j in tqdm(zip(list_suffixes, list_main_titles)):
    telsendimg(conf=tel_config,
               path='Output/BoomBustPO_TwoSidedKF_MacroComparison_' + i + '.png',
               cap=j)

# IV --- Notify
telsendmsg(conf=tel_config,
           msg='boombustpo_plot_macro_comparison_twosidedkf: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
