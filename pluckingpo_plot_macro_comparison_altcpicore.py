# ---------------- Compares the Plucking output gap against a selection of macro indicators ----------------

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
        try: input.remove(np.nan)  # brute force remove all np.nans from series ID list
        except: print('no more np.nan')
    k = 1
    for i in tqdm(input):
        series_result = Ceic.series(i, start_date=start_date)  # retrieves ceicseries
        y = series_result.data
        series_name = y[0].metadata.name  # retrieves name of series
        time_points_dict = dict((tp.date, tp.value) for tp in y[0].time_points)  # this is a list of 1 dictionary,
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

# Old core CPI series
df_cpi_core_old = pd.read_csv('old_static_cpi_core.txt', sep=',')
df_cpi_core_old['quarter'] = pd.to_datetime(df_cpi_core_old['quarter']).dt.to_period('Q')
df_cpi_core_old = df_cpi_core_old.set_index('quarter')
df_cpi_core_old['cpi_core_old_yoy'] = ((df_cpi_core_old['cpi_core'] / df_cpi_core_old['cpi_core'].shift(4)) - 1) * 100
del df_cpi_core_old['cpi_core']

# Output gap
df_pluck = pd.read_parquet('pluckingpo_estimates_pf.parquet')
df_pluck['quarter'] = pd.to_datetime(df_pluck['quarter']).dt.to_period('Q')
df_pluck = df_pluck.set_index('quarter')
df_pluck = df_pluck[['output_gap', 'output_gap_lb']]

# ALTERNATE core exclusion series
df_core_cpi_alt = pd.read_csv('CoreInflationAlt.txt')
df_core_cpi_alt['quarter'] = pd.to_datetime(df_core_cpi_alt['month']).dt.to_period('q')
del df_core_cpi_alt['month']
df_core_cpi_alt = df_core_cpi_alt.groupby('quarter').agg('mean')
for i in ['cpi_core_dosm_yoy', 'cpi_core_alt']:
    del df_core_cpi_alt[i]
df_ceic = pd.concat([df_ceic, df_core_cpi_alt], axis=1)

# Merging
df = pd.concat([df_pluck, df_ceic, df_cpi_core_old], axis=1)
df = df.sort_index()

# Splicing / filling backseries
df.loc[df['ppi_yoy'].isna(), 'ppi_yoy'] = df['ppi_old_yoy']
df.loc[df['cpi_core_yoy'].isna(), 'cpi_core_yoy'] = df['cpi_core_old_yoy']
del df['ppi_old_yoy']
del df['cpi_core_old_yoy']

# ALTERNATE core CPI
df.loc[(((df.index >= '2015Q1') & (df.index >= '2016Q3')) |
        ((df.index >= '2018Q1') & (df.index >= '2019Q3'))), 'cpi_core_yoy'] = df['cpi_core_alt_yoy'].copy()
del df['cpi_core_alt_yoy']

# Timebound
df = df[(df.index >= T_lb) & (df.index <= T_ub)]

# Generate Lags
# Define contemporaneous
list_y_cols=['cpi_core_yoy']
list_y_nice_names = ['Core CPI YoY (Ex. Tax Effect)']
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
df_exoutliers=df[~(df.index.astype('str').isin(list_T_outliers))]  # separate df for plotting

# III --- Plots against plucking output gap


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
        fig.write_image('Output/PluckingPO_MacroComparison_' + output_suffix + '.png', height=768, width=1366)
        fig.write_html('Output/PluckingPO_MacroComparison_' + output_suffix + '.html')


# Plot contemporaneous
list_main_titles=['"Plucking" Output Gap versus ' + i for i in list_y_nice_names]
list_suffixes = [i + '_alt' for i in list_y_cols.copy()]  # ALTERNATE
scatterplots(
    data_full=df,
    data_ex_outliers=df_exoutliers,
    x_cols=['output_gap'],
    y_cols=list_y_cols,
    x_nice_names = ['"Plucking" Output Gap %'],
    y_nice_names = list_y_nice_names,
    colours_core=['crimson'],
    colours_outliers=['grey'],
    marker_sizes=[16],
    line_widths=[3],
    main_titles=list_main_titles,
    output_suffixes=list_suffixes,
    show_ols=True
)
for i, j in tqdm(zip(list_suffixes, list_main_titles)):
    telsendimg(conf=tel_config,
               path='Output/PluckingPO_MacroComparison_' + i + '.png',
               cap=j)

# Plot lag1
time.sleep(15)
list_main_titles=['"Plucking" Output Gap versus ' + i for i in list_y_nice_names_lag1]
list_suffixes = [i + '_alt' for i in list_y_cols_lag1.copy()]  # ALTERNATE
scatterplots(
    data_full=df,
    data_ex_outliers=df_exoutliers,
    x_cols=['output_gap'],
    y_cols=list_y_cols,
    x_nice_names = ['"Plucking" Output Gap %'],
    y_nice_names = list_y_nice_names,
    colours_core=['crimson'],
    colours_outliers=['grey'],
    marker_sizes=[16],
    line_widths=[3],
    main_titles=list_main_titles,
    output_suffixes=list_suffixes,
    show_ols=True
)
for i, j in tqdm(zip(list_suffixes, list_main_titles)):
    telsendimg(conf=tel_config,
               path='Output/PluckingPO_MacroComparison_' + i + '.png',
               cap=j)

# Plot lag2
time.sleep(15)
list_main_titles=['"Plucking" Output Gap versus ' + i for i in list_y_nice_names_lag2]
list_suffixes = [i + '_alt' for i in list_y_cols_lag2.copy()]  # ALTERNATE
scatterplots(
    data_full=df,
    data_ex_outliers=df_exoutliers,
    x_cols=['output_gap'],
    y_cols=list_y_cols,
    x_nice_names = ['"Plucking" Output Gap %'],
    y_nice_names = list_y_nice_names,
    colours_core=['crimson'],
    colours_outliers=['grey'],
    marker_sizes=[16],
    line_widths=[3],
    main_titles=list_main_titles,
    output_suffixes=list_suffixes,
    show_ols=True
)
for i, j in tqdm(zip(list_suffixes, list_main_titles)):
    telsendimg(conf=tel_config,
               path='Output/PluckingPO_MacroComparison_' + i + '.png',
               cap=j)

# III --- Plots of other variables against each other
list_y_cols_others = ['cpi_core_yoy',
                      'cpi_core_yoy_lag1',
                      'cpi_core_yoy_lag2',
                      'cpi_core_yoy',
                      'cpi_core_yoy_lag1',
                      'cpi_core_yoy_lag2',]
list_y_nice_names_others = ['Core CPI YoY (Ex. Tax Effect)',
                            'Core CPI YoY (1Q Lag) (Ex. Tax Effect)',
                            'Core CPI YoY (2Q Lag) (Ex. Tax Effect)',
                            'Core CPI YoY (Ex. Tax Effect)',
                            'Core CPI YoY (1Q Lag) (Ex. Tax Effect)',
                            'Core CPI YoY (2Q Lag) (Ex. Tax Effect)']
list_x_cols_others = ['ur_yoy',
                      'ur_yoy',
                      'ur_yoy',
                      'rgdp_yoy',
                      'rgdp_yoy',
                      'rgdp_yoy']
list_x_nice_names_others = ['Unemployment Rate YoY',
                            'Unemployment Rate YoY',
                            'Unemployment Rate YoY',
                            'Real GDP YoY',
                            'Real GDP YoY',
                            'Real GDP YoY',]
# Plot
time.sleep(15)
list_main_titles=[y + ' Against ' + x for y, x in zip(list_y_nice_names_others, list_x_nice_names_others)]
list_suffixes = ['PCurveUR_Core_alt',
                 'PCurveUR_Core_Lag1_alt',
                 'PCurveUR_Core_Lag2_alt',
                 'PCurveGDP_Core_alt',
                 'PCurveGDP_Core_Lag1_alt',
                 'PCurveGDP_Core_Lag2_alt']
scatterplots(
    data_full=df,
    data_ex_outliers=df_exoutliers,
    x_cols=list_x_cols_others,
    y_cols=list_y_cols_others,
    x_nice_names = list_x_nice_names_others,
    y_nice_names = list_y_nice_names_others,
    colours_core=['black'] * 6,
    colours_outliers=['grey'] * 6,
    marker_sizes=[16] * 6,
    line_widths=[3] * 6,
    main_titles=list_main_titles,
    output_suffixes=list_suffixes,
    show_ols=True
)
for i, j in tqdm(zip(list_suffixes, list_main_titles)):
    telsendimg(conf=tel_config,
               path='Output/PluckingPO_MacroComparison_' + i + '.png',
               cap=j)

# IV --- Notify
telsendmsg(conf=tel_config,
           msg='pluckingpo_plot_macro_comparison: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')