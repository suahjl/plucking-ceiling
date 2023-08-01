# -------------- Estimate response of PO to supply and demand shocks

import pandas as pd
import numpy as np
from datetime import date, timedelta
import statsmodels.formula.api as smf
import statsmodels.tsa.api as sm
import plotly.graph_objects as go
import plotly.express as px
import telegram_send
import dataframe_image as dfi
from PIL import Image
from tqdm import tqdm
import time
import re
from ceic_api_client.pyceic import Ceic
import localprojections as lp
from dotenv import load_dotenv
import os
import ast

# import pyeviews as evp

time_start = time.time()

# 0 --- Main settings
load_dotenv()
tel_config = os.getenv('TEL_CONFIG')
T_lb = '1995Q1'  # 1995Q1
T_ub = '2022Q4'  # '2022Q3' '2019Q4'
T_lb_day = date(1995, 1, 1)
T_ub_day = date(2022, 12, 31)  # date(2022, 9, 30)
Ceic.login(os.getenv('CEIC_USERNAME'), os.getenv('CEIC_PASSWORD'))


# I --- Functions


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


# II --- Data
# Plucking PO estimates
df = pd.read_parquet('pluckingpo_dns_estimates_pf.parquet')
df['quarter'] = pd.to_datetime(df['quarter']).dt.to_period('Q')
df = df[(df['quarter'] >= T_lb) & (df['quarter'] <= T_ub)]
df = df.set_index('quarter')
df = df.rename(columns={'output_gap': 'output_gap_pluck',
                        'output_gap_lb': 'output_gap_pluck_lb',
                        'gdp_ceiling': 'po_pluck',
                        'gdp_ceiling_lb': 'po_pluck_lb'})

# Boom-bust PO estimates
df_bb = pd.read_parquet('boombustpo_estimates_kf_onesided.parquet')  # onesided
# df_bb = pd.read_parquet('boombustpo_estimates_kf.parquet')
df_bb['quarter'] = pd.to_datetime(df_bb['quarter']).dt.to_period('Q')
df_bb = df_bb[(df_bb['quarter'] >= T_lb) & (df_bb['quarter'] <= T_ub)]
df_bb = df_bb.set_index('quarter')
df_bb = df_bb.sort_index()
df_bb = pd.DataFrame(df_bb[['po_avg', 'output_gap_avg']]).rename(columns={'output_gap_avg': 'output_gap_boombust',
                                                                          'po_avg': 'po_boombust'})
# df_bb = pd.DataFrame(df_bb[['po_pf', 'output_gap_pf']]).rename(columns={'output_gap_pf': 'output_gap_boombust',
#                                                                         'po_pf': 'po_boombust'})
# df_bb = pd.DataFrame(df_bb[['po_kf', 'output_gap_kf']]).rename(columns={'output_gap_kf': 'output_gap_boombust',
#                                                                         'po_kf': 'po_boombust'})

# CEIC data
series_ids = pd.read_csv('ceic_seriesid_forsupplydemand.txt', dtype='str', sep='|')
series_ids = list(series_ids['series_id'])
df_ceic = ceic2pandas_ts(input=series_ids, start_date=T_lb_day)
df_ceic = df_ceic.rename(columns={'Crude Oil: Spot Price: Europe Brent': 'brent',
                                  'Government Securities Yield: 10 Years': 'mgs10y',
                                  'Government Securities Yield: 1 Year': 'mgs1y',
                                  'Government Securities Yield: 2 Years': 'mgs2y',
                                  'Government Securities Yield: 3 years': 'mgs3y',
                                  'Government Securities Yield: 4 years': 'mgs4y',
                                  'Government Securities Yield: 5 years': 'mgs5y',
                                  'Consumer Price Index (CPI)': 'cpi',
                                  'Consumer Price Index (CPI): Core': 'cpi_core',
                                  'Economic Policy Uncertainty Index: Global: PPP-adjusted GDP': 'gepu',
                                  'Base Lending Rate: Period Average: Commercial Banks': 'blr',
                                  'Interbank Offered Rate: Fixing: 1 Month': 'klibor1m'})
df_ceic['quarter'] = pd.to_datetime(df_ceic.index).to_period('Q')
df_ceic = df_ceic[(df_ceic['quarter'] >= T_lb) & (df_ceic['quarter'] <= T_ub)]
df_ceic = df_ceic.groupby('quarter').agg('mean')

# Static historical core cpi
core_cpi_old = pd.read_csv('old_static_cpi_core.txt', sep=',')
core_cpi_old['quarter'] = pd.to_datetime(core_cpi_old['quarter']).dt.to_period('Q')
core_cpi_old = core_cpi_old.set_index('quarter')
core_cpi_old = core_cpi_old.rename(columns={'cpi_core': 'cpi_core_old'})

# Consol data frame
df = pd.concat([df_bb, df, df_ceic, core_cpi_old], axis=1)
df.loc[df.index < '2015Q1', 'cpi_core'] = df['cpi_core_old']  # merge old and new cpi_core
del df['cpi_core_old']
df = df.sort_index()
df = df[(df.index >= T_lb) & (df.index <= T_ub)]

# Trim columns
col_keep = ['po_boombust', 'po_pluck', 'po_pluck_lb',
            'output_gap_boombust', 'output_gap_pluck', 'output_gap_pluck_lb',
            'gdp',
            'brent',
            'gepu',
            'cpi', 'cpi_core',
            'blr',
            'mgs10y', 'mgs1y', 'mgs2y', 'mgs3y', 'mgs4y', 'mgs5y',
            'klibor1m']
df = df[col_keep]

# Seasonally adjust core cpi and brent
list_col = ['cpi_core', 'brent']
for i in tqdm(list_col):
    sadj_res = sm.x13_arima_analysis(df.loc[df[i].notna(), i])  # handles NAs
    sadj_seasadj = sadj_res.seasadj
    df[i] = sadj_seasadj

# first diff and log-diff transformation
col_levels = ['po_boombust', 'po_pluck', 'po_pluck_lb',
              'gdp', 'brent', 'cpi', 'cpi_core']
col_rates = ['output_gap_boombust', 'output_gap_pluck', 'output_gap_pluck_lb',
             'mgs10y', 'mgs1y', 'mgs2y', 'mgs3y', 'mgs4y', 'mgs5y', 'blr', 'klibor1m']
for i in col_levels:
    df[i] = np.log(df[i]) - np.log(df[i]).shift(1)
    # if i == 'po_boombust':
    #     df[i] = df[i] - df[i].shift(1)  # I(2)
for i in col_rates:
    df[i] = df[i] - df[i].shift(1)

# II --- VAR

max_lags_choice = 4
trend_term_choice = 'c'

with_gepu = True
if with_gepu:
    order_base1a = ['gepu', 'brent', 'gdp', 'cpi_core', 'blr']
    order_pluck1a = ['gepu', 'brent', 'po_pluck', 'cpi_core', 'blr']
    order_boombust1a = ['gepu', 'brent', 'po_boombust', 'cpi_core', 'blr']

    order_base1b = ['gepu', 'brent', 'gdp', 'cpi_core', 'mgs10y']
    order_pluck1b = ['gepu', 'brent', 'po_pluck', 'cpi_core', 'mgs10y']
    order_boombust1b = ['gepu', 'brent', 'po_boombust', 'cpi_core', 'mgs10y']

    order_base1c = ['gepu', 'brent', 'gdp', 'cpi_core', 'klibor1m']
    order_pluck1c = ['gepu', 'brent', 'po_pluck', 'cpi_core', 'klibor1m']
    order_boombust1c = ['gepu', 'brent', 'po_boombust', 'cpi_core', 'klibor1m']

    order_base1d = ['gepu', 'brent', 'gdp', 'cpi_core', 'mgs1y']
    order_pluck1d = ['gepu', 'brent', 'po_pluck', 'cpi_core', 'mgs1y']
    order_boombust1d = ['gepu', 'brent', 'po_boombust', 'cpi_core', 'mgs1y']

    order_base2a = ['gepu', 'brent', 'gdp', 'blr']
    order_pluck2a = ['gepu', 'brent', 'po_pluck', 'blr']
    order_boombust2a = ['gepu', 'brent', 'po_boombust', 'blr']

    order_base2b = ['gepu', 'brent', 'gdp', 'mgs10y']
    order_pluck2b = ['gepu', 'brent', 'po_pluck', 'mgs10y']
    order_boombust2b = ['gepu', 'brent', 'po_boombust', 'mgs10y']

    order_base2c = ['gepu', 'brent', 'gdp', 'klibor1m']
    order_pluck2c = ['gepu', 'brent', 'po_pluck', 'klibor1m']
    order_boombust2c = ['gepu', 'brent', 'po_boombust', 'klibor1m']

    order_base2d = ['gepu', 'brent', 'gdp', 'mgs1y']
    order_pluck2d = ['gepu', 'brent', 'po_pluck', 'mgs1y']
    order_boombust2d = ['gepu', 'brent', 'po_boombust', 'mgs1y']

elif not with_gepu:
    order_base1a = ['brent', 'gdp', 'cpi_core', 'blr']
    order_pluck1a = ['brent', 'po_pluck', 'cpi_core', 'blr']
    order_boombust1a = ['brent', 'po_boombust', 'cpi_core', 'blr']

    order_base1b = ['brent', 'gdp', 'cpi_core', 'mgs10y']
    order_pluck1b = ['brent', 'po_pluck', 'cpi_core', 'mgs10y']
    order_boombust1b = ['brent', 'po_boombust', 'cpi_core', 'mgs10y']

    order_base1c = ['brent', 'gdp', 'cpi_core', 'klibor1m']
    order_pluck1c = ['brent', 'po_pluck', 'cpi_core', 'klibor1m']
    order_boombust1c = ['brent', 'po_boombust', 'cpi_core', 'klibor1m']

    order_base1d = ['brent', 'gdp', 'cpi_core', 'mgs1y']
    order_pluck1d = ['brent', 'po_pluck', 'cpi_core', 'mgs1y']
    order_boombust1d = ['brent', 'po_boombust', 'cpi_core', 'mgs1y']

    order_base2a = ['brent', 'gdp', 'blr']
    order_pluck2a = ['brent', 'po_pluck', 'blr']
    order_boombust2a = ['brent', 'po_boombust', 'blr']

    order_base2b = ['brent', 'gdp', 'mgs10y']
    order_pluck2b = ['brent', 'po_pluck', 'mgs10y']
    order_boombust2b = ['brent', 'po_boombust', 'mgs10y']

    order_base2c = ['brent', 'gdp', 'klibor1m']
    order_pluck2c = ['brent', 'po_pluck', 'klibor1m']
    order_boombust2c = ['brent', 'po_boombust', 'klibor1m']

    order_base2d = ['brent', 'gdp', 'mgs1y']
    order_pluck2d = ['brent', 'po_pluck', 'mgs1y']
    order_boombust2d = ['brent', 'po_boombust', 'mgs1y']


def est_var(data, chol_order, max_lags, trend_term, irf_horizon, plot_response, plot_cirf, charts_dst_prefix):
    # prelims
    d = data.copy()

    # trim and reorder data frame
    d = d[chol_order]
    d = d.dropna()  # cannot handle missing values

    # model est
    mod = sm.VAR(d)
    res = mod.fit(maxlags=max_lags, ic='hqic', trend=trend_term)

    # back out IRF
    irf = res.irf(irf_horizon)
    fig_irf = irf.plot(orth=True, response=plot_response, stderr_type='mc')  # 'asym' 'mc'
    fig_irf.savefig(fname=charts_dst_prefix + '_OIRF.png')

    # cumulative IRF
    if plot_cirf:
        fig_cirf = irf.plot_cum_effects(orth=True, response=plot_response)
        fig_irf.savefig(fname=charts_dst_prefix + '_CIRF.png')
    elif not plot_cirf:
        fig_cirf = []

    # output
    return mod, res, irf, fig_irf, fig_cirf


# 1a: All 5 variables, Core CPI, MP = BLR
mod_var_base1a, res_var_base1a, irf_var_base1a, fig_irf_var_base1a, fig_cirf_var_base1a = est_var(
    data=df,
    chol_order=order_base1a,
    max_lags=max_lags_choice,
    trend_term=trend_term_choice,
    irf_horizon=12,
    plot_response='gdp',
    plot_cirf=False,
    charts_dst_prefix='Output/ComparingPO_SupplyDemand_VAR_Base1a'
)
mod_var_pluck1a, res_var_pluck1a, irf_var_pluck1a, fig_irf_var_pluck1a, fig_cirf_var_pluck1a = est_var(
    data=df,
    chol_order=order_pluck1a,
    max_lags=max_lags_choice,
    trend_term=trend_term_choice,
    irf_horizon=12,
    plot_response='po_pluck',
    plot_cirf=False,
    charts_dst_prefix='Output/ComparingPO_SupplyDemand_VAR_Pluck1a'
)
mod_var_boombust1a, res_var_boombust1a, irf_var_boombust1a, fig_irf_var_boombust1a, fig_cirf_var_boombust1a = est_var(
    data=df,
    chol_order=order_boombust1a,
    max_lags=max_lags_choice,
    trend_term=trend_term_choice,
    irf_horizon=12,
    plot_response='po_boombust',
    plot_cirf=False,
    charts_dst_prefix='Output/ComparingPO_SupplyDemand_VAR_BoombustOneSidedAvg1a'
)

# 1b: All 5 variables, Core CPI, MP = MGS10Y
mod_var_base1b, res_var_base1b, irf_var_base1b, fig_irf_var_base1b, fig_cirf_var_base1b = est_var(
    data=df,
    chol_order=order_base1b,
    max_lags=max_lags_choice,
    trend_term=trend_term_choice,
    irf_horizon=12,
    plot_response='gdp',
    plot_cirf=False,
    charts_dst_prefix='Output/ComparingPO_SupplyDemand_VAR_Base1b'
)
mod_var_pluck1b, res_var_pluck1b, irf_var_pluck1b, fig_irf_var_pluck1b, fig_cirf_var_pluck1b = est_var(
    data=df,
    chol_order=order_pluck1b,
    max_lags=max_lags_choice,
    trend_term=trend_term_choice,
    irf_horizon=12,
    plot_response='po_pluck',
    plot_cirf=False,
    charts_dst_prefix='Output/ComparingPO_SupplyDemand_VAR_Pluck1b'
)
mod_var_boombust1b, res_var_boombust1b, irf_var_boombust1b, fig_irf_var_boombust1b, fig_cirf_var_boombust1b = est_var(
    data=df,
    chol_order=order_boombust1b,
    max_lags=max_lags_choice,
    trend_term=trend_term_choice,
    irf_horizon=12,
    plot_response='po_boombust',
    plot_cirf=False,
    charts_dst_prefix='Output/ComparingPO_SupplyDemand_VAR_BoombustOneSidedAvg1b'
)

# 1c: All 5 variables, Core CPI, MP = KLIBOR1M
mod_var_base1c, res_var_base1c, irf_var_base1c, fig_irf_var_base1c, fig_cirf_var_base1c = est_var(
    data=df,
    chol_order=order_base1c,
    max_lags=max_lags_choice,
    trend_term=trend_term_choice,
    irf_horizon=12,
    plot_response='gdp',
    plot_cirf=False,
    charts_dst_prefix='Output/ComparingPO_SupplyDemand_VAR_Base1c'
)
mod_var_pluck1c, res_var_pluck1c, irf_var_pluck1c, fig_irf_var_pluck1c, fig_cirf_var_pluck1c = est_var(
    data=df,
    chol_order=order_pluck1c,
    max_lags=max_lags_choice,
    trend_term=trend_term_choice,
    irf_horizon=12,
    plot_response='po_pluck',
    plot_cirf=False,
    charts_dst_prefix='Output/ComparingPO_SupplyDemand_VAR_Pluck1c'
)
mod_var_boombust1c, res_var_boombust1c, irf_var_boombust1c, fig_irf_var_boombust1c, fig_cirf_var_boombust1c = est_var(
    data=df,
    chol_order=order_boombust1c,
    max_lags=max_lags_choice,
    trend_term=trend_term_choice,
    irf_horizon=12,
    plot_response='po_boombust',
    plot_cirf=False,
    charts_dst_prefix='Output/ComparingPO_SupplyDemand_VAR_BoombustOneSidedAvg1c'
)

# 1d: All 5 variables, Core CPI, MP = MGS1Y
mod_var_base1d, res_var_base1d, irf_var_base1d, fig_irf_var_base1d, fig_cirf_var_base1d = est_var(
    data=df,
    chol_order=order_base1d,
    max_lags=max_lags_choice,
    trend_term=trend_term_choice,
    irf_horizon=12,
    plot_response='gdp',
    plot_cirf=False,
    charts_dst_prefix='Output/ComparingPO_SupplyDemand_VAR_Base1d'
)
mod_var_pluck1d, res_var_pluck1d, irf_var_pluck1d, fig_irf_var_pluck1d, fig_cirf_var_pluck1d = est_var(
    data=df,
    chol_order=order_pluck1d,
    max_lags=max_lags_choice,
    trend_term=trend_term_choice,
    irf_horizon=12,
    plot_response='po_pluck',
    plot_cirf=False,
    charts_dst_prefix='Output/ComparingPO_SupplyDemand_VAR_Pluck1d'
)
mod_var_boombust1d, res_var_boombust1d, irf_var_boombust1d, fig_irf_var_boombust1d, fig_cirf_var_boombust1d = est_var(
    data=df,
    chol_order=order_boombust1d,
    max_lags=max_lags_choice,
    trend_term=trend_term_choice,
    irf_horizon=12,
    plot_response='po_boombust',
    plot_cirf=False,
    charts_dst_prefix='Output/ComparingPO_SupplyDemand_VAR_BoombustOneSidedAvg1d'
)

# 2a: No inflation, MP = BLR
mod_var_base2a, res_var_base2a, irf_var_base2a, fig_irf_var_base2a, fig_cirf_var_base2a = est_var(
    data=df,
    chol_order=order_base2a,
    max_lags=max_lags_choice,
    trend_term=trend_term_choice,
    irf_horizon=12,
    plot_response='gdp',
    plot_cirf=False,
    charts_dst_prefix='Output/ComparingPO_SupplyDemand_VAR_Base2a'
)
mod_var_pluck2a, res_var_pluck2a, irf_var_pluck2a, fig_irf_var_pluck2a, fig_cirf_var_pluck2a = est_var(
    data=df,
    chol_order=order_pluck2a,
    max_lags=max_lags_choice,
    trend_term=trend_term_choice,
    irf_horizon=12,
    plot_response='po_pluck',
    plot_cirf=False,
    charts_dst_prefix='Output/ComparingPO_SupplyDemand_VAR_Pluck2a'
)
mod_var_boombust2a, res_var_boombust2a, irf_var_boombust2a, fig_irf_var_boombust2a, fig_cirf_var_boombust2a = est_var(
    data=df,
    chol_order=order_boombust2a,
    max_lags=max_lags_choice,
    trend_term=trend_term_choice,
    irf_horizon=12,
    plot_response='po_boombust',
    plot_cirf=False,
    charts_dst_prefix='Output/ComparingPO_SupplyDemand_VAR_BoombustOneSidedAvg2a'
)

# 2b: No inflation, MP = MGS10Y
mod_var_base2b, res_var_base2b, irf_var_base2b, fig_irf_var_base2b, fig_cirf_var_base2b = est_var(
    data=df,
    chol_order=order_base2b,
    max_lags=max_lags_choice,
    trend_term=trend_term_choice,
    irf_horizon=12,
    plot_response='gdp',
    plot_cirf=False,
    charts_dst_prefix='Output/ComparingPO_SupplyDemand_VAR_Base2b'
)
mod_var_pluck2b, res_var_pluck2b, irf_var_pluck2b, fig_irf_var_pluck2b, fig_cirf_var_pluck2b = est_var(
    data=df,
    chol_order=order_pluck2b,
    max_lags=max_lags_choice,
    trend_term=trend_term_choice,
    irf_horizon=12,
    plot_response='po_pluck',
    plot_cirf=False,
    charts_dst_prefix='Output/ComparingPO_SupplyDemand_VAR_Pluck2b'
)

mod_var_boombust2b, res_var_boombust2b, irf_var_boombust2b, fig_irf_var_boombust2b, fig_cirf_var_boombust2b = est_var(
    data=df,
    chol_order=order_boombust2b,
    max_lags=max_lags_choice,
    trend_term=trend_term_choice,
    irf_horizon=12,
    plot_response='po_boombust',
    plot_cirf=False,
    charts_dst_prefix='Output/ComparingPO_SupplyDemand_VAR_BoombustOneSidedAvg2b'
)

# 2c: No inflation, MP = KLIBOR1M
mod_var_base2c, res_var_base2c, irf_var_base2c, fig_irf_var_base2c, fig_cirf_var_base2c = est_var(
    data=df,
    chol_order=order_base2c,
    max_lags=max_lags_choice,
    trend_term=trend_term_choice,
    irf_horizon=12,
    plot_response='gdp',
    plot_cirf=False,
    charts_dst_prefix='Output/ComparingPO_SupplyDemand_VAR_Base2c'
)
mod_var_pluck2c, res_var_pluck2c, irf_var_pluck2c, fig_irf_var_pluck2c, fig_cirf_var_pluck2c = est_var(
    data=df,
    chol_order=order_pluck2c,
    max_lags=max_lags_choice,
    trend_term=trend_term_choice,
    irf_horizon=12,
    plot_response='po_pluck',
    plot_cirf=False,
    charts_dst_prefix='Output/ComparingPO_SupplyDemand_VAR_Pluck2c'
)

mod_var_boombust2c, res_var_boombust2c, irf_var_boombust2c, fig_irf_var_boombust2c, fig_cirf_var_boombust2c = est_var(
    data=df,
    chol_order=order_boombust2c,
    max_lags=max_lags_choice,
    trend_term=trend_term_choice,
    irf_horizon=12,
    plot_response='po_boombust',
    plot_cirf=False,
    charts_dst_prefix='Output/ComparingPO_SupplyDemand_VAR_BoombustOneSidedAvg2c'
)

# 2d: No inflation, MP = MGS1Y
mod_var_base2d, res_var_base2d, irf_var_base2d, fig_irf_var_base2d, fig_cirf_var_base2d = est_var(
    data=df,
    chol_order=order_base2d,
    max_lags=max_lags_choice,
    trend_term=trend_term_choice,
    irf_horizon=12,
    plot_response='gdp',
    plot_cirf=False,
    charts_dst_prefix='Output/ComparingPO_SupplyDemand_VAR_Base2d'
)
mod_var_pluck2d, res_var_pluck2d, irf_var_pluck2d, fig_irf_var_pluck2d, fig_cirf_var_pluck2d = est_var(
    data=df,
    chol_order=order_pluck2d,
    max_lags=max_lags_choice,
    trend_term=trend_term_choice,
    irf_horizon=12,
    plot_response='po_pluck',
    plot_cirf=False,
    charts_dst_prefix='Output/ComparingPO_SupplyDemand_VAR_Pluck2d'
)

mod_var_boombust2d, res_var_boombust2d, irf_var_boombust2d, fig_irf_var_boombust2d, fig_cirf_var_boombust2d = est_var(
    data=df,
    chol_order=order_boombust2d,
    max_lags=max_lags_choice,
    trend_term=trend_term_choice,
    irf_horizon=12,
    plot_response='po_boombust',
    plot_cirf=False,
    charts_dst_prefix='Output/ComparingPO_SupplyDemand_VAR_BoombustOneSidedAvg2d'
)

# III --- Local Projections
opt_lags = 2  # include 2 lags in the local projections model
opt_cov = 'robust'  # HAC standard errors
opt_ci = 0.95  # 95% confidence intervals

# 1a: All 5 variables, Core CPI, MP = BLR
irf_lp_base1a = lp.TimeSeriesLP(
    data=df,  # input dataframe
    Y=order_base1a,  # variables in the model
    response=order_base1a.copy(),  # variables whose IRFs should be estimated
    horizon=12,  # estimation horizon of IRFs
    lags=1,  # lags in the model
    newey_lags=2,  # maximum lags when estimating Newey-West standard errors
    ci_width=opt_ci  # width of confidence band
)
irfplot_lp_base1a = lp.IRFPlot(
    irf=irf_lp_base1a,  # take output from the estimated model
    response=['gdp'],  # plot only response of xx ...
    shock=order_base1a,  # ... to shocks from all variables
    n_columns=3,  # max columns in the figure
    n_rows=2,  # max rows in the figure
    maintitle='Local Projections: Impulse Responses',  # self-defined title of the IRF plot
    show_fig=False,
    save_pic=True,
    out_path='Output/',
    out_name='ComparingPO_SupplyDemand_LP_Base1a_IRF'
)

irf_lp_pluck1a = lp.TimeSeriesLP(
    data=df,  # input dataframe
    Y=order_pluck1a,  # variables in the model
    response=order_pluck1a.copy(),  # variables whose IRFs should be estimated
    horizon=12,  # estimation horizon of IRFs
    lags=1,  # lags in the model
    newey_lags=2,  # maximum lags when estimating Newey-West standard errors
    ci_width=opt_ci  # width of confidence band
)
irfplot_lp_pluck1a = lp.IRFPlot(
    irf=irf_lp_pluck1a,  # take output from the estimated model
    response=['po_pluck'],  # plot only response of xx ...
    shock=order_pluck1a,  # ... to shocks from all variables
    n_columns=3,  # max columns in the figure
    n_rows=2,  # max rows in the figure
    maintitle='Local Projections: Impulse Responses',  # self-defined title of the IRF plot
    show_fig=False,
    save_pic=True,
    out_path='Output/',
    out_name='ComparingPO_SupplyDemand_LP_Pluck1a_IRF'
)

irf_lp_boombust1a = lp.TimeSeriesLP(
    data=df,  # input dataframe
    Y=order_boombust1a,  # variables in the model
    response=order_boombust1a.copy(),  # variables whose IRFs should be estimated
    horizon=12,  # estimation horizon of IRFs
    lags=1,  # lags in the model
    newey_lags=2,  # maximum lags when estimating Newey-West standard errors
    ci_width=opt_ci  # width of confidence band
)
irfplot_lp_boombust1a = lp.IRFPlot(
    irf=irf_lp_boombust1a,  # take output from the estimated model
    response=['po_boombust'],  # plot only response of xx ...
    shock=order_boombust1a,  # ... to shocks from all variables
    n_columns=3,  # max columns in the figure
    n_rows=2,  # max rows in the figure
    maintitle='Local Projections: Impulse Responses',  # self-defined title of the IRF plot
    show_fig=False,
    save_pic=True,
    out_path='Output/',
    out_name='ComparingPO_SupplyDemand_LP_BoombustOneSidedAvg1a_IRF'
)

# 1b: All 5 variables, Core CPI, MP = MGS10Y
irf_lp_base1b = lp.TimeSeriesLP(
    data=df,  # input dataframe
    Y=order_base1b,  # variables in the model
    response=order_base1b.copy(),  # variables whose IRFs should be estimated
    horizon=12,  # estimation horizon of IRFs
    lags=1,  # lags in the model
    newey_lags=2,  # maximum lags when estimating Newey-West standard errors
    ci_width=opt_ci  # width of confidence band
)
irfplot_lp_base1b = lp.IRFPlot(
    irf=irf_lp_base1b,  # take output from the estimated model
    response=['gdp'],  # plot only response of xx ...
    shock=order_base1b,  # ... to shocks from all variables
    n_columns=3,  # max columns in the figure
    n_rows=2,  # max rows in the figure
    maintitle='Local Projections: Impulse Responses',  # self-defined title of the IRF plot
    show_fig=False,
    save_pic=True,
    out_path='Output/',
    out_name='ComparingPO_SupplyDemand_LP_Base1b_IRF'
)

irf_lp_pluck1b = lp.TimeSeriesLP(
    data=df,  # input dataframe
    Y=order_pluck1b,  # variables in the model
    response=order_pluck1b.copy(),  # variables whose IRFs should be estimated
    horizon=12,  # estimation horizon of IRFs
    lags=1,  # lags in the model
    newey_lags=2,  # maximum lags when estimating Newey-West standard errors
    ci_width=opt_ci  # width of confidence band
)
irfplot_lp_pluck1b = lp.IRFPlot(
    irf=irf_lp_pluck1b,  # take output from the estimated model
    response=['po_pluck'],  # plot only response of xx ...
    shock=order_pluck1b,  # ... to shocks from all variables
    n_columns=3,  # max columns in the figure
    n_rows=2,  # max rows in the figure
    maintitle='Local Projections: Impulse Responses',  # self-defined title of the IRF plot
    show_fig=False,
    save_pic=True,
    out_path='Output/',
    out_name='ComparingPO_SupplyDemand_LP_Pluck1b_IRF'
)

irf_lp_boombust1b = lp.TimeSeriesLP(
    data=df,  # input dataframe
    Y=order_boombust1b,  # variables in the model
    response=order_boombust1b.copy(),  # variables whose IRFs should be estimated
    horizon=12,  # estimation horizon of IRFs
    lags=1,  # lags in the model
    newey_lags=2,  # maximum lags when estimating Newey-West standard errors
    ci_width=opt_ci  # width of confidence band
)
irfplot_lp_boombust1b = lp.IRFPlot(
    irf=irf_lp_boombust1b,  # take output from the estimated model
    response=['po_boombust'],  # plot only response of xx ...
    shock=order_boombust1b,  # ... to shocks from all variables
    n_columns=3,  # max columns in the figure
    n_rows=2,  # max rows in the figure
    maintitle='Local Projections: Impulse Responses',  # self-defined title of the IRF plot
    show_fig=False,
    save_pic=True,
    out_path='Output/',
    out_name='ComparingPO_SupplyDemand_LP_BoombustOneSidedAvg1b_IRF'
)

# 1c: All 5 variables, Core CPI, MP = KLIBOR1M
irf_lp_base1c = lp.TimeSeriesLP(
    data=df,  # input dataframe
    Y=order_base1c,  # variables in the model
    response=order_base1c.copy(),  # variables whose IRFs should be estimated
    horizon=12,  # estimation horizon of IRFs
    lags=1,  # lags in the model
    newey_lags=2,  # maximum lags when estimating Newey-West standard errors
    ci_width=opt_ci  # width of confidence band
)
irfplot_lp_base1c = lp.IRFPlot(
    irf=irf_lp_base1c,  # take output from the estimated model
    response=['gdp'],  # plot only response of xx ...
    shock=order_base1c,  # ... to shocks from all variables
    n_columns=3,  # max columns in the figure
    n_rows=2,  # max rows in the figure
    maintitle='Local Projections: Impulse Responses',  # self-defined title of the IRF plot
    show_fig=False,
    save_pic=True,
    out_path='Output/',
    out_name='ComparingPO_SupplyDemand_LP_Base1c_IRF'
)

irf_lp_pluck1c = lp.TimeSeriesLP(
    data=df,  # input dataframe
    Y=order_pluck1c,  # variables in the model
    response=order_pluck1c.copy(),  # variables whose IRFs should be estimated
    horizon=12,  # estimation horizon of IRFs
    lags=1,  # lags in the model
    newey_lags=2,  # maximum lags when estimating Newey-West standard errors
    ci_width=opt_ci  # width of confidence band
)
irfplot_lp_pluck1c = lp.IRFPlot(
    irf=irf_lp_pluck1c,  # take output from the estimated model
    response=['po_pluck'],  # plot only response of xx ...
    shock=order_pluck1c,  # ... to shocks from all variables
    n_columns=3,  # max columns in the figure
    n_rows=2,  # max rows in the figure
    maintitle='Local Projections: Impulse Responses',  # self-defined title of the IRF plot
    show_fig=False,
    save_pic=True,
    out_path='Output/',
    out_name='ComparingPO_SupplyDemand_LP_Pluck1c_IRF'
)

irf_lp_boombust1c = lp.TimeSeriesLP(
    data=df,  # input dataframe
    Y=order_boombust1c,  # variables in the model
    response=order_boombust1c.copy(),  # variables whose IRFs should be estimated
    horizon=12,  # estimation horizon of IRFs
    lags=1,  # lags in the model
    newey_lags=2,  # maximum lags when estimating Newey-West standard errors
    ci_width=opt_ci  # width of confidence band
)
irfplot_lp_boombust1c = lp.IRFPlot(
    irf=irf_lp_boombust1c,  # take output from the estimated model
    response=['po_boombust'],  # plot only response of xx ...
    shock=order_boombust1c,  # ... to shocks from all variables
    n_columns=3,  # max columns in the figure
    n_rows=2,  # max rows in the figure
    maintitle='Local Projections: Impulse Responses',  # self-defined title of the IRF plot
    show_fig=False,
    save_pic=True,
    out_path='Output/',
    out_name='ComparingPO_SupplyDemand_LP_BoombustOneSidedAvg1c_IRF'
)

# 1d: All 5 variables, Core CPI, MP = MGS1Y
irf_lp_base1d = lp.TimeSeriesLP(
    data=df,  # input dataframe
    Y=order_base1d,  # variables in the model
    response=order_base1d.copy(),  # variables whose IRFs should be estimated
    horizon=12,  # estimation horizon of IRFs
    lags=1,  # lags in the model
    newey_lags=2,  # maximum lags when estimating Newey-West standard errors
    ci_width=opt_ci  # width of confidence band
)
irfplot_lp_base1d = lp.IRFPlot(
    irf=irf_lp_base1d,  # take output from the estimated model
    response=['gdp'],  # plot only response of xx ...
    shock=order_base1d,  # ... to shocks from all variables
    n_columns=3,  # max columns in the figure
    n_rows=2,  # max rows in the figure
    maintitle='Local Projections: Impulse Responses',  # self-defined title of the IRF plot
    show_fig=False,
    save_pic=True,
    out_path='Output/',
    out_name='ComparingPO_SupplyDemand_LP_Base1d_IRF'
)

irf_lp_pluck1d = lp.TimeSeriesLP(
    data=df,  # input dataframe
    Y=order_pluck1d,  # variables in the model
    response=order_pluck1d.copy(),  # variables whose IRFs should be estimated
    horizon=12,  # estimation horizon of IRFs
    lags=1,  # lags in the model
    newey_lags=2,  # maximum lags when estimating Newey-West standard errors
    ci_width=opt_ci  # width of confidence band
)
irfplot_lp_pluck1d = lp.IRFPlot(
    irf=irf_lp_pluck1d,  # take output from the estimated model
    response=['po_pluck'],  # plot only response of xx ...
    shock=order_pluck1d,  # ... to shocks from all variables
    n_columns=3,  # max columns in the figure
    n_rows=2,  # max rows in the figure
    maintitle='Local Projections: Impulse Responses',  # self-defined title of the IRF plot
    show_fig=False,
    save_pic=True,
    out_path='Output/',
    out_name='ComparingPO_SupplyDemand_LP_Pluck1d_IRF'
)

irf_lp_boombust1d = lp.TimeSeriesLP(
    data=df,  # input dataframe
    Y=order_boombust1d,  # variables in the model
    response=order_boombust1d.copy(),  # variables whose IRFs should be estimated
    horizon=12,  # estimation horizon of IRFs
    lags=1,  # lags in the model
    newey_lags=2,  # maximum lags when estimating Newey-West standard errors
    ci_width=opt_ci  # width of confidence band
)
irfplot_lp_boombust1d = lp.IRFPlot(
    irf=irf_lp_boombust1d,  # take output from the estimated model
    response=['po_boombust'],  # plot only response of xx ...
    shock=order_boombust1d,  # ... to shocks from all variables
    n_columns=3,  # max columns in the figure
    n_rows=2,  # max rows in the figure
    maintitle='Local Projections: Impulse Responses',  # self-defined title of the IRF plot
    show_fig=False,
    save_pic=True,
    out_path='Output/',
    out_name='ComparingPO_SupplyDemand_LP_BoombustOneSidedAvg1d_IRF'
)

# 2a: No inflation, MP = BLR
irf_lp_base2a = lp.TimeSeriesLP(
    data=df,  # input dataframe
    Y=order_base2a,  # variables in the model
    response=order_base2a.copy(),  # variables whose IRFs should be estimated
    horizon=12,  # estimation horizon of IRFs
    lags=1,  # lags in the model
    newey_lags=2,  # maximum lags when estimating Newey-West standard errors
    ci_width=opt_ci  # width of confidence band
)
irfplot_lp_base2a = lp.IRFPlot(
    irf=irf_lp_base2a,  # take output from the estimated model
    response=['gdp'],  # plot only response of xx ...
    shock=order_base2a,  # ... to shocks from all variables
    n_columns=2,  # max columns in the figure
    n_rows=2,  # max rows in the figure
    maintitle='Local Projections: Impulse Responses',  # self-defined title of the IRF plot
    show_fig=False,
    save_pic=True,
    out_path='Output/',
    out_name='ComparingPO_SupplyDemand_LP_Base2a_IRF'
)

irf_lp_pluck2a = lp.TimeSeriesLP(
    data=df,  # input dataframe
    Y=order_pluck2a,  # variables in the model
    response=order_pluck2a.copy(),  # variables whose IRFs should be estimated
    horizon=12,  # estimation horizon of IRFs
    lags=1,  # lags in the model
    newey_lags=2,  # maximum lags when estimating Newey-West standard errors
    ci_width=opt_ci  # width of confidence band
)
irfplot_lp_pluck2a = lp.IRFPlot(
    irf=irf_lp_pluck2a,  # take output from the estimated model
    response=['po_pluck'],  # plot only response of xx ...
    shock=order_pluck2a,  # ... to shocks from all variables
    n_columns=2,  # max columns in the figure
    n_rows=2,  # max rows in the figure
    maintitle='Local Projections: Impulse Responses',  # self-defined title of the IRF plot
    show_fig=False,
    save_pic=True,
    out_path='Output/',
    out_name='ComparingPO_SupplyDemand_LP_Pluck2a_IRF'
)

irf_lp_boombust2a = lp.TimeSeriesLP(
    data=df,  # input dataframe
    Y=order_boombust2a,  # variables in the model
    response=order_boombust2a.copy(),  # variables whose IRFs should be estimated
    horizon=12,  # estimation horizon of IRFs
    lags=1,  # lags in the model
    newey_lags=2,  # maximum lags when estimating Newey-West standard errors
    ci_width=opt_ci  # width of confidence band
)
irfplot_lp_boombust2a = lp.IRFPlot(
    irf=irf_lp_boombust2a,  # take output from the estimated model
    response=['po_boombust'],  # plot only response of xx ...
    shock=order_boombust2a,  # ... to shocks from all variables
    n_columns=2,  # max columns in the figure
    n_rows=2,  # max rows in the figure
    maintitle='Local Projections: Impulse Responses',  # self-defined title of the IRF plot
    show_fig=False,
    save_pic=True,
    out_path='Output/',
    out_name='ComparingPO_SupplyDemand_LP_BoombustOneSidedAvg2a_IRF'
)

# 2b: No inflation, MP = MGS10Y
irf_lp_base2b = lp.TimeSeriesLP(
    data=df,  # input dataframe
    Y=order_base2b,  # variables in the model
    response=order_base2b.copy(),  # variables whose IRFs should be estimated
    horizon=12,  # estimation horizon of IRFs
    lags=1,  # lags in the model
    newey_lags=2,  # maximum lags when estimating Newey-West standard errors
    ci_width=opt_ci  # width of confidence band
)
irfplot_lp_base2b = lp.IRFPlot(
    irf=irf_lp_base2b,  # take output from the estimated model
    response=['gdp'],  # plot only response of xx ...
    shock=order_base2b,  # ... to shocks from all variables
    n_columns=2,  # max columns in the figure
    n_rows=2,  # max rows in the figure
    maintitle='Local Projections: Impulse Responses',  # self-defined title of the IRF plot
    show_fig=False,
    save_pic=True,
    out_path='Output/',
    out_name='ComparingPO_SupplyDemand_LP_Base2b_IRF'
)

irf_lp_pluck2b = lp.TimeSeriesLP(
    data=df,  # input dataframe
    Y=order_pluck2b,  # variables in the model
    response=order_pluck2b.copy(),  # variables whose IRFs should be estimated
    horizon=12,  # estimation horizon of IRFs
    lags=1,  # lags in the model
    newey_lags=2,  # maximum lags when estimating Newey-West standard errors
    ci_width=opt_ci  # width of confidence band
)
irfplot_lp_pluck2b = lp.IRFPlot(
    irf=irf_lp_pluck2b,  # take output from the estimated model
    response=['po_pluck'],  # plot only response of xx ...
    shock=order_pluck2b,  # ... to shocks from all variables
    n_columns=2,  # max columns in the figure
    n_rows=2,  # max rows in the figure
    maintitle='Local Projections: Impulse Responses',  # self-defined title of the IRF plot
    show_fig=False,
    save_pic=True,
    out_path='Output/',
    out_name='ComparingPO_SupplyDemand_LP_Pluck2b_IRF'
)

irf_lp_boombust2b = lp.TimeSeriesLP(
    data=df,  # input dataframe
    Y=order_boombust2b,  # variables in the model
    response=order_boombust2b.copy(),  # variables whose IRFs should be estimated
    horizon=12,  # estimation horizon of IRFs
    lags=1,  # lags in the model
    newey_lags=2,  # maximum lags when estimating Newey-West standard errors
    ci_width=opt_ci  # width of confidence band
)
irfplot_lp_boombust2b = lp.IRFPlot(
    irf=irf_lp_boombust2b,  # take output from the estimated model
    response=['po_boombust'],  # plot only response of xx ...
    shock=order_boombust2b,  # ... to shocks from all variables
    n_columns=2,  # max columns in the figure
    n_rows=2,  # max rows in the figure
    maintitle='Local Projections: Impulse Responses',  # self-defined title of the IRF plot
    show_fig=False,
    save_pic=True,
    out_path='Output/',
    out_name='ComparingPO_SupplyDemand_LP_BoombustOneSidedAvg2b_IRF'
)

# 2c: No inflation, MP = KLIBOR1M
irf_lp_base2c = lp.TimeSeriesLP(
    data=df,  # input dataframe
    Y=order_base2c,  # variables in the model
    response=order_base2c.copy(),  # variables whose IRFs should be estimated
    horizon=12,  # estimation horizon of IRFs
    lags=1,  # lags in the model
    newey_lags=2,  # maximum lags when estimating Newey-West standard errors
    ci_width=opt_ci  # width of confidence band
)
irfplot_lp_base2c = lp.IRFPlot(
    irf=irf_lp_base2c,  # take output from the estimated model
    response=['gdp'],  # plot only response of xx ...
    shock=order_base2c,  # ... to shocks from all variables
    n_columns=3,  # max columns in the figure
    n_rows=3,  # max rows in the figure
    maintitle='Local Projections: Impulse Responses',  # self-defined title of the IRF plot
    show_fig=False,
    save_pic=True,
    out_path='Output/',
    out_name='ComparingPO_SupplyDemand_LP_Base2c_IRF'
)

irf_lp_pluck2c = lp.TimeSeriesLP(
    data=df,  # input dataframe
    Y=order_pluck2c,  # variables in the model
    response=order_pluck2c.copy(),  # variables whose IRFs should be estimated
    horizon=12,  # estimation horizon of IRFs
    lags=1,  # lags in the model
    newey_lags=2,  # maximum lags when estimating Newey-West standard errors
    ci_width=opt_ci  # width of confidence band
)
irfplot_lp_pluck2c = lp.IRFPlot(
    irf=irf_lp_pluck2c,  # take output from the estimated model
    response=['po_pluck'],  # plot only response of xx ...
    shock=order_pluck2c,  # ... to shocks from all variables
    n_columns=2,  # max columns in the figure
    n_rows=2,  # max rows in the figure
    maintitle='Local Projections: Impulse Responses',  # self-defined title of the IRF plot
    show_fig=False,
    save_pic=True,
    out_path='Output/',
    out_name='ComparingPO_SupplyDemand_LP_Pluck2c_IRF'
)

irf_lp_boombust2c = lp.TimeSeriesLP(
    data=df,  # input dataframe
    Y=order_boombust2c,  # variables in the model
    response=order_boombust2c.copy(),  # variables whose IRFs should be estimated
    horizon=12,  # estimation horizon of IRFs
    lags=1,  # lags in the model
    newey_lags=2,  # maximum lags when estimating Newey-West standard errors
    ci_width=opt_ci  # width of confidence band
)
irfplot_lp_boombust2c = lp.IRFPlot(
    irf=irf_lp_boombust2c,  # take output from the estimated model
    response=['po_boombust'],  # plot only response of xx ...
    shock=order_boombust2c,  # ... to shocks from all variables
    n_columns=2,  # max columns in the figure
    n_rows=2,  # max rows in the figure
    maintitle='Local Projections: Impulse Responses',  # self-defined title of the IRF plot
    show_fig=False,
    save_pic=True,
    out_path='Output/',
    out_name='ComparingPO_SupplyDemand_LP_BoombustOneSidedAvg2c_IRF'
)

# 2d: No inflation, MP = KLIBOR1M
irf_lp_base2d = lp.TimeSeriesLP(
    data=df,  # input dataframe
    Y=order_base2d,  # variables in the model
    response=order_base2d.copy(),  # variables whose IRFs should be estimated
    horizon=12,  # estimation horizon of IRFs
    lags=1,  # lags in the model
    newey_lags=2,  # maximum lags when estimating Newey-West standard errors
    ci_width=opt_ci  # width of confidence band
)
irfplot_lp_base2d = lp.IRFPlot(
    irf=irf_lp_base2d,  # take output from the estimated model
    response=['gdp'],  # plot only response of xx ...
    shock=order_base2d,  # ... to shocks from all variables
    n_columns=3,  # max columns in the figure
    n_rows=3,  # max rows in the figure
    maintitle='Local Projections: Impulse Responses',  # self-defined title of the IRF plot
    show_fig=False,
    save_pic=True,
    out_path='Output/',
    out_name='ComparingPO_SupplyDemand_LP_Base2d_IRF'
)

irf_lp_pluck2d = lp.TimeSeriesLP(
    data=df,  # input dataframe
    Y=order_pluck2d,  # variables in the model
    response=order_pluck2d.copy(),  # variables whose IRFs should be estimated
    horizon=12,  # estimation horizon of IRFs
    lags=1,  # lags in the model
    newey_lags=2,  # maximum lags when estimating Newey-West standard errors
    ci_width=opt_ci  # width of confidence band
)
irfplot_lp_pluck2d = lp.IRFPlot(
    irf=irf_lp_pluck2d,  # take output from the estimated model
    response=['po_pluck'],  # plot only response of xx ...
    shock=order_pluck2d,  # ... to shocks from all variables
    n_columns=2,  # max columns in the figure
    n_rows=2,  # max rows in the figure
    maintitle='Local Projections: Impulse Responses',  # self-defined title of the IRF plot
    show_fig=False,
    save_pic=True,
    out_path='Output/',
    out_name='ComparingPO_SupplyDemand_LP_Pluck2d_IRF'
)

irf_lp_boombust2d = lp.TimeSeriesLP(
    data=df,  # input dataframe
    Y=order_boombust2d,  # variables in the model
    response=order_boombust2d.copy(),  # variables whose IRFs should be estimated
    horizon=12,  # estimation horizon of IRFs
    lags=1,  # lags in the model
    newey_lags=2,  # maximum lags when estimating Newey-West standard errors
    ci_width=opt_ci  # width of confidence band
)
irfplot_lp_boombust2d = lp.IRFPlot(
    irf=irf_lp_boombust2d,  # take output from the estimated model
    response=['po_boombust'],  # plot only response of xx ...
    shock=order_boombust2d,  # ... to shocks from all variables
    n_columns=2,  # max columns in the figure
    n_rows=2,  # max rows in the figure
    maintitle='Local Projections: Impulse Responses',  # self-defined title of the IRF plot
    show_fig=False,
    save_pic=True,
    out_path='Output/',
    out_name='ComparingPO_SupplyDemand_LP_BoombustOneSidedAvg2d_IRF'
)


# X --- COMPILE ALL CHARTS


def pil_img2pdf(list_images, extension='png', img_path='Output/', pdf_name='ComparingPO_SupplyDemand_AllCharts'):
    seq = list_images.copy()  # deep copy
    list_img = []
    file_pdf = img_path + pdf_name + '.pdf'
    run = 0
    for i in seq:
        img = Image.open(img_path + i + '.' + extension)
        img = img.convert('RGB')  # PIL cannot save RGBA files as pdf
        if run == 0:
            first_img = img.copy()
        elif run > 0:
            list_img = list_img + [img]
        run += 1
    first_img.save(img_path + pdf_name + '.pdf',
                   'PDF',
                   resolution=100.0,
                   save_all=True,
                   append_images=list_img)


seq_output = [
    'ComparingPO_SupplyDemand_VAR_Base1a_OIRF',
    'ComparingPO_SupplyDemand_VAR_Pluck1a_OIRF',
    'ComparingPO_SupplyDemand_VAR_BoombustOneSidedAvg1a_OIRF',
    'ComparingPO_SupplyDemand_VAR_Base1b_OIRF',
    'ComparingPO_SupplyDemand_VAR_Pluck1b_OIRF',
    'ComparingPO_SupplyDemand_VAR_BoombustOneSidedAvg1b_OIRF',
    'ComparingPO_SupplyDemand_VAR_Base1c_OIRF',
    'ComparingPO_SupplyDemand_VAR_Pluck1c_OIRF',
    'ComparingPO_SupplyDemand_VAR_BoombustOneSidedAvg1c_OIRF',
    'ComparingPO_SupplyDemand_VAR_Base1d_OIRF',
    'ComparingPO_SupplyDemand_VAR_Pluck1d_OIRF',
    'ComparingPO_SupplyDemand_VAR_BoombustOneSidedAvg1d_OIRF',
    'ComparingPO_SupplyDemand_VAR_Base2a_OIRF',
    'ComparingPO_SupplyDemand_VAR_Pluck2a_OIRF',
    'ComparingPO_SupplyDemand_VAR_BoombustOneSidedAvg2a_OIRF',
    'ComparingPO_SupplyDemand_VAR_Base2b_OIRF',
    'ComparingPO_SupplyDemand_VAR_Pluck2b_OIRF',
    'ComparingPO_SupplyDemand_VAR_BoombustOneSidedAvg2b_OIRF',
    'ComparingPO_SupplyDemand_VAR_Base2c_OIRF',
    'ComparingPO_SupplyDemand_VAR_Pluck2c_OIRF',
    'ComparingPO_SupplyDemand_VAR_BoombustOneSidedAvg2c_OIRF',
    'ComparingPO_SupplyDemand_VAR_Base2d_OIRF',
    'ComparingPO_SupplyDemand_VAR_Pluck2d_OIRF',
    'ComparingPO_SupplyDemand_VAR_BoombustOneSidedAvg2d_OIRF',
    'ComparingPO_SupplyDemand_LP_Base1a_IRF',
    'ComparingPO_SupplyDemand_LP_Pluck1a_IRF',
    'ComparingPO_SupplyDemand_LP_BoombustOneSidedAvg1a_IRF',
    'ComparingPO_SupplyDemand_LP_Base1b_IRF',
    'ComparingPO_SupplyDemand_LP_Pluck1b_IRF',
    'ComparingPO_SupplyDemand_LP_BoombustOneSidedAvg1b_IRF',
    'ComparingPO_SupplyDemand_LP_Base1c_IRF',
    'ComparingPO_SupplyDemand_LP_Pluck1c_IRF',
    'ComparingPO_SupplyDemand_LP_BoombustOneSidedAvg1c_IRF',
    'ComparingPO_SupplyDemand_LP_Base1d_IRF',
    'ComparingPO_SupplyDemand_LP_Pluck1d_IRF',
    'ComparingPO_SupplyDemand_LP_BoombustOneSidedAvg1d_IRF',
    'ComparingPO_SupplyDemand_LP_Base2a_IRF',
    'ComparingPO_SupplyDemand_LP_Pluck2a_IRF',
    'ComparingPO_SupplyDemand_LP_BoombustOneSidedAvg2a_IRF',
    'ComparingPO_SupplyDemand_LP_Base2b_IRF',
    'ComparingPO_SupplyDemand_LP_Pluck2b_IRF',
    'ComparingPO_SupplyDemand_LP_BoombustOneSidedAvg2b_IRF',
    'ComparingPO_SupplyDemand_LP_Base2c_IRF',
    'ComparingPO_SupplyDemand_LP_Pluck2c_IRF',
    'ComparingPO_SupplyDemand_LP_BoombustOneSidedAvg2c_IRF',
    'ComparingPO_SupplyDemand_LP_Base2d_IRF',
    'ComparingPO_SupplyDemand_LP_Pluck2d_IRF',
    'ComparingPO_SupplyDemand_LP_BoombustOneSidedAvg2d_IRF',
]
pil_img2pdf(list_images=seq_output,
            img_path='Output/',
            extension='png',
            pdf_name='ComparingPO_SupplyDemand_AllCharts_OneSidedAvg')
telsendfiles(conf=tel_config,
             path='Output/ComparingPO_SupplyDemand_AllCharts_OneSidedAvg.pdf',
             cap='All charts on comparing the responses of PO estimates to supply and demand shocks')

# XX --- Notify
telsendmsg(conf=tel_config,
           msg='comparingpo_supply_demand_onesidedavg: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
