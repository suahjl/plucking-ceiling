import gc

import pandas as pd
import numpy as np
from datetime import date, timedelta
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
import plotly.graph_objects as go
import telegram_send
import dataframe_image as dfi
import time
from tqdm import tqdm
from dotenv import load_dotenv
import os
import ast
import pyeviews as evp

time_start = time.time()

# 0 --- Main settings
load_dotenv()
t_start = '1995Q4'
t_burnin = str(pd.to_datetime(t_start).to_period('Q') + 26)  # 26Q burn-in
t_start_plus1 = str(pd.to_datetime(t_start).to_period('Q') + 1)  # 1Q after start of time series
tel_config = os.getenv('TEL_CONFIG')
use_forecast = ast.literal_eval(os.getenv('USE_FORECAST_BOOL'))
if use_forecast:
    file_suffix_fcast = '_forecast'
    fcast_start = '2023Q1'
    t_burnin = str(pd.to_datetime(t_start).to_period('Q') + 44)  # 44Q burn-in if include forecasts (end-point issues)
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
df = pd.read_parquet('boombustpo_input_data_kf_onesided' + file_suffix_fcast + '.parquet')
df['quarter'] = pd.to_datetime(df['quarter']).dt.to_period('Q')
df = df.set_index('quarter')

# III --- Estimate initial parameter values


def est_init_values(data):
    d = data.copy()

    # ProdFunc PO log
    mod = smf.ols(
        'ln_po_pf ~ ln_po_pf.shift(1) - 1',
        data=d,
        offset=1 * d['ln_po_pf_d'].shift(1)
    )  # fix coef on logdiff(po_pf) to 1
    res = mod.fit()
    po_pf_se = np.sqrt(1 - res.rsquared_adj) * np.std(d['ln_po_pf'])

    # ProdFunc PO logdiff
    d['new_y'] = d['ln_po_pf_d'] - d['ln_po_pf_d_trend']
    d['new_x'] = d['ln_po_pf_d'].shift(1) - d['ln_po_pf_d_trend']
    mod = smf.ols(
        'new_y ~ new_x - 1',
        data=d,
    )
    res = mod.fit()
    po_pf_d_beta1 = res.params['new_x']
    po_pf_d_se = np.sqrt(1 - res.rsquared_adj) * np.std(d['new_y'])
    del d['new_y']
    del d['new_x']

    # ProdFunc output gap
    mod = smf.ols(
        'output_gap_pf ~ output_gap_pf.shift(1) + output_gap_pf.shift(2) - 1',
        data=d
    )
    res = mod.fit()
    output_gap_pf_beta_1 = res.params['output_gap_pf.shift(1)']
    output_gap_pf_beta_2 = res.params['output_gap_pf.shift(2)']
    output_gap_pf_se = np.sqrt(1 - res.rsquared_adj) * np.std(d['output_gap_pf'])

    # Phillips curve (core cpi, output gap, brent gap)
    d['new_y'] = d['ln_cpi_core_d'] - d['ln_cpi_core_d'].shift(2)
    d['new_x1'] = d['ln_cpi_core_d'].shift(1) - d['ln_cpi_core_d'].shift(2)
    mod = smf.ols(
        'new_y ~ new_x1 + output_gap_pf.shift(1) + com_2q_gap - 1',
        data=d,
    )
    res = mod.fit()
    pc_beta1 = res.params['new_x1']
    pc_beta2 = res.params['output_gap_pf.shift(1)']
    pc_beta3 = res.params['com_2q_gap']
    pc_se = np.sqrt(1 - res.rsquared_adj) * np.std(d['new_y'])
    del d['new_y']
    del d['new_x1']

    # Initial lambda
    lambda_init = po_pf_d_se / po_pf_se  # 0.45 # po_pf_d_se / po_pf_se

    # Compile all 'initial values'
    all_init_values = pd.DataFrame(
        {'po_pf_se': [po_pf_se],
         'po_pf_d_beta1': [po_pf_d_beta1],
         'po_pf_d_se': [po_pf_d_se],
         'output_gap_pf_beta_1': [output_gap_pf_beta_1],
         'output_gap_pf_beta_2': [output_gap_pf_beta_2],
         'output_gap_pf_se': [output_gap_pf_se],
         'pc_beta1': [pc_beta1],
         'pc_beta2': [pc_beta2],
         'pc_beta3': [pc_beta3],
         'pc_se': [pc_se],
         'lambda_init': [lambda_init]}
    )
    all_init_values = all_init_values.transpose().rename(columns={0:'init_values'})

    return all_init_values


all_init_values = est_init_values(data=df)


# IV --- Estimate NKPC Using Kalman Filter


def kfilter_po_evp(data, initial_values):

    # A. Create deep copies of input
    init_val = initial_values.copy()  # all_init_values

    # B. Reformat time index
    d = data.copy().reset_index()  # Different date columns (in timestamp, but last day of the quarter)
    qgap = (d['quarter'].max() - d['quarter'].min()).n + 1
    qmin = str(d['quarter'].min())
    del d['quarter']
    d['quarter'] = pd.date_range(qmin, periods=qgap, freq='Q')
    d = d.set_index('quarter')

    # C. Setup EViews
    # Connect to COM
    eviewsapp = evp.GetEViewsApp(instance='new', showwindow=True)

    # Port dataframe as EViews workfile
    evp.PutPythonAsWF(d, app=eviewsapp, newwf=True)

    # D. The state-space model
    # Set up object
    evp.Run('scalar lambda = ' + str(init_val.loc['lambda_init', 'init_values']),
            app=eviewsapp)  # initial lambda value
    evp.Run('sspace kalmanpo',
            app=eviewsapp)  # create state-space object

    # Set up equations
    evp.Run('kalmanpo.append @signal ln_gdp = ln_po_pf + output_gap_pf',
            app=eviewsapp)  # Y_{t} = Yp_{t} + x_{t}
    evp.Run('kalmanpo.append @state ln_po_pf = ln_po_pf(-1) + ln_po_pf_d(-1) + [var = c(1)^2]',
            app=eviewsapp)  # Yp_t = Yp_{t-1} + delta(Yp_{t-1}) + e_t
    evp.Run(
        'kalmanpo.append @state ln_po_pf_d = c(2)*ln_po_pf_d(-1) + (1 - c(2))*ln_po_pf_d_trend + [var = (lambda * c(1))^2]',
        app=eviewsapp)  # delta(Yp_{t}) = beta1*delta(Yp_{t-1}) + (1-beta1)*delta(Yp)_{trend} + e_t
    evp.Run(
        'kalmanpo.append @state output_gap_pf = c(4)*output_gap_pf(-1) + c(11)*output_gap_pf_lag1(-1) + [var = c(5)^2]',
        app=eviewsapp)  # x_{t} = beta2*x_{t-1} + beta3*x_{t-2} + e_t
    evp.Run('kalmanpo.append @state output_gap_pf_lag1 = output_gap_pf(-1)',
            app=eviewsapp)  # x_{t-1} = x_{t-1}
    evp.Run(
        'kalmanpo.append @signal ln_cpi_core_d = c(6)*ln_cpi_core_d(-1) + (1-c(6))*ln_cpi_core_d(-2) + c(7)*output_gap_pf + c(9)*com_2q_gap + [var = c(10)^2]',
        app=eviewsapp)  # pi_{t} = beta3*pi_{t-1} + beta4*pi_{t-2} + beta5*x_{t} + beta6*oilpgap_{t} + e_t

    # Retrieve initial state values
    evp.Run('c(1)=' + str(init_val.loc['po_pf_se', 'init_values']), app=eviewsapp)  # alt: 2.5
    evp.Run('c(2)=' + str(init_val.loc['po_pf_d_beta1', 'init_values']), app=eviewsapp)  # alt: 1
    evp.Run('c(3)=' + str(init_val.loc['po_pf_d_se', 'init_values']), app=eviewsapp)  # alt: 1
    evp.Run('c(4)=' + str(init_val.loc['output_gap_pf_beta_1', 'init_values']), app=eviewsapp)
    evp.Run('c(11)=' + str(init_val.loc['output_gap_pf_beta_2', 'init_values']), app=eviewsapp)
    evp.Run('c(5)=' + str(init_val.loc['output_gap_pf_se', 'init_values']), app=eviewsapp)
    evp.Run('c(6)=' + str(init_val.loc['pc_beta1', 'init_values']), app=eviewsapp)
    evp.Run('c(7)=' + str(init_val.loc['pc_beta2', 'init_values']), app=eviewsapp)
    evp.Run('c(8)=' + str(init_val.loc['pc_beta3', 'init_values']), app=eviewsapp)
    evp.Run('c(9)=' + str(init_val.loc['pc_beta3', 'init_values']), app=eviewsapp)
    evp.Run('c(10)=' + str(init_val.loc['pc_se', 'init_values']), app=eviewsapp)

    # Retrieve initial values of potential output and output gap (using HP-filtered PO instead)
    po_hp_ini = data.loc[data.index == t_start_plus1, 'po_hp'][0]
    po_hp_d_ini = 0.0482 / 4
    output_gap_hp_ini = data.loc[data.index == t_start_plus1, 'output_gap_hp'][0]
    output_gap_hp_ini2 = data.loc[data.index == t_start, 'output_gap_hp'][0]

    # Create vector of initial states of potential output and output gap
    evp.Run('vector(4) ini_states', app=eviewsapp)
    evp.Run('scalar po_hp_ini = ' + str(po_hp_ini), app=eviewsapp)
    evp.Run('scalar po_hp_d_ini = ' + str(po_hp_d_ini), app=eviewsapp)
    evp.Run('scalar output_gap_hp_ini = ' + str(output_gap_hp_ini), app=eviewsapp)
    evp.Run('scalar output_gap_hp_ini2 = ' + str(output_gap_hp_ini2), app=eviewsapp)
    evp.Run('ini_states.fill po_hp_ini, po_hp_d_ini, output_gap_hp_ini, output_gap_hp_ini2', app=eviewsapp)

    # Estimate model
    evp.Run('kalmanpo.append @mprior ini_states', app=eviewsapp)  # Append vector containing initial states
    evp.Run('kalmanpo.ml', app=eviewsapp)  # MLE estimation

    evp.Run('kalmanpo.forecast(i=o,m=s) @state *_kal', app=eviewsapp)  # Forecast / predict

    evp.Run('pagecopy ln_po_pf_kal output_gap_pf_kal', app=eviewsapp)  # Retrieve forecast values of ln(Yp) and x)

    # E. Cleaning + housekeeping
    kfilter_output = evp.GetWFAsPython(app=eviewsapp)  # Send back to python as pandas dataframe
    # Switch off EViews COM
    eviewsapp.Hide()
    eviewsapp = None
    evp.Cleanup()
    # Clean output data frame
    kfilter_output = kfilter_output.rename(columns={'LN_PO_PF_KAL': 'ln_po_kf',
                                                    'OUTPUT_GAP_PF_KAL': 'output_gap_kf'})  # rename output
    kfilter_output['po_kf'] = np.exp(kfilter_output['ln_po_kf'])  # convert back into levels
    # Revert time index back to input setting (Q-DEC, YYYYQQ)
    kfilter_output = kfilter_output.reset_index()
    kfilter_output = kfilter_output.rename(columns={'index': 'quarter'})
    kfilter_output['quarter'] = kfilter_output['quarter'].dt.to_period('Q')
    kfilter_output = kfilter_output.set_index('quarter')

    return kfilter_output


kfilter_po = kfilter_po_evp(data=df, initial_values=all_init_values)


# V --- Consolidate output
df = pd.concat([df, kfilter_po], axis=1)  # left-right concat
list_col_output = ['gdp', 'po_pf', 'po_kf',
                   'output_gap_pf', 'output_gap_kf',
                   'com_2q_gap', 'usdmyr_gap', 'ln_cpi_core_d']
df = df[list_col_output]

# Blank out po_kf during burn-in period
df.loc[df.index <= t_burnin, 'po_kf'] = np.nan

# Calculate averages of methods
df['po_avg'] = (df['po_pf'] + df['po_kf']) / 2
df.loc[df.index <= t_burnin, 'po_avg'] = df['po_pf'].copy()  # for burn-in period, take PF values
df['output_gap_avg'] = (df['gdp'] / df['po_avg'] - 1)  # will be multiplied by 100 next

# Convert gaps (and log diff of CPI) into percentages
list_col_perc = ['output_gap_pf', 'output_gap_kf', 'output_gap_avg', 'com_2q_gap', 'usdmyr_gap', 'ln_cpi_core_d']
for i in list_col_perc:
    df[i] = 100 * df[i]

# Reorganise columns
list_col_output = ['gdp', 'po_pf', 'po_kf', 'po_avg',
                   'output_gap_pf', 'output_gap_kf', 'output_gap_avg',
                   'com_2q_gap', 'usdmyr_gap', 'ln_cpi_core_d']
df = df[list_col_output]

# Calculate YoY growth of GDP and PO
list_col = ['gdp', 'po_pf', 'po_kf', 'po_avg']
list_col_yoy = [i + '_yoy' for i in list_col]
for i, j in zip(list_col, list_col_yoy):
    df[j] = 100 * ((df[i] / df[i].shift(4)) - 1)

# VI --- Export data frames
df = df.reset_index()
df['quarter'] = df['quarter'].astype('str')
df.to_parquet('boombustpo_estimates_kf_onesided' + file_suffix_fcast + '.parquet')

# V --- Notify
telsendmsg(conf=tel_config,
           msg='boombustpo_compute_kf_onesided: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
