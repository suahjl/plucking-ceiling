# ------------ Runs the entire PF-KF script using multiple data vintages

import gc

import pandas as pd
import numpy as np
from datetime import date, timedelta
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
from quantecon import hamilton_filter
import plotly.graph_objects as go
import telegram_send
import dataframe_image as dfi
import time
from tqdm import tqdm

from ceic_api_client.pyceic import Ceic

import pyeviews as evp

time_start = time.time()

# 0 --- Main settings
t_start = '1995Q4'
t_output_start = str(pd.to_datetime(t_start).to_period('Q') + 26)  # 26Q burn-in
t_start_plus1 = str(pd.to_datetime(t_start).to_period('Q') + 1)  # 1Q after start of time series
t_now = str(pd.to_datetime(str(date.today())).to_period('Q'))
list_t_ends = ['2007Q2', '2008Q2', '2009Q3', '2015Q4', '2019Q4', '2022Q2', '2027Q4']
list_colours = ['lightcoral', 'crimson', 'red', 'steelblue', 'darkblue', 'gray', 'black']
list_dash_styles = ['solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'longdash']
dict_revision_pairs = {'2009Q3': '2007Q2',
                       '2019Q4': '2015Q4',
                       '2022Q2': '2019Q4',
                       '2027Q4': '2022Q2'}
tel_config = 'EcMetrics_Config_GeneralFlow.conf'  # EcMetrics_Config_GeneralFlow EcMetrics_Config_RMU

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


def jek_clean(
        data,
        trim_start=None,
        trim_end=None,
        seasonal_adj=True,
        log_transform=True,
        filter_using_hamilton=False
):
    # Trimming and renaming columns
    d = data.copy()
    d['quarter'] = pd.to_datetime(d['quarter']).dt.to_period('Q')

    # Timebound
    if trim_start is not None:
        d = d[d['quarter'] >= trim_start]
    if trim_end is not None:
        d = d[d['quarter'] <= trim_end]

    d = d.set_index('quarter')

    # Generate YoY growth on unseasonal-adjusted GDP
    d['gdp15_yoy'] = 100 * ((d['gdp15'] / d['gdp15'].shift(-4)) - 1)

    # Seasonal adjustment: only GDP, labour, and employment
    list_col = ['gdp15', 'labour', 'employment']
    if seasonal_adj:
        for i in list_col:
            sadj_res = smt.x13_arima_analysis(d[i])
            sadj_seasadj = sadj_res.seasadj
            d[i] = sadj_seasadj

    # Take logs post-seasonal adjustment: now including capital stock
    list_col = list_col + ['k_stock15']
    list_col_ln = ['ln_' + i for i in list_col]
    if log_transform:
        for i, j in zip(list_col, list_col_ln):
            d[j] = np.log(d[i])

    # filter trend
    list_col_ln_trend = [i + '_trend' for i in list_col_ln]
    if not filter_using_hamilton:
        for i, j in zip(list_col_ln, list_col_ln_trend):
            cycle, trend = smt.filters.hpfilter(d[i], lamb=1600)
            d[j] = trend  # don't replace original with trend component
    elif filter_using_hamilton:
        for i, j in zip(list_col_ln, list_col_ln_trend):
            cycle, trend = hamilton_filter(d[i], h=8)  # 2 years (2 for annual, 8 for quarter, 24 for monthly, ...)
            d[j] = trend  # don't replace original with trend component

    return d


def prodfunc_po(
        data,
        use_labour=True,
        filter_using_hamilton=False
):
    d = data.copy()

    # TFP: a*ln(k) + (1-a)*ln(l)
    if use_labour:
        d['implied_y'] = (d['alpha'] / 100) * d['ln_k_stock15'] + (1 - d['alpha'] / 100) * d['ln_labour']
    elif not use_labour:
        d['implied_y'] = (d['alpha'] / 100) * d['k_stock15'] + (1 - d['alpha'] / 100) * d['ln_employment']
    d['ln_tfp'] = d['ln_gdp15'] - d['implied_y']  # ln(tfp)

    # TFP trend
    if not filter_using_hamilton:
        cycle, trend = smt.filters.hpfilter(d['ln_tfp'], lamb=1600)
        d['ln_tfp_trend'] = trend  # don't replace original with trend component
    elif filter_using_hamilton:
        cycle, trend = hamilton_filter(d['ln_tfp'], h=8)  # 2 years (2 for annual, 8 for quarter, 24 for monthly, ...)
        d['ln_tfp_trend'] = trend  # don't replace original with trend component

    # Calculate potential output
    if use_labour:
        d['ln_po'] = d['ln_tfp_trend'] + \
                     ((d['alpha'] / 100) * d['ln_k_stock15_trend']) + \
                     (1 - d['alpha'] / 100) * d['ln_labour_trend']
    elif not use_labour:
        d['ln_po'] = d['ln_tfp_trend'] + \
                     ((d['alpha'] / 100) * d['ln_k_stock15_trend']) + \
                     (1 - d['alpha'] / 100) * d['ln_employment_trend']

    # Back out levels (PO)
    d['po'] = np.exp(d['ln_po'])
    d['output_gap'] = 100 * (d['gdp15'] / d['po'] - 1)  # % PO
    d['capital_input'] = np.exp(d['ln_k_stock15_trend'])
    if use_labour: d['labour_input'] = np.exp(d['ln_labour_trend'])
    elif not use_labour: d['labour_input'] = np.exp(d['ln_employment_trend'])
    d['tfp_input'] = np.exp(d['ln_tfp_trend'])

    # Back out levels (observed output)
    d['capital_observed'] = np.exp(d['ln_k_stock15'])
    if use_labour: d['labour_observed'] = np.exp(d['ln_labour'])
    if use_labour: d['labour_observed'] = np.exp(d['ln_employment'])
    d['tfp_observed'] = np.exp(d['ln_tfp'])

    # trim data frame
    list_col_keep = ['output_gap', 'po', 'gdp15',
                     'capital_input', 'labour_input', 'tfp_input',
                     'capital_observed', 'labour_observed', 'tfp_observed',
                     'alpha']
    d = d[list_col_keep]

    return d


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


def kfilter_po_evp(data, initial_values):

    # A. Create deep copies of input
    all_init_values = initial_values.copy()

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
    evp.Run('scalar lambda = ' + str(all_init_values.loc['lambda_init', 'init_values']),
            app=eviewsapp)  # initial lambda value
    evp.Run('sspace kalmanpo',
            app=eviewsapp)  # create state-space object

    # Set up equations
    evp.Run('kalmanpo.append @signal ln_gdp15 = ln_po_pf + output_gap_pf',
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
    evp.Run('c(1)=' + str(all_init_values.loc['po_pf_se', 'init_values']), app=eviewsapp)  # alt: 2.5
    evp.Run('c(2)=' + str(all_init_values.loc['po_pf_d_beta1', 'init_values']), app=eviewsapp)  # alt: 1
    evp.Run('c(3)=' + str(all_init_values.loc['po_pf_d_se', 'init_values']), app=eviewsapp)  # alt: 1
    evp.Run('c(4)=' + str(all_init_values.loc['output_gap_pf_beta_1', 'init_values']), app=eviewsapp)
    evp.Run('c(11)=' + str(all_init_values.loc['output_gap_pf_beta_2', 'init_values']), app=eviewsapp)
    evp.Run('c(5)=' + str(all_init_values.loc['output_gap_pf_se', 'init_values']), app=eviewsapp)
    evp.Run('c(6)=' + str(all_init_values.loc['pc_beta1', 'init_values']), app=eviewsapp)
    evp.Run('c(7)=' + str(all_init_values.loc['pc_beta2', 'init_values']), app=eviewsapp)
    evp.Run('c(8)=' + str(all_init_values.loc['pc_beta3', 'init_values']), app=eviewsapp)
    evp.Run('c(9)=' + str(all_init_values.loc['pc_beta3', 'init_values']), app=eviewsapp)
    evp.Run('c(10)=' + str(all_init_values.loc['pc_se', 'init_values']), app=eviewsapp)

    # Retrieve initial values of potential output and output gap (using HP-filtered PO instead)
    po_hp_ini = df.loc[df.index == t_start_plus1, 'po_hp'][0]
    po_hp_d_ini = 0.0482 / 4
    output_gap_hp_ini = df.loc[df.index == t_start_plus1, 'output_gap_hp'][0]
    output_gap_hp_ini2 = df.loc[df.index == t_start, 'output_gap_hp'][0]

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

# ------------------------ PULL FROM CEIC ------------------------

# II --- Load CEIC
Ceic.login("suahjinglian@bnm.gov.my ", "dream1234")  # login to CEIC
series_id = pd.read_csv('ceic_seriesid_forkf.txt', sep='|')
series_id = list(series_id['series_id'])
df_ceic_full = ceic2pandas_ts(series_id, start_date=date(1987, 5, 20)).fillna(method='ffill')

# III.A --- Input data
df_raw = pd.read_csv('testdata.txt', sep='|')

# ------------------------ [LOOP BEGINS HERE] ------------------------

round = 1
for t_end in tqdm(list_t_ends):

    # III.B --- Input data per loop
    df = jek_clean(data=df_raw,
                   trim_start=None,
                   trim_end=t_end,
                   seasonal_adj=True,
                   log_transform=True,
                   filter_using_hamilton=False)

    # ------------------------ PRODUCTION FUNCTION ------------------------

    # IV --- Execution
    # Estimate PO
    df_pf = prodfunc_po(data=df,
                        use_labour=True,
                        filter_using_hamilton=False)


    # ------------------------ KALMAN FILTER ------------------------

    # V --- Data (KF)
    # A. CEIC
    # Cleaning CEIC downloads: merging old + new brent
    df_ceic = df_ceic_full.copy()
    dict_ceic_rename = {'Crude Oil: Spot Price: Brent': 'com',
                        'FX Spot Rate: FRB: Malaysian Ringgit': 'usdmyr',
                        'Consumer Price Index (CPI): Core': 'cpi_core'}
    df_ceic = df_ceic.rename(columns=dict_ceic_rename)
    df_ceic = df_ceic.reset_index()
    df_ceic = df_ceic.rename(columns={'index': 'quarter'})
    df_ceic['quarter'] = pd.to_datetime(df_ceic['quarter']).dt.to_period('Q')
    df_ceic = df_ceic.groupby('quarter').agg('mean')
    # Load static historical core cpi
    core_cpi_old = pd.read_csv('old_static_cpi_core.txt', sep=',')
    core_cpi_old['quarter'] = pd.to_datetime(core_cpi_old['quarter']).dt.to_period('Q')
    core_cpi_old = core_cpi_old.set_index('quarter')
    core_cpi_old = core_cpi_old.rename(columns={'cpi_core': 'cpi_core_old'})
    # Merge
    df = pd.concat([df_ceic, core_cpi_old], axis=1)
    df.loc[df.index < '2015Q1', 'cpi_core'] = df['cpi_core_old']  # merge old and new cpi_core
    del df['cpi_core_old']  # delete old cpi_core
    # Trim time period
    df = df[df.index <= t_end]

    # B. ProdFunc Output
    # Import output from ProdFunc
    df_pf = df_pf.rename(columns={'output_gap': 'output_gap_pf',
                                  'po': 'po_pf'})
    # Merge ProdFunc with macro data
    df = pd.concat([df, df_pf], axis=1)
    # New column
    df['com_2q'] = ((df['com'] + df['com'].shift(1)) / 2)  # 2QMA

    # C. Cleaning
    # Trim input data
    df = df[(df.index >= t_start) & (df.index <= t_end)]

    # Seasonally adjust core cpi and com prices (2QMA)
    list_col = ['cpi_core', 'com', 'com_2q']
    for i in tqdm(list_col):
        sadj_res = smt.x13_arima_analysis(df.loc[df[i].notna(), i])  # handles NAs
        sadj_seasadj = sadj_res.seasadj
        df[i] = sadj_seasadj

    # Take logs
    list_col = list_col + ['gdp15', 'po_pf', 'usdmyr']
    list_col_ln = ['ln_' + i for i in list_col]
    for i, j in tqdm(zip(list_col, list_col_ln)):
        df[j] = np.log(df[i])  # log(x)

    # Take diff
    list_col_ln_d = [i + '_d' for i in list_col_ln]
    for i, j in tqdm(zip(list_col_ln, list_col_ln_d)):
        df[j] = df[i] - df[i].shift(1)  # logdiff(x)

    # Gaps (relative to sample avg)
    list_col_ln_d = ['ln_' + i + '_d' for i in ['usdmyr', 'com', 'com_2q']]
    list_col_ln_d_gap = [i + '_gap' for i in ['usdmyr', 'com', 'com_2q']]
    for i, j in tqdm(zip(list_col_ln_d, list_col_ln_d_gap)):
        df[j] = df[i] - df[i].mean()  # logdiff(x) - mean(logdiff(x))

    # Rescale variables
    list_col = ['output_gap_pf']
    for i in tqdm(list_col):
        df[i] = df[i] / 100

    # D. Additional
    # Assumed trend growth for ProdFunc PO
    df['ln_po_pf_d_trend'] = np.power(1.0479, 0.25) - 1
    # HP-filtered PO (for initial states later)
    trend, cycle = smt.filters.hpfilter(df['ln_gdp15'].dropna(), lamb=1600)
    df['po_hp'] = trend
    df['output_gap_hp'] = cycle

    # VI --- Execution
    # Obtain initial values
    all_init_values = est_init_values(data=df)

    # Estimate potential output
    kfilter_po = kfilter_po_evp(data=df, initial_values=all_init_values)

    # Merge output back with original dataframe
    df = pd.concat([df, kfilter_po], axis=1)  # left-right concat
    list_col_output = ['gdp15', 'po_pf', 'po_kf', 'output_gap_pf', 'output_gap_kf', 'com_2q_gap', 'usdmyr_gap', 'ln_cpi_core_d']
    df_kf = df[list_col_output]

    # Calculate averages of methods
    df_kf['po_avg'] = (df_kf['po_pf'] + df_kf['po_kf']) / 2
    df_kf['output_gap_avg'] = (df_kf['gdp15'] / df_kf['po_avg'] - 1)  # will be multiplied by 100 next

    # Convert gaps (and log diff of CPI) into percentages
    list_col_perc = ['output_gap_pf', 'output_gap_kf', 'output_gap_avg', 'com_2q_gap', 'usdmyr_gap', 'ln_cpi_core_d']
    for i in list_col_perc:
        df_kf[i] = 100 * df_kf[i]

    # Reorganise columns
    list_col_output = ['gdp15', 'po_pf', 'po_kf', 'po_avg',
                       'output_gap_pf', 'output_gap_kf', 'output_gap_avg',
                       'com_2q_gap', 'usdmyr_gap', 'ln_cpi_core_d']
    df_kf = df_kf[list_col_output]

    # Calculate YoY growth of GDP and PO
    list_col = ['gdp15', 'po_pf', 'po_kf', 'po_avg']
    list_col_yoy = [i + '_yoy' for i in list_col]
    for i, j in zip(list_col, list_col_yoy):
        df_kf[j] = 100 * ((df_kf[i] / df_kf[i].shift(4)) - 1)

    # Trim burn-in period
    df_kf = df_kf[df_kf.index >= t_output_start]

    # ------------------------ CONSOLIDATING VINTAGES ------------------------

    # VII --- Merging
    if round == 1:
        df_final = pd.DataFrame(df_kf['output_gap_avg']).rename(columns={'output_gap_avg': t_end})
    elif round > 1:
        df_final = pd.concat([df_final,
                              pd.DataFrame(df_kf['output_gap_avg']).rename(columns={'output_gap_avg': t_end})],
                             axis=1)
    round += 1

# ------------------------ PLOTTING VINTAGES ------------------------

# VIII --- Plotting


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
    fig.write_image('Output/BoomBustPO_Vintage_' + output_suffix + '.png', height=768, width=1366)
    fig.write_html('Output/BoomBustPO_Vintage_' + output_suffix + '.html')
    return fig

df_final['ref'] = 0

fig_vintages = plot_linechart(
    data=df_final,
    cols=list_t_ends + ['ref'],
    nice_names=list_t_ends + ['Reference (Y=0)'],
    colours=list_colours + ['black'],
    dash_styles=list_dash_styles + ['solid'],
    y_axis_title='% Potential Output',
    main_title='Vintages of Current Output Gap Estimates (Average of PF and KF Methods)',
    output_suffix='OutputGap'
)
telsendimg(conf=tel_config,
           path='Output/BoomBustPO_Vintage_OutputGap.png',
           cap='Vintages of Current Output Gap Estimates (Average of PF and KF Methods)')

# ------------------------ PLOTTING SIZE OF REVISIONS ------------------------

# IX --- Calculating revision sizes by pairs
df_rev = pd.DataFrame(columns=['rev_consol'])
round = 1
for post, pre in tqdm(dict_revision_pairs.items()):
    df_rev['rev_' + post + '_' + pre] = df_final[post] - df_final[pre]
    if round == 1:
        df_rev['rev_consol'] = df_rev['rev_' + post + '_' + pre].copy()
    elif round > 1:
        df_rev.loc[df_rev['rev_consol'].isna(), 'rev_consol'] = df_rev['rev_' + post + '_' + pre]
    round += 1
df_rev = df_rev[df_rev.index <= list_t_ends[-2]]

# X --- Plotting


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
    fig.write_image('Output/BoomBustPO_Vintage_' + output_suffix + '.png', height=768, width=1366)
    fig.write_html('Output/BoomBustPO_Vintage_' + output_suffix + '.html')
    return fig


fig_rev = plot_areachart(
    data=df_rev,
    cols=['rev_consol'],
    nice_names=['Revisions'],
    colours=['lightcoral'],
    y_axis_title='Percentage Points (% Potential Output)',
    main_title='Revisions in Current Output Gap Across Consecutive Vintages',
    show_legend=False,
    ymin=-5,
    ymax=5,
    output_suffix='OutputGap_Revisions'
)
telsendimg(conf=tel_config,
           path='Output/BoomBustPO_Vintage_OutputGap_Revisions.png',
           cap='Revisions in Current Output Gap Across Consecutive Vintages')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
