# %%
# Check sensitivity to lambda in HP filter
# %%
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

# %%
# 0 --- Main settings
load_dotenv()
tel_config = os.getenv('TEL_CONFIG')
path_output = "./Output/"
T_lb = '1995Q1'
T_lb_day = date(1995, 1, 1)
show_conf_bands = False
use_forecast = ast.literal_eval(os.getenv('USE_FORECAST_BOOL'))
if use_forecast:
    file_suffix_fcast = '_forecast'
    fcast_start = '2023Q1'
elif not use_forecast:
    file_suffix_fcast = ''

# %%
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


# %%
# II --- Load data
# use same data input as for plucking
df = pd.read_parquet('pluckingpo_input_data' + file_suffix_fcast + '.parquet')
df['quarter'] = pd.to_datetime(df['quarter']).dt.to_period('Q')
df = df.set_index('quarter')

# %%
# III --- Set up parameters
# Lambdas to loop over
list_hplambdas = [1600, 3200, 8000, 11200]
list_hplambdas.sort()
# Alphas to loop over
alpha_base = df["alpha"].max()
list_alphas = [alpha_base] + [25, 75]
list_alphas.sort()
# Essential column labels
list_col_ln = ['ln_gdp', 'ln_lforce', 'ln_nks']
list_col_ln_trend = [i + '_trend' for i in list_col_ln]
# Colours (steps of 3)
list_colours = [
    "pink", "red", "crimson",
    "lightblue", "blue", "darkblue",
    "lightgreen", "green", "darkgreen",
    "lightgrey", "grey", "black"
]

# %%
# IV --- Define functions


def prodfunc_po(data):
    d = data.copy()

    # TFP: a*ln(k) + (1-a)*ln(l)
    d['implied_y'] = (d['alpha'] / 100) * d['ln_nks'] + \
        (1 - d['alpha'] / 100) * d['ln_lforce']
    d['ln_tfp'] = d['ln_gdp'] - d['implied_y']  # ln(tfp)

    # TFP trend
    cycle, trend = sm.filters.hpfilter(
        d.loc[~d['ln_tfp'].isna(), 'ln_tfp'], lamb=11200)  # deals with NA
    d['ln_tfp_trend'] = trend  # don't replace original with trend component

    # hamilton version
    # cycle, trend = hamilton_filter(d['ln_tfp'], h=8, p=4)  # 2 years (2 for annual, 8 for quarter, 24 for monthly, ...)
    # d['ln_tfp_trend'] = trend  # don't replace original with trend component

    # Calculate potential output
    d['ln_po'] = d['ln_tfp_trend'] + \
        ((d['alpha'] / 100) * d['ln_nks_trend']) + \
        (1 - d['alpha'] / 100) * d['ln_lforce_trend']

    # Back out levels (PO)
    d['po'] = np.exp(d['ln_po'])
    d['output_gap'] = 100 * (d['gdp'] / d['po'] - 1)  # % PO
    d['capital_input'] = np.exp(d['ln_nks_trend'])
    d['labour_input'] = np.exp(d['ln_lforce_trend'])
    d['tfp_input'] = np.exp(d['ln_tfp_trend'])

    # Back out levels (observed output)
    d['capital_observed'] = np.exp(d['ln_nks'])
    d['labour_observed'] = np.exp(d['ln_lforce'])
    d['tfp_observed'] = np.exp(d['ln_tfp'])

    # trim data frame
    list_col_keep = ['output_gap', 'po', 'gdp',
                     'capital_input', 'labour_input', 'tfp_input',
                     'capital_observed', 'labour_observed', 'tfp_observed',
                     'alpha']
    d = d[list_col_keep]

    return d


# diff from other scripts that were written earlier
def plot_linechart(
        data, cols, nice_names,
        colours,
        # dash_styles,
        y_axis_title, main_title,
        use_forecast_choice
):
    d = data.copy()
    fig = go.Figure()
    for col, nice_name, colour in tqdm(zip(cols, nice_names, colours)):
        fig.add_trace(
            go.Scatter(
                x=d.index.astype('str'),
                y=d[col],
                name=nice_name,
                mode='lines',
                line=dict(color=colour)
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
    return fig


# %%
# V --- Run the Holy Puki (HP) Filter
list_nice_param_combos = []
combo_counter = 0
burn_in_duration = 20
for hplambda in tqdm(list_hplambdas):
    for alpha in list_alphas:
        # deep copy per lambda-alpha combo
        df_sub = df.copy()
        # swap alpha
        df_sub["alpha"] = alpha
        # add to master list of nice params combos
        list_nice_param_combos = list_nice_param_combos + \
            ["Lambda=" + str(hplambda) + "; Alpha=" + str(round(alpha, 1))]
        # apply filter to raw y, k, n using specified lambda (one-sided version)
        for i, j in zip(list_col_ln, list_col_ln_trend):
            t_count = 0
            for t in tqdm(list(df_sub.index)):
                if t_count < burn_in_duration:
                    pass
                elif t_count >= burn_in_duration:
                    cycle, trend = sm.filters.hpfilter(
                        df_sub.loc[(~df_sub[i].isna()) & (df_sub.index <= t), i], lamb=hplambda)
                    if t_count == burn_in_duration:
                        # don't replace original with trend component
                        df_sub[j] = trend
                    elif t_count > burn_in_duration:
                        # fill in NA with new trend
                        df_sub.loc[df_sub[j].isna(), j] = trend
                t_count += 1
        # estimate tfp and potential output
        df_pf_sub = prodfunc_po(data=df_sub)
        # split into PO and OG dataframes + trim to columns to be plotted
        df_po_sub = pd.DataFrame(df_pf_sub["po"])
        df_og_sub = pd.DataFrame(df_pf_sub["output_gap"])
        # rename
        df_po_sub = df_po_sub.rename(
            columns={
                "po": "lambda" + str(round(hplambda)) + "_alpha" + str(round(alpha))
            }
        )
        df_og_sub = df_og_sub.rename(
            columns={
                "output_gap": "lambda" + str(round(hplambda)) + "_alpha" + str(round(alpha))
            }
        )
        # consolidate
        if combo_counter == 0:
            df_po_consol = df_po_sub.copy()
            df_og_consol = df_og_sub.copy()
        elif combo_counter > 0:
            df_po_consol = df_po_consol.merge(
                df_po_sub, how="outer", right_index=True, left_index=True)
            df_og_consol = df_og_consol.merge(
                df_og_sub, how="outer", right_index=True, left_index=True)
        # tq next
        combo_counter += 1

# %%
# VI --- Export data frames
# PO
df_po_consol = df_po_consol.reset_index()
df_po_consol['quarter'] = df_po_consol['quarter'].astype('str')
df_po_consol.to_parquet('boombustpo_estimates_pf_onesided_po_parametercheck' +
                        file_suffix_fcast + '.parquet', compression='brotli')
# OG
df_og_consol = df_og_consol.reset_index()
df_og_consol['quarter'] = df_og_consol['quarter'].astype('str')
df_og_consol.to_parquet('boombustpo_estimates_pf_onesided_og_parametercheck' +
                        file_suffix_fcast + '.parquet', compression='brotli')

# %%
# VII --- Plot
# OG
df_og_consol = df_og_consol.set_index("quarter")
fig_og = plot_linechart(
    data=df_og_consol,
    cols=list(df_og_consol.columns),
    colours=list_colours,
    nice_names=list_nice_param_combos,
    y_axis_title="%",
    main_title="Sensitivity check for one-sided HP-filtered PF output gap estimates",
    use_forecast_choice=use_forecast
)
file_name = path_output + "BoombustPO_PF_OneSided_OG_ParameterCheck"
fig_og.write_image(file_name + ".png", height=1080, width=1920)
telsendimg(
    conf=tel_config,
    path=file_name + ".png",
    cap=file_name
)
# PO
df_po_consol = df_po_consol.set_index("quarter")
fig_po = plot_linechart(
    data=df_po_consol,
    cols=list(df_po_consol.columns),
    colours=list_colours,
    nice_names=list_nice_param_combos,
    y_axis_title="%",
    main_title="Sensitivity check for one-sided HP-filtered PF potential output estimates",
    use_forecast_choice=use_forecast
)
file_name = path_output + "BoombustPO_PF_OneSided_PO_ParameterCheck"
fig_po.write_image(file_name + ".png", height=1080, width=1920)
telsendimg(
    conf=tel_config,
    path=file_name + ".png",
    cap=file_name
)

# %%
# VI --- Notify
telsendmsg(conf=tel_config,
           msg='boombustpo_compute_pf_onesided_parametercheck: COMPLETED')

# End
print('\n----- Ran in ' +
      "{:.0f}".format(time.time() - time_start) + ' seconds -----')
# %%
