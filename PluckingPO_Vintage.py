# -------------- Based on Est1c: Bands as 'uncertainty in timing of peaks by 1Q earlier'

import pandas as pd
import numpy as np
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
list_T_ub = ['2007Q2', '2008Q2', '2009Q3', '2015Q4', '2019Q4', '2022Q2']
list_colours = ['lightcoral', 'crimson', 'red', 'steelblue', 'darkblue', 'gray']
list_dash_styles = ['solid', 'solid', 'solid', 'solid', 'solid', 'solid']
dict_revision_pairs = {'2009Q3': '2007Q2',
                       '2019Q4': '2015Q4',
                       '2022Q2': '2019Q4'}
list_threshold = [1, 1, 1, 1, 1, 0.8]
list_interpolate_method = ['slinear', 'slinear', 'slinear', 'slinear', 'slinear', 'quadratic']

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


def wrangle(
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
            sadj_res = sm.x13_arima_analysis(d[i])
            sadj_seasadj = sadj_res.seasadj
            d[i] = sadj_seasadj  # Ideally, use MYS-specific calendar effects

    # Take logs post-seasonal adjustment: now including capital stock
    list_col = list_col + ['k_stock15']
    list_col_ln = ['ln_' + i for i in list_col]
    if log_transform:
        for i, j in zip(list_col, list_col_ln):
            d[j] = np.log(d[i])

    # Take log-difference
    list_col_ln_diff = [i + '_diff' for i in list_col_ln]
    for i, j in zip(list_col_ln, list_col_ln_diff):
        d[j] = d[i] - d[i].shift(1)

    # filter trend
    list_col_ln_trend = [i + '_trend' for i in list_col_ln]
    list_col_ln_cycle = [i + '_cycle' for i in list_col_ln]
    if not filter_using_hamilton:
        for i, j, k in zip(list_col_ln, list_col_ln_trend, list_col_ln_cycle):
            cycle, trend = sm.filters.hpfilter(d[i], lamb=1600)
            d[j] = trend  # don't replace original with trend component
            d[k] = cycle  # don't replace original with cycle component
    elif filter_using_hamilton:
        for i, j, k in zip(list_col_ln, list_col_ln_trend, list_col_ln_cycle):
            cycle, trend = hamilton_filter(d[i], h=8, p=4)  # 2 years (2 for annual, 8 for quarter, 24 for monthly, ...)
            d[j] = trend  # don't replace original with trend component
            d[k] = cycle  # don't replace original with cycle component

    return d


def compute_ceilings(data,
                     levels_labels,
                     ref_level_label,
                     downturn_threshold,
                     bounds_timing_shift,
                     interpolation_method):

    # Deep copy
    d = data.copy()

    col_levels = levels_labels.copy()
    col_diff = [i + '_diff' for i in col_levels]
    col_trough = [i + '_trough' for i in col_levels]
    col_peak = [i + '_peak' for i in col_levels]
    col_epi = [i + '_epi' for i in col_levels]
    col_cepi = [i + '_cepi' for i in col_levels]  # ceiling episodes
    col_pace = [i + '_pace' for i in col_levels]
    col_ceiling = [i + '_ceiling' for i in col_levels]
    col_cpace = [i + '_cpace' for i in col_levels]

    ref_levels = ref_level_label
    ref_diff = ref_levels + '_diff'
    ref_trough = ref_levels + '_trough'
    ref_peak = ref_levels + '_peak'
    ref_epi = ref_levels + '_epi'
    ref_cepi = ref_levels + '_cepi'
    ref_pace = ref_levels + '_pace'
    ref_ceiling = ref_levels + '_ceiling'
    ref_cpace = ref_levels + '_cpace'

    # --------- Cycle over point est, and confidence bounds
    round = 1
    while round <= 2:

        print('Round: ' + str(round))

        # Peak-trough selection
        for diff, levels, trough, peak, epi, cepi, pace, ceiling, cpace \
                in \
                tqdm(zip(col_diff, col_levels,
                         col_trough, col_peak,
                         col_epi,
                         col_cepi,
                         col_pace,
                         col_ceiling,
                         col_cpace)):

            if round == 2:
                # Minus 1Q peak-peak timing
                trough = trough + '_lb'
                peak = peak + '_lb'
                epi = epi + '_lb'
                cepi = cepi + '_lb'
                pace = pace + '_lb'
                ceiling = ceiling + '_lb'
                cpace = cpace + '_lb'

            if round < 3:

                # Check unit roots in difference series
                # adf_pvalue = sm.adfuller(d[input].dropna())[1]
                # telsendmsg(conf=tel_config,
                #            msg=text_adf)

                # Check unit roots in second diff series
                # adf_pvalue = sm.adfuller((d[input] - d[input].shift(1)).dropna())[1]
                # telsendmsg(conf=tel_config,
                #            msg=text_adf)

                # Calculate standard deviation
                stdev = np.std(d[diff])
                factor_threshold = downturn_threshold  # adjust to include / exclude obvious episodes

                # if logdiff < 0, met with logdiff >= 0 in t+1, and abs(logdiff) >= stdev, then a trough
                d.loc[
                    (
                            (d[diff] < 0) &
                            (d[diff].shift(-1) >= 0) &
                            (np.abs(d[diff]) >= factor_threshold * stdev)
                    ),
                    trough
                ] = 1
                d[trough] = d[trough].fillna(0)

                # if logdiff > 0, met with logdiff < 0 in t+1, and trough within next 4 quarters (variable), then a peak
                d.loc[
                    (
                            (d[diff] >= 0) &
                            (d[diff].shift(-1) < 0) &
                            (
                                    (d[trough].shift(-1) == 1) |
                                    (d[trough].shift(-2) == 1) |
                                    (d[trough].shift(-3) == 1) |
                                    (d[trough].shift(-4) == 1)
                            )
                    )
                    , peak
                ] = 1
                d[peak] = d[peak].fillna(0)

                # Confidence bands
                if round == 2:
                    d[peak] = d[peak].shift(bounds_timing_shift)  # minus X
                    d[peak] = d[peak].fillna(0)

                # Episodes
                d[epi] = (d[trough] + d[peak]).cumsum()  # 0 = initial expansion, odd = gaps, even = expansion
                d.loc[((d[trough] == 1) | (d[peak] == 1)), epi] = d[epi] - 1  # exp start after trough, and end at peaks
                # d.loc[~((d[epi] % 2) == 0), epi] = np.nan
                # d[epi] = d[epi] / 2

                # Ceiling episodes, peak to peak
                d[cepi] = d[peak].cumsum()
                d.loc[d[peak] == 1, cepi] = d[cepi] - 1  # exp start after trough, and end at peaks

                # Calculate average episodic pace
                d[pace] = d.groupby(epi)[diff].transform('mean')
                tab = d.groupby(epi)[diff].agg('mean').reset_index()
                print(tab)
                # telsendmsg(conf=tel_config,
                #            msg=str(tab))

                # Compute 'ceiling'
                # Check if more than 1 expansion episodes
                single_exp = bool(d[epi].max() == 0)
                if not single_exp:

                    # Check if at least single or multiple peaks
                    single_peak = bool(d[peak].sum() == 1)
                    if single_peak:
                        d.loc[d.index == T_ub, peak] = 1  # if single peak, set last observation as a peak
                    # Check if the latest expansion has been at least 6-10 years (force a new peak)
                    ten_years = bool(d.loc[d[epi] == d[epi].max(), epi].count() >= 24)
                    if ten_years:
                        d.loc[d.index == T_ub, peak] = 1  # if X years since last peak, set last observation as a peak

                    # interpolate
                    d.loc[d[peak] == 1, ceiling] = d[levels]  # peaks as joints
                    d = d.reset_index()
                    d[ceiling] = d[ceiling].interpolate(method=interpolation_method)  # too sparse for cubic
                    d = d.set_index('quarter')

                    # end-point extrapolation
                    cepi_minusone = d[cepi].max() - 1
                    d['_x'] = d[ceiling] - d[ceiling].shift(1)
                    ceiling_minusone_avgdiff = (d.loc[d[cepi] == cepi_minusone, '_x']).mean()
                    del d['_x']
                    nrows_na = len(d.isna())
                    for r in tqdm(range(nrows_na)):
                        d.loc[d[ceiling].isna(), ceiling] = d[ceiling].shift(1) + ceiling_minusone_avgdiff

                    # start-point extrapolation
                    d['_x'] = d[ceiling] - d[ceiling].shift(1)
                    ceiling_one_avgdiff = (d.loc[d[cepi] == 1, '_x']).mean()
                    del d['_x']
                    nrows_na = len(d.isna())
                    for r in tqdm(range(nrows_na)):
                        d.loc[d[ceiling].isna(), ceiling] = d[ceiling].shift(-1) - ceiling_one_avgdiff  # reverse

                elif single_exp:  # then follow GDP peaks and troughs
                    d[peak] = d[ref_peak].copy()
                    d[cepi] = d[ref_cepi].copy()

                    # interpolate
                    d.loc[d[peak] == 1, ceiling] = d[levels]  # peaks as joints
                    d = d.reset_index()
                    d[ceiling] = d[ceiling].interpolate(method=interpolation_method)
                    d = d.set_index('quarter')

                    # end-point extrapolation
                    cepi_minusone = d[cepi].max() - 1
                    d['_x'] = d[ceiling] - d[ceiling].shift(1)
                    ceiling_minusone_avgdiff = (d.loc[d[cepi] == cepi_minusone, '_x']).mean()
                    del d['_x']
                    nrows_na = len(d.isna())
                    for r in tqdm(range(nrows_na)):
                        d.loc[d[ceiling].isna(), ceiling] = d[ceiling].shift(1) + ceiling_minusone_avgdiff

                    # start-point extrapolation
                    d['_x'] = d[ceiling] - d[ceiling].shift(1)
                    ceiling_one_avgdiff = (d.loc[d[cepi] == 1, '_x']).mean()
                    del d['_x']
                    nrows_na = len(d.isna())
                    for r in tqdm(range(nrows_na)):
                        d.loc[d[ceiling].isna(), ceiling] = d[ceiling].shift(-1) - ceiling_one_avgdiff  # reverse

        # Left right merge + bounds
        if round == 1:
            d_consol = d.copy()  # initial copy
        elif round == 2:
            d_consol = d_consol.combine_first(d)  # minusX bound

        # indicator to trigger computation of confidence bands
        round += 1

    # Output
    return d_consol


def output_gap(
        data,
        use_labour=True
):
    d = data.copy()

    d['gdp15_ceiling'] = np.exp(d['ln_gdp15_ceiling'])
    d['gdp15_ceiling_lb'] = np.exp(d['ln_gdp15_ceiling_lb'])

    d['output_gap'] = 100 * (d['gdp15'] / d['gdp15_ceiling'] - 1)  # % PO
    d['output_gap_lb'] = 100 * (d['gdp15'] / d['gdp15_ceiling_lb'] - 1) # % PO

    # trim data frame
    list_col_keep = ['output_gap', 'output_gap_lb',
                     'gdp15_ceiling', 'gdp15_ceiling_lb',
                     'gdp15']
    d = d[list_col_keep]

    return d


# II.A --- Wrangling
# Base data
df_raw = pd.read_csv('testdata.txt', sep='|')
df_full = wrangle(
    data=df_raw,
    trim_start=T_lb,
    trim_end=list_T_ub[-1],
    seasonal_adj=True,
    log_transform=True,
    filter_using_hamilton=False
)
df_full = df_full[['gdp15', 'ln_gdp15', 'ln_gdp15_diff']]  # Only GDP is required


# ------------ COMPUTING VINTAGES ------------

round = 1
for T_ub, interpolate_method, threshold in zip(list_T_ub, list_interpolate_method, list_threshold):

    # II.B --- Generate vintage
    df = df_full[df_full.index <= T_ub]

    # III --- Compute ceiling

    list_ln = ['ln_gdp15']
    df = compute_ceilings(
        data=df,
        levels_labels=list_ln,
        ref_level_label='ln_gdp15',
        downturn_threshold=threshold,  # 0.65 to 0.8
        bounds_timing_shift=-1,
        interpolation_method=interpolate_method
    )

    # IV --- Compute output gap

    df_og = output_gap(data=df, use_labour=False)
    df_og = pd.DataFrame(df_og).rename(columns={'output_gap': T_ub})

    # V --- Consolidate vintages
    if round == 1:
        df_final = df_og.copy()
    elif round > 1:
        df_final = pd.concat([df_final, df_og], axis=1)
    round += 1

# ------------ PLOTTING VINTAGES ------------

# VI --- Plotting


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
    fig.write_image('Output/PluckingPO_Vintage_' + output_suffix + '.png', height=768, width=1366)
    fig.write_html('Output/PluckingPO_Vintage_' + output_suffix + '.html')
    return fig


df_final['ref'] = 0

fig_vintages = plot_linechart(
    data=df_final,
    cols=list_T_ub + ['ref'],
    nice_names=list_T_ub + ['Reference (Y=0)'],
    colours=list_colours + ['black'],
    dash_styles=list_dash_styles + ['solid'],
    y_axis_title='% Potential Output',
    main_title='Vintages of Plucking Output Gap Estimates',
    output_suffix='OutputGap'
)
telsendimg(conf=tel_config,
           path='Output/PluckingPO_Vintage_OutputGap.png',
           cap='Vintages of Plucking Output Gap Estimates')

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
df_rev = df_rev[df_rev.index <= list_T_ub[-2]]

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
    fig.write_image('Output/PluckingPO_Vintage_' + output_suffix + '.png', height=768, width=1366)
    fig.write_html('Output/PluckingPO_Vintage_' + output_suffix + '.html')
    return fig


fig_rev = plot_areachart(
    data=df_rev,
    cols=['rev_consol'],
    nice_names=['Revisions'],
    colours=['lightcoral'],
    y_axis_title='Percentage Points (% Potential Output)',
    main_title='Revisions in Plucking Output Gap Across Consecutive Vintages',
    show_legend=False,
    ymin=-5,
    ymax=5,
    output_suffix='OutputGap_Revisions'
)
telsendimg(conf=tel_config,
           path='Output/PluckingPO_Vintage_OutputGap_Revisions.png',
           cap='Revisions in Plucking Output Gap Across Consecutive Vintages')


# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')

