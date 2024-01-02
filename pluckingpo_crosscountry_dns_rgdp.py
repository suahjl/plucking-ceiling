# %%
from ceic_api_client.pyceic import Ceic
import pandas as pd
import numpy as np
from datetime import date, timedelta
import plotly.graph_objects as go
import plotly.express as px
import re
from tqdm import tqdm
import telegram_send
import time
from dotenv import load_dotenv
import os
import ast

time_start = time.time()

# %%
# 0 --- Main settings
load_dotenv()
tel_config = os.getenv('TEL_CONFIG')
seriesid_labels = pd.read_csv(
    'seriesids_crosscountry_rgdp.txt',
    dtype='str'
)

seriesid_labels = seriesid_labels.replace(
    {'\([^()]*\)': ''}, regex=True)  # remove everything inside parentheses
seriesid_labels = seriesid_labels.replace(
    {' ': ''}, regex=True)  # remove spaces

tminus = date(1947, 1, 1)
tfin = date(2022, 12, 31)
col_arrangement = ['country', 'quarter'] + list(seriesid_labels.columns)

load_dotenv()
Ceic.login(os.getenv('CEIC_USERNAME'), os.getenv('CEIC_PASSWORD'))

# %%
# I --- Functions


def ceic2pandas(input, t_start, t_end):  # input should be a list of CEIC Series IDs
    for m in range(len(input)):
        try:
            # brute force remove all np.nans from series ID list
            input.remove(np.nan)
        except:
            print('no more np.nan')
    k = 1
    for i in tqdm(input):
        series_result = Ceic.series(
            i, start_date=t_start, end_date=t_end)  # retrieves ceicseries
        y = series_result.data
        name = y[0].metadata.country.name  # retrieves country name
        longname = y[0].metadata.name  # retrieves CEIC series name
        # this is a list of 1 dictionary,
        time_points_dict = dict((tp._date, tp.value)
                                for tp in y[0].time_points)
        # convert into pandas series indexed to timepoints
        series = pd.Series(time_points_dict)
        if k == 1:
            frame_consol = pd.DataFrame(series)
            frame_consol['country'] = name
            if re.search('Hong Kong', longname):
                frame_consol['country'] = 'Hong Kong'
            if re.search('Macau', longname):
                frame_consol['country'] = 'Macau'
            frame_consol = frame_consol.reset_index(
                drop=False).rename(columns={'index': 'date'})
        elif k > 1:
            frame = pd.DataFrame(series)
            frame['country'] = name
            if re.search('Hong Kong', longname):
                frame['country'] = 'Hong Kong'
            if re.search('Macau', longname):
                frame['country'] = 'Macau'
            frame = frame.reset_index(drop=False).rename(
                columns={'index': 'date'})
            frame_consol = pd.concat(
                [frame_consol, frame], axis=0)  # top-bottom concat
        elif k < 1:
            raise NotImplementedError
        k += 1
    frame_consol = frame_consol.reset_index(
        drop=True)  # avoid repeating indices
    return frame_consol


def telsendimg(path='', conf='', cap=''):
    with open(path, 'rb') as f:
        telegram_send.send(conf=conf,
                           images=[f],
                           captions=[cap])


def telsendfiles(path='', conf='', cap=''):
    with open(path, 'rb') as f:
        telegram_send.send(conf=conf,
                           files=[f],
                           captions=[cap])


def telsendmsg(conf='', msg=''):
    telegram_send.send(conf=conf,
                       messages=[msg])


# %%
# II --- Generate panel data
# Download and concat from CEIC
x = seriesid_labels.columns[0]
# generate list of series IDs for the variable of interest
input = list(seriesid_labels[x])
df_raw = ceic2pandas(input=input, t_start=tminus,
                     t_end=tfin)  # single variable panel
df_raw = df_raw.rename(columns={0: x})  # rename default column name
# Date format
df_raw['date'] = pd.to_datetime(df_raw['date']).dt.to_period('Q')
df_raw = df_raw.rename(columns={'date': 'quarter'})
# Chronological order by country
df_raw = df_raw.sort_values(by=['country', 'quarter'], ascending=[True, True])
df_raw = df_raw.reset_index(drop=True)
# Tier 1 problematic countries (Either no cycles, or problematic start dates, e.g., Cold War transition)
df_raw = df_raw[~(df_raw['country'] == 'Argentina')]
df_raw = df_raw[~(df_raw['country'] == 'Vietnam')]
df_raw = df_raw[~(df_raw['country'] == 'Bolivia')]
df_raw = df_raw[~(df_raw['country'] == 'Denmark')]
df_raw = df_raw[~(df_raw['country'] == 'South Africa')]
df_raw = df_raw[~((df_raw['country'] == 'Bulgaria')
                  & (df_raw['quarter'] < '2000Q1'))]
# Tier 2 problematic countries (context, e.g., war, Cold War transition, tiny countries, dodgy stats)
df_raw = df_raw[~(df_raw['country'] == 'Bosnia and Herzegovina')]
# df_raw = df_raw[~(df_raw['country'] == 'Luxembourg')]
# df_raw = df_raw[~(df_raw['country'] == 'Albania')]
# df_raw = df_raw[~(df_raw['country'] == 'Armenia')]
# df_raw = df_raw[~(df_raw['country'] == 'Georgia')]
# df_raw = df_raw[~(df_raw['country'] == 'Serbia')]
# df_raw = df_raw[~(df_raw['country'] == 'European Union')]
df_raw = df_raw[~(df_raw['country'] == 'Myanmar')]
df_raw = df_raw[~(df_raw['country'] ==
                  'State of Palestine (West Bank and Gaza)')]
# Tier 3 problematic countries (outliers)
df_raw = df_raw[~(df_raw['country'] == 'Japan')]
df_raw = df_raw[~(df_raw['country'] == 'Panama')]
# Generate one more column for levels (100-indexed)
for country in tqdm(list(df_raw['country'].unique())):
    df_raw.loc[
        (
            (df_raw["country"] == country) &
            (df_raw["quarter"] == df_raw.loc[df_raw["country"] == country, "quarter"].min())
        ),
        "rgdpsa"] = 100
    time_count = 0
    for quarter in list(df_raw.loc[df_raw['country'] == country, 'quarter'].unique()):
        if time_count == 0:
            pass
        else:
            df_raw.loc[
                (
                    (df_raw["country"] == country) &
                    (df_raw["quarter"] == quarter) & 
                    (df_raw["rgdpsa"].isna())
                ),
                "rgdpsa"] = df_raw["rgdpsa"].shift(1) * (1 + (df_raw['rgdpsa_qoq'] / 100))
        time_count += 1


# %%
# III --- Compute peak-troughs paces, and Exp_{t} ->> Con_{t+1} and Con_{t} ->> Exp_{t+1}


def compute_pluck_for_all_countries(data, entities_label, rows_per_epi):
    # Deep copy
    df_copy = df_raw.copy()

    # Blank data frame
    d_consol = pd.DataFrame(columns=list(
        df_copy.columns) + ['peak', 'trough', 'epi', 'pace'])
    d_expcon_consol = pd.DataFrame(
        columns=[entities_label, 'expansion_pace', 'subsequent_contraction_pace'])
    d_conexp_consol = pd.DataFrame(
        columns=[entities_label, 'contraction_pace', 'subsequent_expansion_pace'])

    # List of countries
    list_entities = list(data[entities_label].unique())

    # Episodes by country
    for entity in tqdm(list_entities):

        # Preliminaries
        d = df_copy[df_copy[entities_label] == entity].copy()

        # Base list of peaks and troughs
        list_peaks = []
        list_troughs = []

        # Store list of quarters
        list_time = [str(i) for i in list(d['quarter'])]
        list_indices = [i for i in list(d.index)]

        # For convenience, this is not a rate variable
        col_is_rate = False

        # Compute downturn threshold
        # threshold_for_this_col = np.std(d['rgdpsa_qoq']) * downturn_threshold
        threshold_for_this_col = np.std(d['rgdpsa_qoq']) * 0.1

        # Initialise parameters
        t_cp = 0  # peak candidate
        t_ct = 0  # trough candidate
        t_next = t_cp + 1  # initial time stamp
        just_found_peak = False
        just_found_trough = False
        stuck_in_step_one = True
        stuck_in_step_two = True
        stuck_in_step_five = True
        stuck_in_step_six = True

        # Define all requisite steps as functions
        def step_one(just_found_trough: bool,
                     t_cp, t_ct, t_next):
            if just_found_trough:
                t_cp = t_ct + 1
                t_next = t_cp + 1
            elif not just_found_trough:
                t_cp = t_cp + 1
                t_next = t_next + 1
            return t_cp, t_next

        def step_two(df: pd.DataFrame, t_cp, t_next):
            go_back = (
                df.loc[df.index == t_cp, "rgdpsa"].values[0]
                <
                df.loc[df.index == t_next, "rgdpsa"].values[0]
            )
            return go_back

        def step_three(df: pd.DataFrame, is_rate, t_next):
            # only sensible for rates
            if is_rate:
                go_back = (
                    df.loc[df.index == t_cp, "rgdpsa"].values[0]
                    <
                    df.loc[df.index == t_next, "rgdpsa"].values[0] +
                    threshold_for_this_col
                )
                t_next = t_next + 1  # without changing t_cp
            # adapted to non-rate data
            elif not is_rate:
                go_back = (
                    df.loc[df.index == t_next, "rgdpsa_qoq"].values[0]
                    <
                    0 + threshold_for_this_col
                )
                t_next = t_next + 1  # without changing t_cp
            return go_back, t_next

        def step_four(t_cp, list_peaks):
            list_peaks = list_peaks + [list_time[t_cp]]
            just_found_peak = True
            return list_peaks, just_found_peak

        def step_five(
                just_found_peak: bool,
                t_cp, t_ct, t_next
        ):
            if just_found_peak:
                t_ct = t_cp + 1
                t_next = t_ct + 1
            elif not just_found_peak:
                t_ct = t_ct + 1
                t_next = t_next + 1
            return t_ct, t_next

        def step_six(df: pd.DataFrame, t_ct, t_next):
            go_back = (
                df.loc[df.index == t_ct, "rgdpsa"].values[0]
                >
                df.loc[df.index == t_next, "rgdpsa"].values[0]
            )
            return go_back

        def step_seven(df: pd.DataFrame, is_rate, t_next):
            # only sensible for rates
            if is_rate:
                go_back = (
                    df.loc[df.index == t_ct, "rgdpsa"].values[0]
                    >
                    df.loc[df.index == t_next, "rgdpsa"].values[0] -
                    threshold_for_this_col
                )
                t_next = t_next + 1  # without changing t_cp
            # adapted to non-rate data
            elif not is_rate:
                go_back = (
                    df.loc[df.index == t_next, "rgdpsa_qoq"].values[0]
                    >
                    0 - threshold_for_this_col
                )
                t_next = t_next + 1  # without changing t_cp
            return go_back, t_next

        def step_eight(t_ct, list_troughs):
            list_troughs = list_troughs + [list_time[t_ct]]
            just_found_trough = True
            return list_troughs, just_found_trough

        while t_next <= list_indices[-1]:
            try:
                # FIND PEAK
                # step one and two
                while stuck_in_step_one:
                    # print('Step 1: t_next = ' + list_time[t_next] + ' for ' + col_level)
                    t_cp, t_next = step_one(just_found_trough=just_found_trough, t_cp=t_cp, t_ct=t_ct,
                                            t_next=t_next)
                    just_found_trough = False  # only allow just_found_trough to be true once per loop
                    stuck_in_step_one = step_two(
                        df=d, t_cp=t_cp, t_next=t_next)
                stuck_in_step_one = True  # reset so loop will run again
                # step three
                while stuck_in_step_two:
                    # print('Step 2-3: t_next = ' + list_time[t_next] + ' for ' + col_level)
                    stuck_in_step_two, t_next = step_three(df=d, is_rate=col_is_rate,
                                                           t_next=t_next)  # if true, skips next line
                    restuck_in_step_one = step_two(
                        df=d, t_cp=t_cp, t_next=t_next)
                    while restuck_in_step_one:  # if step 3 is executed, but then fails step 2, so back to step 1
                        # print('Back to step 1-2: t_next = ' + list_time[t_next] + ' for ' + col_level)
                        t_cp, t_next = step_one(just_found_trough=just_found_trough, t_cp=t_cp, t_ct=t_ct,
                                                t_next=t_next)
                        restuck_in_step_one = step_two(
                            df=d, t_cp=t_cp, t_next=t_next)
                stuck_in_step_two = True  # reset so loop will run again
                # step four
                # print('Step 4: t_cp = ' + list_time[t_cp] + ' for ' + col_level)
                list_peaks, just_found_peak = step_four(
                    t_cp=t_cp, list_peaks=list_peaks)  # we have a peak
                just_found_peak = True  # voila

                # FIND TROUGH
                # step five and six (equivalent to one and two)
                while stuck_in_step_five:
                    # print('Step 5: t_next = ' + list_time[t_next] + ' for ' + col_level)
                    t_ct, t_next = step_five(
                        just_found_peak=just_found_peak, t_cp=t_cp, t_ct=t_ct, t_next=t_next)
                    just_found_peak = False  # only allow just_found_peak to be true once per loop
                    stuck_in_step_five = step_six(
                        df=d, t_ct=t_ct, t_next=t_next)
                stuck_in_step_five = True  # reset so loop will run again
                # step seven (equivalent to three)
                while stuck_in_step_six:
                    # print('Step 6-7: t_next = ' + list_time[t_next] + ' for ' + col_level)
                    stuck_in_step_six, t_next = step_seven(df=d, is_rate=col_is_rate,
                                                           t_next=t_next)  # if true, skips next line
                    restuck_in_step_five = step_six(
                        df=d, t_ct=t_ct, t_next=t_next)
                    while restuck_in_step_five:
                        # print('Back to step 5-6: t_next = ' + list_time[t_next] + ' for ' + col_level)
                        t_ct, t_next = step_five(just_found_peak=just_found_peak, t_cp=t_cp, t_ct=t_ct,
                                                 t_next=t_next)
                        restuck_in_step_five = step_six(
                            df=d, t_ct=t_ct, t_next=t_next)
                stuck_in_step_six = True  # reset so loop will run again
                # step eight (equivalent to four)
                # print('Step 8: t_ct = ' + list_time[t_ct] + ' for ' + col_level)
                list_troughs, just_found_trough = step_eight(t_ct=t_ct,
                                                             list_troughs=list_troughs)  # we have a trough
            except:
                pass

        # Check
        print('Peaks in ' + "rgdpsa" + ': ' + ', '.join(list_peaks))
        print('Troughs in ' + "rgdpsa" + ': ' + ', '.join(list_troughs))

        # Add columns indicating peaks and troughs
        d.loc[d['quarter'].isin(list_peaks), "peak"] = 1
        d["peak"] = d["peak"].fillna(0)
        d.loc[d['quarter'].isin(list_troughs), "trough"] = 1
        d["trough"] = d["trough"].fillna(0)

        # Episodes
        # 0 = initial expansion, odd = gaps, even = expansion
        d['epi'] = (d['trough'] + d['peak']).cumsum()
        # exp start after trough, and end at peaks
        d.loc[((d['trough'] == 1) | (d['peak'] == 1)), 'epi'] = d['epi'] - 1
        # d.loc[~((d['epi'] % 2) == 0), 'epi'] = np.nan
        # d['epi'] = d['epi'] / 2

        # Calculate average episodic pace
        d['pace'] = d.groupby('epi')['rgdpsa_qoq'].transform('mean')

        # First n rows of each episode
        if rows_per_epi is not None:
            d = d.groupby('epi').head(rows_per_epi).sort_values(
                by='epi').reset_index(drop=False)
        elif rows_per_epi is None:
            pass

        # Concat into dummy dataframe
        d_consol = pd.concat([d_consol, d], axis=0)

        # Exp --> Con and Con --> Exp
        if d['epi'].max() == 0:
            pass
        else:
            d = d[['pace', 'epi']]
            d = d.groupby('epi').agg('mean')

            expansions = d.iloc[::2].reset_index(drop=True)
            subsequent_contractions = d.iloc[1::2].reset_index(drop=True)
            d_expcon = pd.concat(
                [expansions, subsequent_contractions], axis=1).dropna()
            d_expcon.columns = ['expansion_pace',
                                'subsequent_contraction_pace']
            d_expcon['subsequent_contraction_pace'] = d_expcon['subsequent_contraction_pace'] * \
                (-1)
            d_expcon[entities_label] = entity
            d_expcon_consol = pd.concat([d_expcon_consol, d_expcon], axis=0)

            contractions = d.iloc[1::2].reset_index(drop=True)
            subsequent_expansions = d.iloc[2::2].reset_index(drop=True)
            d_conexp = pd.concat(
                [contractions, subsequent_expansions], axis=1).dropna()
            d_conexp.columns = ['contraction_pace',
                                'subsequent_expansion_pace']
            d_conexp['contraction_pace'] = d_conexp['contraction_pace'] * (-1)
            d_conexp[entities_label] = entity
            d_conexp_consol = pd.concat([d_conexp_consol, d_conexp], axis=0)

    # Output
    return d_consol, d_expcon_consol, d_conexp_consol


df, df_expcon, df_conexp = compute_pluck_for_all_countries(
    data=df_raw, entities_label='country', rows_per_epi=None)  # 24
df_expcon_avg = df_expcon.groupby('country').agg('mean')
df_conexp_avg = df_conexp.groupby('country').agg('mean')


# %%
# IV --- Plot plucking properties


def plot_scatter(data, y_col, x_col, colour, chart_title, output_suffix):
    fig = px.scatter(data, x=x_col, y=y_col, trendline='ols',
                     color_discrete_sequence=[colour])
    fig.update_traces(marker=dict(size=16),
                      selector=dict(mode='markers'))
    fig.update_layout(title=chart_title,
                      plot_bgcolor='white',
                      hovermode='x',
                      font=dict(size=20, color='black'))
    fig.write_image('Output/PluckingPO_CrossCountry_DNS_Scatter_' +
                    output_suffix + '.png', height=768, width=1366)
    fig.write_html(
        'Output/PluckingPO_CrossCountry_DNS_Scatter_' + output_suffix + '.html')
    return fig


# Exp --> Con
fig_expcon = plot_scatter(data=df_expcon,
                          y_col='subsequent_contraction_pace',
                          x_col='expansion_pace',
                          colour='black',
                          chart_title='Real GDP: Expansion Pace vs. Subsequent Contraction Pace (%QoQ SA)',
                          output_suffix='RGDP_ExpCon')
telsendimg(conf=tel_config,
           path='Output/PluckingPO_CrossCountry_DNS_Scatter_RGDP_ExpCon' + '.png')
# Exp --> Con (avg)
fig_expcon_avg = plot_scatter(data=df_expcon_avg,
                              y_col='subsequent_contraction_pace',
                              x_col='expansion_pace',
                              colour='black',
                              chart_title='Real GDP: Average Expansion Pace vs. Subsequent Contraction Pace (%QoQ SA)',
                              output_suffix='RGDP_ExpCon_Avg')
telsendimg(conf=tel_config,
           path='Output/PluckingPO_CrossCountry_DNS_Scatter_RGDP_ExpCon_Avg' + '.png')

# Con --> Exp
fig_conexp = plot_scatter(data=df_conexp,
                          y_col='subsequent_expansion_pace',
                          x_col='contraction_pace',
                          colour='crimson',
                          chart_title='Real GDP: Contraction Pace vs. Subsequent Expansion Pace (%QoQ SA)',
                          output_suffix='RGDP_ConExp')
telsendimg(conf=tel_config,
           path='Output/PluckingPO_CrossCountry_DNS_Scatter_RGDP_ConExp' + '.png')
# Con --> Exp (Avg)
fig_conexp_avg = plot_scatter(data=df_conexp_avg,
                              y_col='subsequent_expansion_pace',
                              x_col='contraction_pace',
                              colour='crimson',
                              chart_title='Real GDP: Average Contraction Pace vs. Subsequent Expansion Pace (%QoQ SA)',
                              output_suffix='RGDP_ConExp_Avg')
telsendimg(conf=tel_config,
           path='Output/PluckingPO_CrossCountry_DNS_Scatter_RGDP_ConExp_Avg' + '.png')

# List of countries
telsendmsg(conf=tel_config,
           msg=str(df['country'].nunique()) + ' Countries Included: \n' +
           str(', '.join(list(df['country'].unique()))))


# %%
# End
print('\n----- Ran in ' +
      "{:.0f}".format(time.time() - time_start) + ' seconds -----')
