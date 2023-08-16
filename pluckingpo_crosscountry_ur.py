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

# 0 --- Main settings
load_dotenv()
tel_config = os.getenv('TEL_CONFIG')
seriesid_labels = pd.read_csv(
    'seriesids_crosscountry_ur.txt',
    dtype='str'
)

seriesid_labels = seriesid_labels.replace({'\([^()]*\)': ''}, regex=True)  # remove everything inside parentheses
seriesid_labels = seriesid_labels.replace({' ': ''}, regex=True)  # remove spaces

tminus = date(1948, 1, 1)
tfin = date(2022, 9, 30)
col_arrangement = ['country', 'month'] + list(seriesid_labels.columns)

Ceic.login(os.getenv('CEIC_USERNAME'), os.getenv('CEIC_PASSWORD'))

# I --- Functions


def ceic2pandas(input, t_start, t_end):  # input should be a list of CEIC Series IDs
    for m in range(len(input)):
        try: input.remove(np.nan)  # brute force remove all np.nans from series ID list
        except: print('no more np.nan')
    k = 1
    for i in tqdm(input):
        series_result = Ceic.series(i, start_date=t_start, end_date=t_end)  # retrieves ceicseries
        y = series_result.data
        name = y[0].metadata.country.name  # retrieves country name
        longname = y[0].metadata.name # retrieves CEIC series name
        time_points_dict = dict((tp._date, tp.value) for tp in y[0].time_points)  # this is a list of 1 dictionary,
        series = pd.Series(time_points_dict)  # convert into pandas series indexed to timepoints
        if k == 1:
            frame_consol = pd.DataFrame(series)
            frame_consol['country'] = name
            if re.search('Hong Kong', longname):
                frame_consol['country'] = 'Hong Kong'
            if re.search('Macau', longname):
                frame_consol['country'] = 'Macau'
            frame_consol = frame_consol.reset_index(drop=False).rename(columns={'index': 'date'})
        elif k > 1:
            frame = pd.DataFrame(series)
            frame['country'] = name
            if re.search('Hong Kong', longname):
                frame['country'] = 'Hong Kong'
            if re.search('Macau', longname):
                frame['country'] = 'Macau'
            frame = frame.reset_index(drop=False).rename(columns={'index': 'date'})
            frame_consol = pd.concat([frame_consol, frame], axis=0) # top-bottom concat
        elif k < 1:
            raise NotImplementedError
        k += 1
    frame_consol = frame_consol.reset_index(drop=True)  # avoid repeating indices
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


# II --- Generate panel data
# Download and concat from CEIC
x = seriesid_labels.columns[0]
input = list(seriesid_labels[x])  # generate list of series IDs for the variable of interest
df_raw = ceic2pandas(input=input, t_start=tminus, t_end=tfin)  # single variable panel
df_raw = df_raw.rename(columns={0: x})  # rename default column name
# Date format
df_raw['date'] = pd.to_datetime(df_raw['date']).dt.to_period('M')  # monthly
df_raw = df_raw.rename(columns={'date': 'month'})
# Chronological order by country
df_raw = df_raw.sort_values(by=['country', 'month'], ascending=[True, True])
df_raw = df_raw.reset_index(drop=True)
# Generate change in unemployment rate (SA)
df_raw['ursa_qoq'] = df_raw['ursa'] - df_raw.groupby('country')['ursa'].shift(1)
# Drop NA
df_raw = df_raw.dropna(subset='ursa_qoq')
# Tier 1 problematic countries (Either no cycles, or problematic start dates, e.g., Cold War transition)
df_raw = df_raw[~(df_raw['country'] == 'Argentina')]
df_raw = df_raw[~(df_raw['country'] == 'Vietnam')]
df_raw = df_raw[~(df_raw['country'] == 'Bolivia')]
df_raw = df_raw[~(df_raw['country'] == 'Denmark')]
df_raw = df_raw[~(df_raw['country'] == 'South Africa')]
df_raw = df_raw[~((df_raw['country'] == 'Bulgaria') & (df_raw['month'] < '2000-01'))]
# Tier 2 problematic countries (context, e.g., war, Cold War transition, tiny countries, dodgy stats)
df_raw = df_raw[~(df_raw['country'] == 'Bosnia and Herzegovina')]
# df_raw = df_raw[~(df_raw['country'] == 'Luxembourg')]
# df_raw = df_raw[~(df_raw['country'] == 'Albania')]
# df_raw = df_raw[~(df_raw['country'] == 'Armenia')]
# df_raw = df_raw[~(df_raw['country'] == 'Georgia')]
# df_raw = df_raw[~(df_raw['country'] == 'Serbia')]
# df_raw = df_raw[~(df_raw['country'] == 'European Union')]
df_raw = df_raw[~(df_raw['country'] == 'Myanmar')]
df_raw = df_raw[~(df_raw['country'] == 'State of Palestine (West Bank and Gaza)')]
# Tier 3 problematic countries (outliers)
df_raw = df_raw[~(df_raw['country'] == 'Japan')]
df_raw = df_raw[~(df_raw['country'] == 'Panama')]

# III --- Compute peak-troughs paces, and Exp_{t} ->> Con_{t+1} and Con_{t} ->> Exp_{t+1}; REVERSED DIRECTION


def compute_pluck(data, entities_label, rows_per_epi):

    # Deep copy
    df_copy = df_raw.copy()

    # Blank data frame
    d_consol = pd.DataFrame(columns=list(df_copy.columns) + ['peak', 'trough', 'epi', 'pace'])
    d_expcon_consol = pd.DataFrame(columns=[entities_label, 'expansion_pace', 'subsequent_contraction_pace'])
    d_conexp_consol = pd.DataFrame(columns=[entities_label, 'contraction_pace', 'subsequent_expansion_pace'])

    # List of countries
    list_entities = list(data[entities_label].unique())

    # Episodes by country
    for entity in tqdm(list_entities):

        # Preliminaries
        d = df_copy[df_copy[entities_label] == entity]

        # Calculate standard deviation
        stdev = np.std(d['ursa_qoq'])
        factor1 = 1  # adjust to include / exclude obvious episodes

        # if qoqsa > 0, met with qoqsa <= 0 in next 4Q, qoqsa >= stdev, AND NO TROUGHS IN PAST 4Q, then a trough
        d.loc[
            (
                    (d['ursa_qoq'] > 0) &
                    (d['ursa_qoq'].shift(-1) <= 0) &
                    (d['ursa_qoq'].shift(-2) <= 0) &
                    (d['ursa_qoq'].shift(-3) <= 0) &
                    (d['ursa_qoq'].shift(-4) <= 0)
                    &
                    (d['ursa_qoq'] >= factor1 * stdev)
            ),
            'trough'
        ] = 1
        d.loc[
            (
                    (d['trough'] == 1)
                    &
                    (
                            (d['trough'].shift(1) == 1) |
                            (d['trough'].shift(2) == 1) |
                            (d['trough'].shift(3) == 1) |
                            (d['trough'].shift(4) == 1) |
                            (d['trough'].shift(5) == 1)
                    )

            ),
            'trough'
        ] = 0
        d['trough'] = d['trough'].fillna(0)

        # if qoqsa < 0, met with qoqsa > 0 in t+1, trough within next 4Q, and no peak in past 4Q, then a peak
        d.loc[
            (
                    (d['ursa_qoq'] <= 0) &
                    (d['ursa_qoq'].shift(-1) > 0)
                    &
                    (
                            (d['trough'].shift(-1) == 1) |
                            (d['trough'].shift(-2) == 1) |
                            (d['trough'].shift(-3) == 1) |
                            (d['trough'].shift(-4) == 1)
                    )
            ),
            'peak'
        ] = 1
        d.loc[
            (
                    (d['peak'] == 1)
                    &
                    (
                            (d['peak'].shift(1) == 1) |
                            (d['peak'].shift(2) == 1) |
                            (d['peak'].shift(3) == 1) |
                            (d['peak'].shift(4) == 1)
                    )

            ),
            'peak'
        ] = 0
        d['peak'] = d['peak'].fillna(0)

        # Episodes
        d['epi'] = (d['trough'] + d['peak']).cumsum()  # 0 = initial expansion, odd = gaps, even = expansion
        d.loc[((d['trough'] == 1) | (d['peak'] == 1)), 'epi'] = d['epi'] - 1  # exp start after trough, and end at peaks
        # d.loc[~((d['epi'] % 2) == 0), 'epi'] = np.nan
        # d['epi'] = d['epi'] / 2

        # Calculate average episodic pace
        d['pace'] = d.groupby('epi')['ursa_qoq'].transform('mean')

        # First n rows of each episode
        if rows_per_epi is not None:
            d = d.groupby('epi').head(rows_per_epi).sort_values(by='epi').reset_index(drop=False)
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
            d_expcon = pd.concat([expansions, subsequent_contractions], axis=1).dropna()
            d_expcon.columns = ['expansion_pace', 'subsequent_contraction_pace']
            # d_expcon['expansion_pace'] = d_expcon['expansion_pace'] * (-1)
            d_expcon[entities_label] = entity
            d_expcon_consol = pd.concat([d_expcon_consol, d_expcon], axis=0)

            contractions = d.iloc[1::2].reset_index(drop=True)
            subsequent_expansions = d.iloc[2::2].reset_index(drop=True)
            d_conexp = pd.concat([contractions, subsequent_expansions], axis=1).dropna()
            d_conexp.columns = ['contraction_pace', 'subsequent_expansion_pace']
            # d_conexp['subsequent_expansion_pace'] = d_conexp['subsequent_expansion_pace'] * (-1)
            d_conexp[entities_label] = entity
            d_conexp_consol = pd.concat([d_conexp_consol, d_conexp], axis=0)

    # Output
    return d_consol, d_expcon_consol, d_conexp_consol

df, df_expcon, df_conexp = compute_pluck(data=df_raw, entities_label='country', rows_per_epi=None)
df_expcon_avg = df_expcon.groupby('country').agg('mean')
df_conexp_avg = df_conexp.groupby('country').agg('mean')

# IV --- Plot plucking properties


def plot_scatter(data, y_col, x_col, colour, chart_title, output_suffix):
    fig = px.scatter(data, x=x_col, y=y_col, trendline='ols', color_discrete_sequence=[colour])
    fig.update_traces(marker=dict(size=20),
                      selector=dict(mode='markers'))
    fig.update_layout(title=chart_title,
                      plot_bgcolor='white',
                      hovermode='x',
                      font=dict(size=20, color='black'))
    fig.write_image('Output/PluckingPO_CrossCountry_Scatter_' + output_suffix + '.png', height=768, width=1366)
    fig.write_html('Output/PluckingPO_CrossCountry_Scatter_' + output_suffix + '.html')
    return fig


# Exp --> Con
fig_expcon = plot_scatter(data=df_expcon,
                          y_col='subsequent_contraction_pace',
                          x_col='expansion_pace',
                          colour='black',
                          chart_title='Unemployment Rate: Expansion Pace vs. Subsequent Contraction Pace (QoQ SA)',
                          output_suffix='UR_ExpCon')
telsendimg(conf=tel_config,
           path='Output/PluckingPO_CrossCountry_Scatter_UR_ExpCon' + '.png')
# Exp --> Con (avg)
fig_expcon_avg = plot_scatter(data=df_expcon_avg,
                              y_col='subsequent_contraction_pace',
                              x_col='expansion_pace',
                              colour='black',
                              chart_title='Unemployment Rate: Average Expansion Pace vs. Subsequent Contraction Pace (%QoQ SA)',
                              output_suffix='UR_ExpCon_Avg')
telsendimg(conf=tel_config,
           path='Output/PluckingPO_CrossCountry_Scatter_UR_ExpCon_Avg' + '.png')

# Con --> Exp
fig_conexp = plot_scatter(data=df_conexp,
                          y_col='subsequent_expansion_pace',
                          x_col='contraction_pace',
                          colour='crimson',
                          chart_title='Unemployment Rate: Contraction Pace vs. Subsequent Expansion Pace (%QoQ SA)',
                          output_suffix='UR_ConExp')
telsendimg(conf=tel_config,
           path='Output/PluckingPO_CrossCountry_Scatter_UR_ConExp' + '.png')
# Con --> Exp (Avg)
fig_conexp_avg = plot_scatter(data=df_conexp_avg,
                              y_col='subsequent_expansion_pace',
                              x_col='contraction_pace',
                              colour='crimson',
                              chart_title='Unemployment Rate: Average Contraction Pace vs. Subsequent Expansion Pace (%QoQ SA)',
                              output_suffix='UR_ConExp_Avg')
telsendimg(conf=tel_config,
           path='Output/PluckingPO_CrossCountry_Scatter_UR_ConExp_Avg' + '.png')

# List of countries
telsendmsg(conf=tel_config,
           msg=str(df['country'].nunique()) + ' Countries Included: \n' +
               str(', '.join(list(df['country'].unique()))))

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
