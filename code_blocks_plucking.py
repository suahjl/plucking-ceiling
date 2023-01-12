import pandas as pd
import numpy as np
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

# Load data
df = pd.read_csv('...')

# --- Seasonally adjusting data

list_col = ['gdp15', 'labour', 'employment']
seasonal_adj = True
if seasonal_adj:
    for i in list_col:
        sadj_res = sm.x13_arima_analysis(d[i])
        sadj_seasadj = sadj_res.seasadj
        df[i] = sadj_seasadj  # Ideally, use MYS-specific calendar effects

# Take logs post-seasonal adjustment: now including capital stock
list_col = list_col + ['k_stock15']
list_col_ln = ['ln_' + i for i in list_col]
log_transform = True
if log_transform:
    for i, j in zip(list_col, list_col_ln):
        df[j] = np.log(df[i])

# Take log-difference
list_col_ln_diff = [i + '_diff' for i in list_col_ln]
for i, j in zip(list_col_ln, list_col_ln_diff):
    df[j] = df[i] - df[i].shift(1)


# --- Computing ceiling


col = ['ln_gdp15_diff', 'ln_labour_diff', 'ln_employment_diff','ln_k_stock15_diff']
col_levels = ['ln_gdp15', 'ln_labour', 'ln_employment', 'ln_k_stock15']
col_trough = ['ln_gdp15_trough', 'ln_labour_trough', 'ln_employment_trough', 'ln_k_stock15_trough']
col_peak = ['ln_gdp15_peak', 'ln_labour_peak', 'ln_employment_peak', 'ln_k_stock15_peak']
col_epi = ['ln_gdp15_epi', 'ln_labour_epi', 'ln_employment_epi', 'ln_k_stock15_epi']
col_cepi = ['ln_gdp15_cepi', 'ln_labour_cepi', 'ln_employment_cepi', 'ln_k_stock15_cepi']  # ceiling episodes
col_pace = ['ln_gdp15_pace', 'ln_labour_pace', 'ln_employment_pace', 'ln_k_stock15_pace']
col_ceiling = ['ln_gdp15_ceiling', 'ln_labour_ceiling', 'ln_employment_ceiling', 'ln_k_stock15_ceiling']
col_cpace = ['ln_gdp15_cpace', 'ln_labour_cpace', 'ln_employment_cpace', 'ln_k_stock15_cpace']
ref_levels = 'ln_gdp15'
ref_diff = ref_levels + '_diff'
ref_trough = ref_levels + '_trough'
ref_peak = ref_levels + '_peak'
ref_epi = ref_levels + '_epi'
ref_cepi = ref_levels + '_cepi'
ref_pace = ref_levels + '_pace'
ref_ceiling = ref_levels + '_ceiling'
ref_cpace = ref_levels + '_cpace'
# Peak-trough selection
for input, levels, trough, peak, epi, cepi, pace, ceiling, cpace \
        in \
        tqdm(zip(col, col_levels, col_trough, col_peak, col_epi, col_cepi, col_pace, col_ceiling, col_cpace)):

    # Check unit roots in difference series
    # adf_pvalue = sm.adfuller(df[input].dropna())[1]
    # telsendmsg(conf=tel_config,
    #            msg=text_adf)

    # Check unit roots in second diff series
    # adf_pvalue = sm.adfuller((df[input] - df[input].shift(1)).dropna())[1]
    # telsendmsg(conf=tel_config,
    #            msg=text_adf)

    # Calculate standard deviation
    stdev = np.std(df[input])
    factor1 = 0.8  # adjust to include / exclude obvious episodes

    # if logdiff < 0, met with logdiff >= 0 in t+1, and abs(logdiff) >= stdev, then a trough
    df.loc[
        (
                (df[input] < 0) &
                (df[input].shift(-1) >= 0) &
                (np.abs(df[input]) >= factor1 * stdev)
        ),
        trough
    ] = 1
    df[trough] = df[trough].fillna(0)

    # if logdiff > 0, met with logdiff < 0 in t+1, and trough within next 4 quarters (variable), then a peak
    df.loc[
        (
                (df[input] >= 0) &
                (df[input].shift(-1) < 0) &
                (
                    (df[trough].shift(-1) == 1) |
                    (df[trough].shift(-2) == 1) |
                    (df[trough].shift(-3) == 1) |
                    (df[trough].shift(-4) == 1)
                )
        )
    , peak
    ] = 1
    df[peak] = df[peak].fillna(0)

    # Episodes, blanking out peak-trough gaps
    df[epi] = (df[trough] + df[peak]).cumsum()  # 0 = initial expansion, odd = gaps, even = expansion
    df.loc[((df[trough] == 1) | (df[peak] == 1)), epi] = df[epi] - 1  # exp start after trough, and end at peaks
    # df.loc[~((df[epi] % 2) == 0), epi] = np.nan
    # df[epi] = df[epi] / 2
    df.loc[df[epi].isna(), epi] = -1  # placeholder

    # Ceiling episodes, peak to peak
    df[cepi] = df[peak].cumsum()
    df.loc[df[peak] == 1, cepi] = df[cepi] - 1  # exp start after trough, and end at peaks

    # Calculate average expansion pace
    df[pace] = df.groupby(epi)[input].transform('mean')
    tab = df.groupby(epi)[input].agg('mean').reset_index()
    print(tab)
    # telsendmsg(conf=tel_config,
    #            msg=str(tab))

    # Compute 'ceiling'
    # Check if more than 1 expansion episodes
    single_exp = bool(df[epi].max() == 0)
    nrows = len(df)
    list_quarters = list(df.index)
    if not single_exp:
        # interpolate
        df.loc[df[peak] == 1, ceiling] = df[levels]  # peaks as joints
        df = df.reset_index()
        df[ceiling] = df[ceiling].interpolate(method='quadratic')  # too sparse for cubic
        df = df.set_index('quarter')

        # end-point extrapolation
        cepi_minusone = df[cepi].max() - 1
        df['_x'] = df[ceiling] - df[ceiling].shift(1)
        ceiling_minusone_avgdiff = (df.loc[df[cepi] == cepi_minusone, '_x']).mean()
        del df['_x']
        nrows_na = len(df.isna())
        for r in tqdm(range(nrows_na)):
            df.loc[df[ceiling].isna(), ceiling] = df[ceiling].shift(1) + ceiling_minusone_avgdiff

        # start-point extrapolation
        df['_x'] = df[ceiling] - df[ceiling].shift(1)
        ceiling_one_avgdiff = (df.loc[df[cepi] == 1, '_x']).mean()
        del df['_x']
        nrows_na = len(df.isna())
        for r in tqdm(range(nrows_na)):
            df.loc[df[ceiling].isna(), ceiling] = df[ceiling].shift(-1) - ceiling_one_avgdiff  # reverse

    elif single_exp:  # then follow GDP peaks and troughs
        df[peak] = df['ln_gdp15_peak'].copy()
        df[cepi] = df['ln_gdp15_cepi'].copy()

        # interpolate
        df.loc[df[peak] == 1, ceiling] = df[levels]  # peaks as joints
        df = df.reset_index()
        df[ceiling] = df[ceiling].interpolate(method='quadratic')  # too sparse for cubic
        df = df.set_index('quarter')

        # end-point extrapolation
        cepi_minusone = df[cepi].max() - 1
        df['_x'] = df[ceiling] - df[ceiling].shift(1)
        ceiling_minusone_avgdiff = (df.loc[df[cepi] == cepi_minusone, '_x']).mean()
        del df['_x']
        nrows_na = len(df.isna())
        for r in tqdm(range(nrows_na)):
            df.loc[df[ceiling].isna(), ceiling] = df[ceiling].shift(1) + ceiling_minusone_avgdiff

        # start-point extrapolation
        df['_x'] = df[ceiling] - df[ceiling].shift(1)
        ceiling_one_avgdiff = (df.loc[df[cepi] == 1, '_x']).mean()
        del df['_x']
        nrows_na = len(df.isna())
        for r in tqdm(range(nrows_na)):
            df.loc[df[ceiling].isna(), ceiling] = df[ceiling].shift(-1) - ceiling_one_avgdiff  # reverse

