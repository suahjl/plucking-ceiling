# -------------- Allow PO to deviate from first guess based on K and N
# -------------- Casts the bands as 'uncertainty in timing of peaks by 1Q earlier'
# -------------- Option to show or hide confidence bands
# -------------- Option to not show any forecasts
# -------------- Open data version (est3)


import pandas as pd
import numpy as np
from datetime import date, timedelta
import telegram_send
from PIL import Image
from tqdm import tqdm
import time
from dotenv import load_dotenv
import os
import ast

time_start = time.time()

# 0 --- Main settings
load_dotenv()
tel_config = os.getenv('TEL_CONFIG')
T_lb = '1995Q1'
T_lb_day = date(1995, 1, 1)
use_forecast = ast.literal_eval(os.getenv('USE_FORECAST_BOOL'))
if use_forecast:
    file_suffix_fcast = '_forecast'
    fcast_start = '2023Q1'
    fcast_start_int = 13
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


def pil_img2pdf(list_images, extension='png', img_path='Output/', pdf_name='PluckingPO_AllCharts'):
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


# II --- Compile

if not use_forecast:
    seq_output = [
        'PluckingPO_UpdateCeiling_Decomp',
        'PluckingPO_HardAndNoBound',
        'PluckingPO_HardAndNoBound_Diff',
        'PluckingPO_ObsCeiling_GDP',
        'PluckingPO_ObsCeiling_Labour',
        'PluckingPO_ObsCeiling_Capital',
        'PluckingPO_ObsCeiling_TFP',
        'PluckingPO_ObsCeiling_OG',
        'PluckingPO_ObsCeiling_OG_Norm',
        'PluckingPO_HistDecomp_Ceiling',
        'BoomBustPO_HistDecomp_PO',
        'PluckingPO_HistDecomp_Obs',
        'PluckingPO_ObsCeiling_CrisisRecoveries',
    ]
if use_forecast:
    seq_output = [
        'PluckingPO_UpdateCeiling_Decomp' + file_suffix_fcast,
        'PluckingPO_HardAndNoBound' + file_suffix_fcast,
        'PluckingPO_HardAndNoBound_Diff' + file_suffix_fcast,
        'PluckingPO_ObsCeiling_GDP' + file_suffix_fcast,
        'PluckingPO_ObsCeiling_Labour' + file_suffix_fcast,
        'PluckingPO_ObsCeiling_Capital' + file_suffix_fcast,
        'PluckingPO_ObsCeiling_TFP' + file_suffix_fcast,
        'PluckingPO_ObsCeiling_OG' + file_suffix_fcast,
        'PluckingPO_ObsCeiling_OG_Norm' + file_suffix_fcast,
        'PluckingPO_HistDecomp_Ceiling' + file_suffix_fcast,
        'BoomBustPO_HistDecomp_PO' + file_suffix_fcast,
        'PluckingPO_HistDecomp_Obs' + file_suffix_fcast,
        'PluckingPO_ObsCeiling_CrisisRecoveries' + file_suffix_fcast
    ]
pil_img2pdf(list_images=seq_output,
            img_path='Output/',
            extension='png',
            pdf_name='PluckingPO_AllCharts' + file_suffix_fcast)
telsendfiles(conf=tel_config,
             path='Output/PluckingPO_AllCharts' + file_suffix_fcast + '.pdf',
             cap='All charts from the PluckingPO estimation flow')


# III --- Notify
telsendmsg(conf=tel_config,
           msg='pluckingpo_compile_charts: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
