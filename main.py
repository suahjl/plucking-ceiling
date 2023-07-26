import telegram_send
import time
import os
from dotenv import load_dotenv

time_start_main = time.time()

# 0 --- Main settings
load_dotenv()
tel_config = os.getenv('TEL_CONFIG')  # EcMetrics_Config_GeneralFlow EcMetrics_Config_RMU


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


# II --- Run scripts, in intended sequence
# import pluckingpo_compile_input_data
import pluckingpo_compute_ceiling_dns
import pluckingpo_plot_bound_versus_nobound
import pluckingpo_plot_updateceiling
import pluckingpo_plot_histdecomp

import pluckingpo_compute_ceiling_dns_urate

import boombustpo_compute_pf
import boombustpo_compute_pf_onesided

import boombustpo_plot_histdecomp
import boombustpo_plot_histdecomp_onesided

# import boombustpo_compile_input_data_kf
# import boombustpo_compile_input_data_kf_onesided

import boombustpo_compute_kf
import boombustpo_compute_kf_onesided

time.sleep(15)

import pluckingpo_plot_ceiling_po_og
import pluckingpo_compute_crisisrecoveries
import pluckingpo_plot_crisisrecoveries

import pluckingpo_compile_charts

time.sleep(15)

import pluckingpo_compute_vintages_dns
import pluckingpo_plot_vintages_dns

import boombustpo_compute_vintages
import boombustpo_compute_vintages_onesided
import boombustpo_compute_vintages_onesided_pfonly

import boombustpo_plot_vintages
import boombustpo_plot_vintages_onesided
import boombustpo_plot_vintages_onesided_pfonly

time.sleep(15)
import pluckingpo_plot_macro_comparison  # some lines don't always run; need to check

import boombustpo_plot_macro_comparison_twosidedavg
import boombustpo_plot_macro_comparison_twosidedkf
import boombustpo_plot_macro_comparison_twosidedpf
import boombustpo_plot_macro_comparison_onesidedavg
import boombustpo_plot_macro_comparison_onesidedkf
import boombustpo_plot_macro_comparison_onesidedpf

time.sleep(15)

import comparingpo_supply_demand_twosidedavg
import comparingpo_supply_demand_twosidedpf
import comparingpo_supply_demand_twosidedkf
import comparingpo_supply_demand_onesidedavg
import comparingpo_supply_demand_onesidedpf
import comparingpo_supply_demand_onesidedkf

time.sleep(15)

# import pluckingpo_crosscountry_rgdp
# import pluckingpo_crosscountry_ur

import generate_qr

# End
time_text = '\n----- Full pluckingpo-boombustpo estimation routine completed in ' + "{:.0f}".format(
    time.time() - time_start_main) + ' seconds -----'
print(time_text)
telsendmsg(conf=tel_config,
           msg=time_text)
