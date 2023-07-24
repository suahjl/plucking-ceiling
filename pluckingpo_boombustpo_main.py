import telegram_send
import time

time_start_main = time.time()

# 0 --- Main settings
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


# II --- Run scripts, in intended sequence
import pluckingpo_compile_input_data
import pluckingpo_compute_ceiling_dns  # pluckingpo_compute_ceiling
import pluckingpo_plot_bound_versus_nobound
import pluckingpo_plot_updateceiling
import pluckingpo_plot_histdecomp

import boombustpo_compute_pf
import boombustpo_plot_histdecomp
import boombustpo_compile_input_data_kf
import boombustpo_compute_kf

time.sleep(15)

import pluckingpo_plot_ceiling_po_og
import pluckingpo_compute_crisisrecoveries
import pluckingpo_plot_crisisrecoveries

import pluckingpo_compile_charts

# time.sleep(15)

# import pluckingpo_compute_vintages_dns
# import pluckingpo_plot_vintages
# import boombustpo_compute_vintages_dns
# import boombustpo_plot_vintages

# time.sleep(15)
# import pluckingpo_plot_macro_comparison
# import boombustpo_plot_macro_comparison

# import pluckingpo_crosscountry_rgdp
# time.sleep(15)
# import pluckingpo_crosscountry_ur
# import comparingpo_supply_demand



# End
time_text = '\n----- Full pluckingpo-boombustpo estimation routine completed in ' + "{:.0f}".format(time.time() - time_start_main) + ' seconds -----'
print(time_text)
telsendmsg(conf=tel_config,
           msg=time_text)