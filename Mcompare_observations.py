# @Author: Geethma Werapitiya <wgeethma>
# @Date:   2022-07-05T09:54:01-06:00
# @Email:  wgeethma@uwyo.edu
# @Last modified by:   geethmawerapitiya
# @Last modified time: 2022-07-18T03:53:24-06:00

## Compare M computed by SST and 800hPa with 2m and 700hPa for OBSERVATIONS
## Updated 700hPa to 850hPa   p_level_700 = 0  ### 850hPa and t2m   CAOI_700  = np.array(np.subtract(theta_t2m,theta_700))


import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from global_land_mask import globe
import glob
import math
import os
import netCDF4 as nc
from regrid_wght_3d import regrid_wght_wnans

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 100
plt.clf()
plt.rcParams['figure.figsize'] = (8, 8)

plt.style.use('seaborn-whitegrid')

# #####Constants
# Cp = 1004           #J/kg/K
# Rd = 287            #J/kg/K
# con= Rd/Cp
from con_models import get_cons
con, use_colors, varname, pvarname, modname, warming_modname, hiresmd, amip_md = get_cons()
#latitude range
latr1 = 30
latr2 = 80
M_range = (0,15)
max_c   = 500
n_bins  = 100

from obs_data_function import obs
merwind_t1, macwind_t1, temp_t1, sfctemp_t1, sfcpres_t1, p_lev_obs_t1, p_mer_lat, merlon, merlev = obs('surface','T2M', 700, latr1, latr2)
merwind_t2, macwind_t2, temp_t2, sfctemp_t2, sfcpres_t2, p_lev_obs_t2, p_mer_lat, merlon, merlev = obs('surface','TS', 800, latr1, latr2)

theta_700 = np.array(np.ma.filled(np.multiply(temp_t1, (100000/(merlev[p_lev_obs_t1]*100))**(con)), fill_value=np.nan))
theta_800 = np.array(np.ma.filled(np.multiply(temp_t2, (100000/(merlev[p_lev_obs_t2]*100))**(con)), fill_value=np.nan))
theta_t2m = np.array(np.ma.filled(np.multiply(sfctemp_t1, (100000/sfcpres_t1)**(con)), fill_value=np.nan))
theta_sfc = np.array(np.ma.filled(np.multiply(sfctemp_t2, (100000/sfcpres_t2)**(con)), fill_value=np.nan))

CAOI_700  = np.array(np.subtract(theta_t2m,theta_700))
CAOI_800  = np.array(np.subtract(theta_sfc,theta_800))

#Mask for the ocean
maskm = np.ones((len(p_mer_lat),len(merlon)))

for a in range(len(p_mer_lat)):
    for b in range(len(merlon)):
        if globe.is_land(p_mer_lat[a], merlon[b])==True:
            maskm[a,b] = math.nan
    ##############################
#######masked CAOI
plot_CAOI_700  = np.array(np.multiply(maskm,CAOI_700)).ravel()
plot_CAOI_800  = np.array(np.multiply(maskm,CAOI_800)).ravel()

plot_indx = np.isnan(plot_CAOI_700*plot_CAOI_800)==False
###################################
ind = np.argsort(plot_CAOI_800[plot_indx])
x   = np.sort(plot_CAOI_800[plot_indx])
y   = plot_CAOI_700[plot_indx][ind]

ind_sst = np.where(x>0)
xx = x[ind_sst]
yy = y[ind_sst]

ind = np.isnan(xx*yy)==False

# plt.plot(xx[ind], yy[ind])



# x   = x[y>-30]
# y   = y[y>-30]
#
n_bins = 200
from scipy import stats
bin_means, bin_edges, binnumber       = stats.binned_statistic(xx[ind], yy[ind], 'mean',  bins=n_bins)
bin_means_x, bin_edges_x, binnumber_x = stats.binned_statistic(xx[ind], xx[ind], 'mean',  bins=n_bins)

ind_c = np.isnan(bin_means*bin_means_x)==False

M_800 = np.ma.masked_invalid(bin_means_x[ind_c])
M_700 = np.ma.masked_invalid(bin_means[ind_c])

corr = np.ma.corrcoef(M_700, M_800)

plt.clf()
plt.plot(xx[ind], yy[ind], label='observations: cor-coef: '+str(np.round(corr[0,1],10)), color="#117733")
plt.ylabel('t2m, 700hPa')
# yti = str(merlev[p_level])
plt.title('Stability matrix comparison\nfor Observations',fontsize=7)
plt.xlabel("SST, 800hPa")
plt.savefig('../figures/noBinsnew_800&700.png')
#
# plt.plot([-30,0], [-30,0], linestyle='--', color='red',label='1-1')
#
# x1_vals = np.abs(M_800-0)
# x2_vals = np.abs(M_800-10)
# x1      = np.where(x1_vals==np.min(x1_vals))[0][0]
# x2      = np.where(x2_vals==np.min(x2_vals))[0][0]
# plt.fill_between(M_800[x1:x2] , M_850[x1:x2], np.repeat(-30, x2-x1) ,color ='red', alpha=0.2)
# plt.fill_betweenx(M_850[x1:x2], np.repeat(-30, x2-x1), M_800[x1:x2] ,color ='red', alpha=0.2)
#
# plt.legend()
# plt.ylabel('t2m, 850hPa')
# # yti = str(merlev[p_level])
# plt.title('Stability matrix comparison\nfor Observations',fontsize=7)
# plt.xlabel("SST, 800hPa")
# plt.savefig('../figures/800&TS_850&t2m_MforOBS.png')
