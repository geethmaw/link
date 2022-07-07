# @Author: Geethma Werapitiya <wgeethma>
# @Date:   2022-07-05T09:54:01-06:00
# @Email:  wgeethma@uwyo.edu
# @Last modified by:   wgeethma
# @Last modified time: 2022-07-05T13:03:54-06:00

## Compare M computed by SST and 800hPa with 2m and 700hPa for OBSERVATIONS


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
plt.rcParams['figure.figsize'] = (12.0/2.5, 8.0/2.5)

plt.style.use('seaborn-whitegrid')

#####Constants
Cp = 1004           #J/kg/K
Rd = 287            #J/kg/K
con= Rd/Cp

#latitude range
latr1 = 30
latr2 = 80

#pressure levels in observations
p_level_800 = 1  ### 800hPa
p_level_700 = 3  ### 700hPa

time1=[2010, 1, 1]
time2=[2012, 12, 30]

## OBSERVATIONS
import glob
merlist = np.sort(glob.glob('../data_merra/all_lat_lon/level/MERRA2_*.nc'))
sfclist = np.sort(glob.glob('../data_merra/all_lat_lon/surface/MERRA2_*.nc'))


new_list_s = []
new_list_m = []

s = 0
m = 0
length = max(len(merlist), len(sfclist))

while m != length:
    print(s,m)
    name_s = os.path.basename(sfclist[s])
    date_s = name_s.split(".")[2]

    name_m = os.path.basename(merlist[m])
    date_m = name_m.split(".")[2]

    print(sfclist[s],date_s)
    print(merlist[m],date_m)

    if date_s==date_m:
        new_list_s.append(sfclist[s])
        new_list_m.append(merlist[m])
        s = s+1
        m = m+1

    elif date_s<date_m:
        s = s+1

    elif date_s>date_m:
        m = m+1

plot_CAOI_800 = []
plot_CAOI_700 = []

for i in range(len(new_list_m)): #len(merlist)
    d_path = new_list_m[i]
    data   = nc.Dataset(d_path)
    # print(d_path)

    if i==0:
        merlat = data.variables['lat'][:]
        merlon = data.variables['lon'][:]
        merlev = data.variables['lev'][:]
        #shape latitude
        mer_lat = np.flip(merlat)
        mer_lat = np.array(mer_lat)
        mlat_ind1 = np.where(mer_lat == mer_lat.flat[np.abs(mer_lat - (latr1)).argmin()])[0]
        mlat_ind2 = np.where(mer_lat == mer_lat.flat[np.abs(mer_lat - (latr2)).argmin()])[0]
        p_mer_lat  = np.array(mer_lat[mlat_ind1[0]:mlat_ind2[0]])
        #shape longitude
        merlon[merlon > 180] = merlon[merlon > 180]-360
        # mer_lon = np.array(merlon)

######## UL temperatures
    merT      = data.variables['T'][:] #(time, lev, lat, lon)
    mer_T_800 = np.array(np.ma.filled(merT[0,p_level_800,::-1,:], fill_value=np.nan))
    mer_T_700 = np.array(np.ma.filled(merT[0,p_level_700,::-1,:], fill_value=np.nan))
    mer_T_800 = mer_T_800[mlat_ind1[0]:mlat_ind2[0],:]
    mer_T_700 = mer_T_700[mlat_ind1[0]:mlat_ind2[0],:]

######## SURFACE temperatures
    s_path = new_list_s[i]
    sdata  = nc.Dataset(s_path)

####### SST
    sfcT      = sdata.variables['TS'][:]
    sfc_T     = np.array(np.ma.filled(sfcT[0,::-1,:], fill_value=np.nan))
    mer_T_sfc = sfc_T[mlat_ind1[0]:mlat_ind2[0],:]

####### T2M
    t2mT      = sdata.variables['T2M'][:]
    t2m_T     = np.array(np.ma.filled(t2mT[0,::-1,:], fill_value=np.nan))
    mer_T_t2m = t2m_T[mlat_ind1[0]:mlat_ind2[0],:]

####### SLP
    sfcP      = sdata.variables['SLP'][:]
    sfc_P     = np.array(np.ma.filled(sfcP[0,::-1,:], fill_value=np.nan))
    mer_P_sfc = sfc_P[mlat_ind1[0]:mlat_ind2[0],:]


    theta_800 = np.array(np.ma.filled(np.multiply(mer_T_800, (100000/(merlev[p_level_800]*100))**(Rd/Cp)), fill_value=np.nan))
    theta_700 = np.array(np.ma.filled(np.multiply(mer_T_700, (100000/(merlev[p_level_700]*100))**(Rd/Cp)), fill_value=np.nan))
    theta_sfc = np.array(np.ma.filled(np.multiply(mer_T_sfc, (100000/mer_P_sfc)**(Rd/Cp)), fill_value=np.nan))
    theta_t2m = np.array(np.ma.filled(np.multiply(mer_T_t2m, (100000/mer_P_sfc)**(Rd/Cp)), fill_value=np.nan))

    CAOI_800  = np.array(np.subtract(theta_sfc,theta_800))
    CAOI_700  = np.array(np.subtract(theta_t2m,theta_700))


    #Mask for the ocean
    maskm = np.ones((len(p_mer_lat),len(merlon)))

    for a in range(len(p_mer_lat)):
        for b in range(len(merlon)):
            if globe.is_land(p_mer_lat[a], merlon[b])==True:
                maskm[a,b] = math.nan
    ##############################
#######masked CAOI
    mask_CAOI_800  = np.array(np.multiply(maskm,CAOI_800)).ravel()
    mask_CAOI_700  = np.array(np.multiply(maskm,CAOI_700)).ravel()

    plot_CAOI_800.extend(mask_CAOI_800)
    plot_CAOI_700.extend(mask_CAOI_700)

plot_CAOI_800 = np.array(plot_CAOI_800)
plot_CAOI_700 = np.array(plot_CAOI_700)
plot_indx = np.isnan(plot_CAOI_800*plot_CAOI_700)==False
###################################

from scipy import stats
bin_means, bin_edges, binnumber       = stats.binned_statistic(plot_CAOI_800[plot_indx], plot_CAOI_700[plot_indx], 'mean', bins=100, range=(-30,20))
bin_means_x, bin_edges_x, binnumber_x = stats.binned_statistic(plot_CAOI_800[plot_indx], plot_CAOI_800[plot_indx], 'mean', bins=100, range=(-30,20))

M_800 = np.ma.masked_invalid(bin_means_x)
M_700 = np.ma.masked_invalid(bin_means)

corr = np.ma.corrcoef(M_800, M_700)
plt.plot(M_800, M_700, label='observations: cor-coef: '+str(np.round(corr[0,1],10)))

plt.plot([-30,20], [-30,20], linestyle='--', color='red',label='1-1')

x1_vals = np.abs(M_800-0)
x1      = np.where(x1_vals==np.min(x1_vals))[0][0]
x2_vals = np.abs(M_800-10)
x2      = np.where(x2_vals==np.min(x2_vals))[0][0]
plt.fill_between(M_800[x1:x2] , M_700[x1:x2], np.repeat(-30, x2-x1) ,color ='red', alpha=0.2)
plt.fill_betweenx(M_700[x1:x2], np.repeat(-30, x2-x1), M_800[x1:x2] ,color ='red', alpha=0.2)

plt.legend()
plt.xlabel('SST, 800hPa')
# yti = str(merlev[p_level])
plt.title('Stability matrix comparison\nfor Observations')
plt.ylabel("t2m, 700hPa")
plt.savefig('../figures/MforOBS.png')

# index = np.isnan(bin_means_x*bin_means)==False
