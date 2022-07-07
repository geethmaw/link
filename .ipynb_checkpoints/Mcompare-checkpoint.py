# @Author: geethmawerapitiya
# @Date:   2022-06-22T21:04:46-06:00
# @Last modified by:   geethmawerapitiya
# @Last modified time: 2022-06-27T02:05:32-06:00

## Compare M computed by SST and 800hPa with 2m and 700hPa


import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from myReadGCMsDaily import read_var_mod
import calendar
from global_land_mask import globe
import glob
import math

#####JOB NUM
# job = '32'

#####Constants
Cp = 1004           #J/kg/K
Rd = 287            #J/kg/K
con= Rd/Cp

#latitude range
latr1 = 30
latr2 = 80

#pressure levels in observations
p_level = 1  ### 800hPa

### GCM
modname = ['CESM2']#,'CESM2-WACCM','HadGEM3-GC31-LL','HadGEM3-GC31-MM','IPSL-CM6A-LR','CNRM-ESM2-1','INM-CM5-0','MPI-ESM1-2-HR','UKESM1-0-LL','CMCC-CM2-SR5','CMCC-CM2-HR4','CNRM-CM6-1','CNRM-ESM2-1','IPSL-CM5A2-INCA']
varname = ['sfcWind', 'tas','psl', 'ts'] #'sfcWind', 'hfss', 'hfls', 'tas', 'ps', 'psl',,'pr'
pvarname= ['ta']


l = 0
m = len(modname)   #l+1

time1=[2010, 1, 1]
time2=[2010, 12, 30]

plt.clf()
fig = plt.figure(figsize=(12,4))

for j in range(l,m):
    print(modname[j])
    for i in varname:
        locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)
        # print(i)
    for k in pvarname:
        locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)
        # print(k)
    print('variable retrieving done')

for i in range(l,m):
    lat  = locals()['sfcWind__'+str(i+1)][0]
    lon  = locals()['sfcWind__'+str(i+1)][1]
    time = locals()['sfcWind__'+str(i+1)][2]

    print(modname[i])
    print('latitudes: length = ',len(lat), 'resolution = ',str(lat[1]-lat[0]))
    print('longitudes: length = ',len(lon), 'resolution = ',str(lon[1]-lon[0]))

    for j in varname:
        locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
        locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
        # print(np.shape(locals()[j+str(i+1)]))
#
    for k in pvarname:
        locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
        locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
        # print(np.shape(locals()[k+str(i+1)]))

        locals()['lev'+str(i+1)] = locals()['ta__'+str(i+1)][3]
        print(locals()['lev'+str(i+1)][1])
        print(locals()['lev'+str(i+1)][2])

    x_lat = np.array(lat)
    lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
    lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
    lats = lat[lat_ind1[0]:lat_ind2[0]]

    x_lon = lon
    lon = np.array(lon)
    lon[lon > 180] = lon[lon > 180]-360

    maskm = np.ones((len(time),len(lats),len(lon)))

    for a in range(len(lats)):
        for b in range(len(lon)):
            if globe.is_land(lats[a], lon[b])==True:
                maskm[:,a,b] = math.nan

 ###  averaged theta at 800hPa and surface
    theta_850 = locals()['ta'+str(i+1)][:,1,:,:]*(100000/85000)**con
    theta_700 = locals()['ta'+str(i+1)][:,2,:,:]*(100000/70000)**con
    theta_800 = theta_700 + ((2/3) * (theta_850 - theta_700))

    theta_sfc = locals()['ts'+str(i+1)]*(100000/locals()['psl'+str(i+1)])**con
    theta_t2m = locals()['tas'+str(i+1)]*(100000/locals()['psl'+str(i+1)])**con

### CAOI at 800hPa and SST
    M_sfc = theta_sfc - theta_800

### CAOI at 700hPa and t2m
    M_t2m = theta_t2m - theta_700

    lats = lat[lat_ind1[0]:lat_ind2[0]]

    m_M_sfc = M_sfc[:,lat_ind1[0]:lat_ind2[0],:]
    m_M_t2m = M_t2m[:,lat_ind1[0]:lat_ind2[0],:]

    cao_sfc = np.array(m_M_sfc)
    cao_t2m = np.array(m_M_t2m)

    plot_CAOI_sfc = np.array(np.multiply(maskm,cao_sfc))
    plot_CAOI_t2m = np.array(np.multiply(maskm,cao_t2m))

    pl_theta_sfc  = plot_CAOI_sfc.reshape(-1)
    pl_theta_t2m  = plot_CAOI_t2m.reshape(-1)

    plo_theta_sfc = pl_theta_sfc[pl_theta_sfc>-40]
    plo_theta_t2m = pl_theta_t2m[pl_theta_t2m>-40]

    plot_theta_sfc = plo_theta_sfc[plo_theta_sfc<40]
    plot_theta_t2m = plo_theta_t2m[plo_theta_t2m<40]

    xx = np.flip(np.sort(plot_theta_sfc))
    yy = np.flip(np.sort(plot_theta_t2m))


    from scipy import stats
    bin_means, bin_edges, binnumber = stats.binned_statistic(xx[0:len(yy)], yy, 'mean', bins=200)
    bin_means_x, bin_edges_x, binnumber_x = stats.binned_statistic(xx[0:len(yy)], xx[0:len(yy)], 'mean', bins=200)

    plt.plot(bin_means_x, bin_means, label=modname[i])

    print('plot done')

plt.plot([-40,10], [-40,10], linestyle='--', label='1-1')

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.ylabel('SST, 800hPa',fontsize='12')
# yti = str(merlev[p_level])
plt.xlabel("t2m, 700hPa",fontsize='12')
plt.title('M for GCMs')
plt.savefig('../figures/MforGCM.png')
