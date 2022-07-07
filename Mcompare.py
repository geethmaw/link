# @Author: geethmawerapitiya
# @Date:   2022-06-22T21:04:46-06:00
# @Last modified by:   wgeethma
# @Last modified time: 2022-07-05T10:03:27-06:00

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
from metpy.interpolate import log_interpolate_1d

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
modname = ['CESM2','CESM2-WACCM','HadGEM3-GC31-LL','HadGEM3-GC31-MM','IPSL-CM6A-LR','CNRM-ESM2-1','INM-CM5-0','MPI-ESM1-2-HR','UKESM1-0-LL','CMCC-CM2-SR5','CMCC-CM2-HR4','CNRM-CM6-1','CNRM-ESM2-1','IPSL-CM5A2-INCA','MPI-ESM1-2-LR','MPI-ESM-1-2-HAM']
varname = ['sfcWind', 'tas','psl', 'ts'] #'sfcWind', 'hfss', 'hfls', 'tas', 'ps', 'psl',,'pr'
pvarname= ['ta']


l = 0
m = len(modname)   #l+1

time1=[2010, 1, 1]
time2=[2010, 12, 30]

for j in range(l,m):
    print(modname[j])
    for i in varname:
        locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)
        # print(i)
    for k in pvarname:
        locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)
        # print(k)
    print('done')

plt.clf()
fig = plt.figure(figsize=(8,8))


for i in range(l,m):

    lat  = locals()['sfcWind__'+str(i+1)][0]
    lon  = locals()['sfcWind__'+str(i+1)][1]
    time = locals()['sfcWind__'+str(i+1)][2]

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

    print(modname[i])

    for j in varname:
        locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
        locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
        locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))



    for k in pvarname:
        levels = locals()['ta__'+str(i+1)][3]
        locals()['plot_'+k+str(i+1)] = []

        for p in range(len(levels)):
            locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
            locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
            plev  = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
            locals()['plot_'+k+str(i+1)].append(plev)
            # x   = np.array(np.repeat(levels[p],len(y)))

    locals()['plot_'+k+str(i+1)] = np.array(locals()['plot_'+k+str(i+1)])
    temp_800  = log_interpolate_1d(80000, levels,np.array(locals()['plot_ta'+str(i+1)]), axis=0, fill_value=np.nan)
    theta_800 = temp_800*(100000/80000)**con
    theta_700 = locals()['plot_ta'+str(i+1)][2,:,:,:]*(100000/70000)**con


    theta_sfc = locals()['plot_ts'+str(i+1)]*(100000/locals()['plot_psl'+str(i+1)])**con
    theta_t2m = locals()['plot_tas'+str(i+1)]*(100000/locals()['plot_psl'+str(i+1)])**con

### CAOI at 800hPa and SST
    M_800  = theta_sfc - theta_800[0,:,:,:]
    plot_M_800 = np.sort(M_800.flatten())

### CAOI at 700hPa and t2m
    M_700  = theta_t2m - theta_700
    plot_M_700 = np.sort(M_700.flatten())

    index = np.isnan(plot_M_700 * plot_M_800) == False

    from scipy import stats
    bin_means, bin_edges, binnumber = stats.binned_statistic(plot_M_700[index], plot_M_700[index], 'mean', bins=200,range=(-40,20))
    bin_means_x, bin_edges_x, binnumber_x = stats.binned_statistic(plot_M_800[index], plot_M_800[index], 'mean', bins=200,range=(-40,20))

    import numpy.ma as ma
    corr = ma.corrcoef(ma.masked_invalid(bin_means_x), ma.masked_invalid(bin_means))
    print(corr[0,1])
    plt.plot(ma.masked_invalid(bin_means_x), ma.masked_invalid(bin_means), label=modname[i]+' '+str(np.round(corr[0,1],10)))

    print('plot done')

plt.plot([-40,20], [-40,20], linestyle='--', label='1-1')

plt.legend()
plt.xlabel('SST, 800hPa',fontsize='12')
# yti = str(merlev[p_level])
plt.title('M for GCMs')
plt.ylabel("t2m, 700hPa",fontsize='12')
plt.savefig('../figures/MforGCM.png')
plt.show(block=False)
