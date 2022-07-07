# @Author: geethmawerapitiya
# @Date:   2022-06-26T23:22:59-06:00
# @Last modified by:   geethmawerapitiya
# @Last modified time: 2022-06-27T03:29:32-06:00


# relationship between temperature and pressure

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

#####Constants
Cp = 1004           #J/kg/K
Rd = 287            #J/kg/K
con= Rd/Cp

#latitude range
latr1 = 30
latr2 = 80

#pressure levels in observations
p_level = 3  ### 700hPa

### GCM
modname = ['CESM2']#,'CESM2-WACCM','HadGEM3-GC31-LL','HadGEM3-GC31-MM','IPSL-CM6A-LR'] #,'CNRM-ESM2-1','INM-CM5-0','MPI-ESM1-2-HR','UKESM1-0-LL','CMCC-CM2-SR5','CMCC-CM2-HR4','CNRM-CM6-1','CNRM-ESM2-1','IPSL-CM5A2-INCA']
varname = ['tas','psl'] #'sfcWind', 'hfss', 'hfls', 'tas', 'ps', 'psl',,'pr'
pvarname= ['ta']

l = 0
m = len(modname)   #l+1

time1=[2010, 1, 1]
time2=[2012, 12, 30]



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
fig = plt.figure(figsize=(6,4))


for i in range(l,m):

    temperature = []
    pressure    = []

    lat  = locals()['tas__'+str(i+1)][0]
    lon  = locals()['tas__'+str(i+1)][1]
    time = locals()['tas__'+str(i+1)][2]

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
#
    temperature.extend(locals()['plot_tas'+str(i+1)])
    pressure.extend(locals()['plot_psl'+str(i+1)])

    pressure    = np.array(pressure).ravel()
    temperature = np.array(temperature).ravel()



# for i in range(l,m):
    pres = []
    temp = []

    for k in pvarname:
        levels = locals()['ta__'+str(i+1)][3]

        for p in range(len(levels)):
            locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
            locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
            y   = (np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))).ravel()
            x   = np.array(np.repeat(levels[p],len(y)))

            pres.extend(x)
            temp.extend(y)

    pressure = np.append(pressure,pres)
    temperature = np.append(temperature,temp)

    ind = np.argsort(temperature)
    plot_temperature    = np.sort(temperature)
    plot_pressure = pressure[ind]

    indx = np.isnan(plot_pressure*plot_temperature)==False


    from scipy import stats
    # M_range = [np.percentile(xx[indx],2.5),np.percentile(xx[indx],97.5)]
    bin_means, bin_edges, binnumber       = stats.binned_statistic(plot_temperature[indx], plot_pressure[indx],  'mean', bins=100,range=(0,350))
    bin_means_x, bin_edges_x, binnumber_x = stats.binned_statistic(plot_temperature[indx], plot_temperature[indx], 'mean', bins=100,range=(0,350))

    index = np.isnan(bin_means_x*bin_means)==False


    plt.plot(bin_means[index], bin_means_x[index], label=modname[i],alpha=0.5)

#interp_800 = metpy.interpolate.log_interpolate_1d(80000,bin_means[index],bin_means_x[index])

f = np.polyfit(np.log(bin_means[index]),bin_means_x[index] , 1)
y = f[0]*np.log(np.arange(1,100000,100)) + f[1]
plt.plot(np.arange(1,100000,100),y, linestyle='--', label='log-plot')
    
plt.ylim(180,320)
plt.ylabel('temperature')
plt.xlabel('pressure')
plt.legend()

plt.savefig('../figures/tempVspres_log.png')
