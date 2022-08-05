# @Author: Geethma Werapitiya <wgeethma>
# @Date:   2022-07-17T20:51:57-06:00
# @Email:  wgeethma@uwyo.edu
# @Last modified by:   wgeethma
# @Last modified time: 2022-07-20T01:23:58-06:00



import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import sys
# sys.path.append("/glade/u/home/dtmccoy/scripts")
import seaborn as sns
from numpy.ma import *
import datetime
from numpy import *
import glob
import matplotlib.pyplot as plt
import scipy.stats as st
import netCDF4 as nc
from global_land_mask import globe
from scipy import stats
from warming_gcm_function import read_warming
from myReadGCMsDaily import read_var_mod
%matplotlib inline

import matplotlib as mpl
# plt.clf()
mpl.rcParams['figure.dpi'] = 100
plt.gcf().set_size_inches(6.4, 4.8)

#latitude range
latr1 = 30
latr2 = 80

from con_models import get_cons
con, use_colors, varname, pvarname, modname, warming_modname, hiresmd, amip = get_cons()

pp_path_scratch='/glade/scratch/geethma/cmip6'
def read_warming_temp(level, modn, exper, varnm, n_month):
    files = []
    path    = pp_path_scratch+'/'+level+'/'

    ncname  = 'CMIP6.CMIP.*'+modn+'.'+exper+'.*'+varnm

    fn      = np.sort(glob.glob(path+ncname+'*nc*'))
    data    = xr.open_dataset(fn[-1])

    Jan     = data.where((data['time.month'] == n_month), drop=True)
    data1   = Jan[varnm]
    times   = []
    data    = []

    if (np.shape(data1)[0]>=90):
        data   = data1[-90::,:,:]

    else:
        print('last file too small')
        data.extend(data1)
        d_len = np.shape(data1)[0]
        r_len = 90 - d_len
        data_e    = xr.open_dataset(fn[-2])
        Jan_e     = data_e.where((data_e['time.month'] == n_month), drop=True)
        data1_e   = Jan_e[varnm]
        data.extend(data1_e[-r_len::,:,:])
        de_len = np.shape(data)[0]

        if (de_len<90):
            s_len     = 90 - de_len
            data_e    = xr.open_dataset(fn[-3])
            Jan_e     = data_e.where((data_e['time.month'] == n_month), drop=True)
            data1_e   = Jan_e[varnm]
            data.extend(data1_e[-s_len::,:,:])


        # times.extend(Jan_e['time'])

    lats    = Jan['lat']
    lons    = Jan['lon']
    lev     = []


    if level=='p_level':
        lev = Jan['plev']


    return(lats,lons,times,lev,np.array(data))

l = 0
m = len(warming_modname)


############################# JANUARY ####################################################################

DJF_warm = []
DJF_lats_warm = []
DJF = []
DJF_lats = []

for mod in range(0,m):

    M_warm = []

    print('JANUARY')
    print(warming_modname[mod])
    for i in varname:
        locals()[i+'__'+str(mod+1)] = read_warming_temp('surface', warming_modname[mod], 'abrupt-4xCO2', i, 1)

    for k in pvarname:
        locals()[k+'__'+str(mod+1)] = read_warming_temp('p_level', warming_modname[mod], 'abrupt-4xCO2', k, 1)


    # print(warming_modname[mod])
    for j in varname:
        lat  = locals()[j+'__'+str(mod+1)][0]
        lon  = locals()[j+'__'+str(mod+1)][1]
        # time = locals()[j+'__'+str(i+1)][2]

        x_lat = np.array(lat)
        lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
        lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
        lats = lat[lat_ind1[0]:lat_ind2[0]]

        x_lon = lon
        lon = np.array(lon)
        lon[lon > 180] = lon[lon > 180]-360

        maskm = np.ones((90,len(lats),len(lon)))

        for a in range(len(lats)):
            for b in range(len(lon)):
                if globe.is_land(lats[a], lon[b])==True:
                    maskm[:,a,b] = math.nan
        #print(j)
        locals()[j+str(mod+1)] = locals()[j+'__'+str(mod+1)][4]
        locals()[j+str(mod+1)] = np.ma.filled(locals()[j+str(mod+1)], fill_value=np.nan)
        locals()['plot_'+j+str(mod+1)] = np.array(np.multiply(maskm,locals()[j+str(mod+1)][:,lat_ind1[0]:lat_ind2[0],:]))
        locals()['grid_'+j+str(mod+1)] = locals()['plot_'+j+str(mod+1)]
        #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

    for k in pvarname:
        #print(k)
        lat  = locals()[k+'__'+str(mod+1)][0]
        lon  = locals()[k+'__'+str(mod+1)][1]
        # time = locals()[k+'__'+str(i+1)][2]

        x_lat = np.array(lat)
        lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
        lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
        lats = lat[lat_ind1[0]:lat_ind2[0]]

        x_lon = lon
        lon = np.array(lon)
        lon[lon > 180] = lon[lon > 180]-360

        maskm = np.ones((90,len(lats),len(lon)))

        for a in range(len(lats)):
            for b in range(len(lon)):
                if globe.is_land(lats[a], lon[b])==True:
                    maskm[:,a,b] = math.nan
        locals()['plot_levels'+str(mod+1)] = locals()['ta__'+str(mod+1)][3]
        locals()['grid_'+k+str(mod+1)] = []

        levels = locals()['plot_levels'+str(mod+1)]

        for p in range(len(levels)):
            if levels[p] == 70000:
                #print(levels[p])
                locals()[k+str(mod+1)] = locals()[k+'__'+str(mod+1)][4]
                locals()[k+str(mod+1)] = np.ma.filled(locals()[k+str(mod+1)], fill_value=np.nan)
                temp_700   = np.array(np.multiply(maskm,locals()[k+str(mod+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                break;

    theta_700 = temp_700*(100000/70000)**con
    # theta_700 = grid_t_700*(100000/70000)**con
    theta_t2m = locals()['grid_tas'+str(mod+1)]*(100000/locals()['grid_psl'+str(mod+1)])**con

    # t = min(len(theta_t2m),len(theta_700))
    M_700  = theta_t2m - theta_700
    M_warm.extend(M_700)
    print(np.shape(M_700))
    print(np.shape(M_warm))

############################# FEBRUARY ####################################################################
    print('FEBRUARY')
    for i in varname:
        locals()[i+'__'+str(mod+1)] = read_warming_temp('surface', warming_modname[mod], 'abrupt-4xCO2', i, 1)

    for k in pvarname:
        locals()[k+'__'+str(mod+1)] = read_warming_temp('p_level', warming_modname[mod], 'abrupt-4xCO2', k, 1)


    # print(warming_modname[i])
    for j in varname:
        lat  = locals()[j+'__'+str(mod+1)][0]
        lon  = locals()[j+'__'+str(mod+1)][1]
        # time = locals()[j+'__'+str(i+1)][2]

        x_lat = np.array(lat)
        lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
        lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
        lats = lat[lat_ind1[0]:lat_ind2[0]]

        x_lon = lon
        lon = np.array(lon)
        lon[lon > 180] = lon[lon > 180]-360

        maskm = np.ones((90,len(lats),len(lon)))

        for a in range(len(lats)):
            for b in range(len(lon)):
                if globe.is_land(lats[a], lon[b])==True:
                    maskm[:,a,b] = math.nan
        #print(j)
        locals()[j+str(mod+1)] = locals()[j+'__'+str(mod+1)][4]
        locals()[j+str(mod+1)] = np.ma.filled(locals()[j+str(mod+1)], fill_value=np.nan)
        locals()['plot_'+j+str(mod+1)] = np.array(np.multiply(maskm,locals()[j+str(mod+1)][:,lat_ind1[0]:lat_ind2[0],:]))
        locals()['grid_'+j+str(mod+1)] = locals()['plot_'+j+str(mod+1)]

    for k in pvarname:
        #print(k)
        lat  = locals()[k+'__'+str(mod+1)][0]
        lon  = locals()[k+'__'+str(mod+1)][1]
        # time = locals()[k+'__'+str(i+1)][2]

        x_lat = np.array(lat)
        lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
        lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
        lats = lat[lat_ind1[0]:lat_ind2[0]]

        x_lon = lon
        lon = np.array(lon)
        lon[lon > 180] = lon[lon > 180]-360

        maskm = np.ones((90,len(lats),len(lon)))

        for a in range(len(lats)):
            for b in range(len(lon)):
                if globe.is_land(lats[a], lon[b])==True:
                    maskm[:,a,b] = math.nan
        locals()['plot_levels'+str(mod+1)] = locals()['ta__'+str(mod+1)][3]
        locals()['grid_'+k+str(mod+1)] = []

        levels = locals()['plot_levels'+str(mod+1)]

        for p in range(len(levels)):
            if levels[p] == 70000:
                #print(levels[p])
                locals()[k+str(mod+1)] = locals()[k+'__'+str(mod+1)][4]
                locals()[k+str(mod+1)] = np.ma.filled(locals()[k+str(mod+1)], fill_value=np.nan)
                temp_700   = np.array(np.multiply(maskm,locals()[k+str(mod+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                break;

    theta_700 = temp_700*(100000/70000)**con
    # theta_700 = grid_t_700*(100000/70000)**con
    theta_t2m = locals()['grid_tas'+str(mod+1)]*(100000/locals()['grid_psl'+str(mod+1)])**con

    # t = min(len(theta_t2m),len(theta_700))
    M_700  = theta_t2m - theta_700
    M_warm.extend(M_700)
    print(np.shape(M_700))
    print(np.shape(M_warm))


############################# DECEMBER ####################################################################
    print('DECEMBER')
    for i in varname:
        locals()[i+'__'+str(mod+1)] = read_warming_temp('surface', warming_modname[mod], 'abrupt-4xCO2', i, 1)

    for k in pvarname:
        locals()[k+'__'+str(mod+1)] = read_warming_temp('p_level', warming_modname[mod], 'abrupt-4xCO2', k, 1)


    # print(warming_modname[i])
    for j in varname:
        lat  = locals()[j+'__'+str(mod+1)][0]
        lon  = locals()[j+'__'+str(mod+1)][1]
        # time = locals()[j+'__'+str(i+1)][2]

        x_lat = np.array(lat)
        lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
        lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
        lats = lat[lat_ind1[0]:lat_ind2[0]]

        x_lon = lon
        lon = np.array(lon)
        lon[lon > 180] = lon[lon > 180]-360

        maskm = np.ones((90,len(lats),len(lon)))

        for a in range(len(lats)):
            for b in range(len(lon)):
                if globe.is_land(lats[a], lon[b])==True:
                    maskm[:,a,b] = math.nan
        #print(j)
        locals()[j+str(mod+1)] = locals()[j+'__'+str(mod+1)][4]
        locals()[j+str(mod+1)] = np.ma.filled(locals()[j+str(mod+1)], fill_value=np.nan)
        locals()['plot_'+j+str(mod+1)] = np.array(np.multiply(maskm,locals()[j+str(mod+1)][:,lat_ind1[0]:lat_ind2[0],:]))
        locals()['grid_'+j+str(mod+1)] = locals()['plot_'+j+str(mod+1)]

    for k in pvarname:
        #print(k)
        lat  = locals()[k+'__'+str(mod+1)][0]
        lon  = locals()[k+'__'+str(mod+1)][1]
        # time = locals()[k+'__'+str(i+1)][2]

        x_lat = np.array(lat)
        lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
        lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
        lats = lat[lat_ind1[0]:lat_ind2[0]]

        x_lon = lon
        lon = np.array(lon)
        lon[lon > 180] = lon[lon > 180]-360

        maskm = np.ones((90,len(lats),len(lon)))

        for a in range(len(lats)):
            for b in range(len(lon)):
                if globe.is_land(lats[a], lon[b])==True:
                    maskm[:,a,b] = math.nan
        locals()['plot_levels'+str(mod+1)] = locals()['ta__'+str(mod+1)][3]
        locals()['grid_'+k+str(mod+1)] = []

        levels = locals()['plot_levels'+str(mod+1)]

        for p in range(len(levels)):
            if levels[p] == 70000:
                #print(levels[p])
                locals()[k+str(mod+1)] = locals()[k+'__'+str(mod+1)][4]
                locals()[k+str(mod+1)] = np.ma.filled(locals()[k+str(mod+1)], fill_value=np.nan)
                temp_700   = np.array(np.multiply(maskm,locals()[k+str(mod+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                break;

    theta_700 = temp_700*(100000/70000)**con
    # theta_700 = grid_t_700*(100000/70000)**con
    theta_t2m = locals()['grid_tas'+str(mod+1)]*(100000/locals()['grid_psl'+str(mod+1)])**con

    # t = min(len(theta_t2m),len(theta_700))
    M_700  = theta_t2m - theta_700
    M_warm.extend(M_700)
    print(np.shape(M_700))
    print(np.shape(M_warm))

#################plot#############################
    num_UM = []
    num_M  = []

    M_warm = np.array(M_warm)

    for j in range(np.shape(M_warm)[1]):
        count_UM = 0

        for i in range(np.shape(M_warm)[0]):
            for k in range(np.shape(M_warm)[2]):
                if M_warm[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_warm)[1]):
        count_M = 0

        for i in range(np.shape(M_warm)[0]):
            for k in range(np.shape(M_warm)[2]):
                if M_warm[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    lats = np.array(lats)

    DJF_warm.append(f)
    DJF_lats_warm.append(lats)

    plt.plot(lats,f,color=use_colors[mod],label=warming_modname[mod],linestyle='dashed')

    M_plot = []

    time1=[2010, 1, 1]
    time2=[2010, 2, 28]

    for i in varname:
        locals()[i+'__'+str(mod+1)] = read_var_mod('surface', warming_modname[mod], 'historical', i, time1, time2)

    for k in pvarname:
        locals()[k+'__'+str(mod+1)] = read_var_mod('p_level', warming_modname[mod], 'historical', k, time1, time2)


    for j in varname:
        lat  = locals()[j+'__'+str(mod+1)][0]
        lon  = locals()[j+'__'+str(mod+1)][1]
        time = locals()[j+'__'+str(mod+1)][2]

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
        #print(j)
        locals()[j+str(mod+1)] = locals()[j+'__'+str(mod+1)][4]
        locals()[j+str(mod+1)] = np.ma.filled(locals()[j+str(mod+1)], fill_value=np.nan)
        locals()['plot_'+j+str(mod+1)] = np.array(np.multiply(maskm,locals()[j+str(mod+1)][:,lat_ind1[0]:lat_ind2[0],:]))
        locals()['grid_'+j+str(mod+1)] = locals()['plot_'+j+str(mod+1)]

    for k in pvarname:
        #print(k)
        lat  = locals()[k+'__'+str(mod+1)][0]
        lon  = locals()[k+'__'+str(mod+1)][1]
        time = locals()[k+'__'+str(mod+1)][2]

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
        locals()['plot_levels'+str(mod+1)] = locals()['ta__'+str(mod+1)][3]
        locals()['grid_'+k+str(mod+1)] = []

        levels = locals()['plot_levels'+str(mod+1)]

        for p in range(len(levels)):
            if levels[p] == 70000:
                locals()[k+str(mod+1)] = locals()[k+'__'+str(mod+1)][4]
                locals()[k+str(mod+1)] = np.ma.filled(locals()[k+str(mod+1)], fill_value=np.nan)
                temp_700   = np.array(np.multiply(maskm,locals()[k+str(mod+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                break;

    theta_700 = temp_700*(100000/70000)**con
    theta_t2m = locals()['grid_tas'+str(mod+1)]*(100000/locals()['grid_psl'+str(mod+1)])**con

    # t = min(len(theta_t2m),len(theta_700))
    M_700  = theta_t2m - theta_700
    M_plot.extend(M_700)

############################################### 0,2
    time1=[2010, 12, 1]
    time2=[2011, 2, 28]

    for i in varname:
        locals()[i+'__'+str(mod+1)] = read_var_mod('surface', warming_modname[mod], 'historical', i, time1, time2)

    for k in pvarname:
        locals()[k+'__'+str(mod+1)] = read_var_mod('p_level', warming_modname[mod], 'historical', k, time1, time2)


    for j in varname:
        lat  = locals()[j+'__'+str(mod+1)][0]
        lon  = locals()[j+'__'+str(mod+1)][1]
        time = locals()[j+'__'+str(mod+1)][2]

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
        #print(j)
        locals()[j+str(mod+1)] = locals()[j+'__'+str(mod+1)][4]
        locals()[j+str(mod+1)] = np.ma.filled(locals()[j+str(mod+1)], fill_value=np.nan)
        locals()['plot_'+j+str(mod+1)] = np.array(np.multiply(maskm,locals()[j+str(mod+1)][:,lat_ind1[0]:lat_ind2[0],:]))
        locals()['grid_'+j+str(mod+1)] = locals()['plot_'+j+str(mod+1)]

    for k in pvarname:
        #print(k)
        lat  = locals()[k+'__'+str(mod+1)][0]
        lon  = locals()[k+'__'+str(mod+1)][1]
        time = locals()[k+'__'+str(mod+1)][2]

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
        locals()['plot_levels'+str(mod+1)] = locals()['ta__'+str(mod+1)][3]
        locals()['grid_'+k+str(mod+1)] = []

        levels = locals()['plot_levels'+str(mod+1)]

        for p in range(len(levels)):
            if levels[p] == 70000:
                locals()[k+str(mod+1)] = locals()[k+'__'+str(mod+1)][4]
                locals()[k+str(mod+1)] = np.ma.filled(locals()[k+str(mod+1)], fill_value=np.nan)
                temp_700   = np.array(np.multiply(maskm,locals()[k+str(mod+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                break;

    theta_700 = temp_700*(100000/70000)**con
    theta_t2m = locals()['grid_tas'+str(mod+1)]*(100000/locals()['grid_psl'+str(mod+1)])**con

    # t = min(len(theta_t2m),len(theta_700))
    M_700  = theta_t2m - theta_700
    M_plot.extend(M_700)

########################### 0,3
    time1=[2011, 12, 1]
    time2=[2012, 2, 28]

    for i in varname:
        locals()[i+'__'+str(mod+1)] = read_var_mod('surface', warming_modname[mod], 'historical', i, time1, time2)

    for k in pvarname:
        locals()[k+'__'+str(mod+1)] = read_var_mod('p_level', warming_modname[mod], 'historical', k, time1, time2)


    for j in varname:
        lat  = locals()[j+'__'+str(mod+1)][0]
        lon  = locals()[j+'__'+str(mod+1)][1]
        time = locals()[j+'__'+str(mod+1)][2]

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
        #print(j)
        locals()[j+str(mod+1)] = locals()[j+'__'+str(mod+1)][4]
        locals()[j+str(mod+1)] = np.ma.filled(locals()[j+str(mod+1)], fill_value=np.nan)
        locals()['plot_'+j+str(mod+1)] = np.array(np.multiply(maskm,locals()[j+str(mod+1)][:,lat_ind1[0]:lat_ind2[0],:]))
        locals()['grid_'+j+str(mod+1)] = locals()['plot_'+j+str(mod+1)]

    for k in pvarname:
        #print(k)
        lat  = locals()[k+'__'+str(mod+1)][0]
        lon  = locals()[k+'__'+str(mod+1)][1]
        time = locals()[k+'__'+str(mod+1)][2]

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
        locals()['plot_levels'+str(mod+1)] = locals()['ta__'+str(mod+1)][3]
        locals()['grid_'+k+str(mod+1)] = []

        levels = locals()['plot_levels'+str(mod+1)]

        for p in range(len(levels)):
            if levels[p] == 70000:
                locals()[k+str(mod+1)] = locals()[k+'__'+str(mod+1)][4]
                locals()[k+str(mod+1)] = np.ma.filled(locals()[k+str(mod+1)], fill_value=np.nan)
                temp_700   = np.array(np.multiply(maskm,locals()[k+str(mod+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                break;

    theta_700 = temp_700*(100000/70000)**con
    theta_t2m = locals()['grid_tas'+str(mod+1)]*(100000/locals()['grid_psl'+str(mod+1)])**con

    # t = min(len(theta_t2m),len(theta_700))
    M_700  = theta_t2m - theta_700
    M_plot.extend(M_700)

########################### 0,4
    time1=[2012, 12, 1]
    time2=[2012, 12, 30]

    for i in varname:
        locals()[i+'__'+str(mod+1)] = read_var_mod('surface', warming_modname[mod], 'historical', i, time1, time2)

    for k in pvarname:
        locals()[k+'__'+str(mod+1)] = read_var_mod('p_level', warming_modname[mod], 'historical', k, time1, time2)


    for j in varname:
        lat  = locals()[j+'__'+str(mod+1)][0]
        lon  = locals()[j+'__'+str(mod+1)][1]
        time = locals()[j+'__'+str(mod+1)][2]

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
        #print(j)
        locals()[j+str(mod+1)] = locals()[j+'__'+str(mod+1)][4]
        locals()[j+str(mod+1)] = np.ma.filled(locals()[j+str(mod+1)], fill_value=np.nan)
        locals()['plot_'+j+str(mod+1)] = np.array(np.multiply(maskm,locals()[j+str(mod+1)][:,lat_ind1[0]:lat_ind2[0],:]))
        locals()['grid_'+j+str(mod+1)] = locals()['plot_'+j+str(mod+1)]

    for k in pvarname:
        #print(k)
        lat  = locals()[k+'__'+str(mod+1)][0]
        lon  = locals()[k+'__'+str(mod+1)][1]
        time = locals()[k+'__'+str(mod+1)][2]

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
        locals()['plot_levels'+str(mod+1)] = locals()['ta__'+str(mod+1)][3]
        locals()['grid_'+k+str(mod+1)] = []

        levels = locals()['plot_levels'+str(mod+1)]

        for p in range(len(levels)):
            if levels[p] == 70000:
                locals()[k+str(mod+1)] = locals()[k+'__'+str(mod+1)][4]
                locals()[k+str(mod+1)] = np.ma.filled(locals()[k+str(mod+1)], fill_value=np.nan)
                temp_700   = np.array(np.multiply(maskm,locals()[k+str(mod+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                break;

    theta_700 = temp_700*(100000/70000)**con
    theta_t2m = locals()['grid_tas'+str(mod+1)]*(100000/locals()['grid_psl'+str(mod+1)])**con

    # t = min(len(theta_t2m),len(theta_700))
    M_700  = theta_t2m - theta_700
    M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    DJF.append(f)
    DJF_lats.append(lats)

    plt.plot(lats,f,color=use_colors[mod],label=warming_modname[mod],linestyle='solid')


    # plt.legend()
    plt.xlabel('latitude')
    plt.title('seasonal CAO zonal mean ditribution\nhistorical and warming')
    plt.savefig('../figures/final/warming_M_lats.png')
