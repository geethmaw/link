# @Author: Geethma Werapitiya <wgeethma>
# @Date:   2022-07-08T15:00:59-06:00
# @Email:  wgeethma@uwyo.edu
# @Last modified by:   wgeethma
# @Last modified time: 2022-07-08T15:22:26-06:00

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from myReadGCMsDaily import read_var_mod
import calendar
from global_land_mask import globe
import glob
import math
from regrid_wght_3d import regrid_wght_wnans
from scipy import stats
import os
import netCDF4 as nc

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 100

from con_models import get_cons
con, use_colors, varname, pvarname, modname, warming_modname, hiresmd = get_cons()

#latitude range
latr1 = 30
latr2 = 80

l = 0
m = len(modname)   #l+1

time1=[2010, 1, 1]
time2=[2012, 12, 30]

lats_edges = np.arange(latr1,latr2+1,5)
lons_edges = np.arange(-180,181,5)

#binning
n_bins  = 20
M_range = (-20,5)

################################################################################################
for i in varname:
    locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

for k in pvarname:
    locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

print('done')

for i in range(l,m):
    print(modname[i])

    for j in varname:
        lat  = locals()[j+'__'+str(i+1)][0]
        lon  = locals()[j+'__'+str(i+1)][1]
        time = locals()[j+'__'+str(i+1)][2]

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
        print(j)
        locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
        locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
        locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
        locals()['grid_'+j+str(i+1)] = regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

    for k in pvarname:
        print(k)
        lat  = locals()[k+'__'+str(i+1)][0]
        lon  = locals()[k+'__'+str(i+1)][1]
        time = locals()[k+'__'+str(i+1)][2]

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
        locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
        locals()['grid_'+k+str(i+1)] = []

        levels = locals()['plot_levels'+str(i+1)]

        for p in range(len(levels)):
            if levels[p] == 85000:
                print(levels[p])
                locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                grid_t_700 = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)[0]
                break;

    theta_700 = grid_t_700*(100000/85000)**con
    theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

    M_700  = theta_t2m - theta_700
    plot_M = M_700.flatten()
    plot_W = locals()['grid_sfcWind'+str(i+1)].flatten()

    ind = np.argsort(plot_M)

    final_M = np.sort(plot_M)
    final_W = plot_W[ind]

    indx = np.isnan(final_M*final_W)==False

    bin_means, bin_edges, binnumber       = stats.binned_statistic(final_M[indx], final_W[indx], 'mean', bins=n_bins,range=M_range)
    bin_means_c, bin_edges_c, binnumber_c = stats.binned_statistic(final_M[indx], final_W[indx], 'count', bins=n_bins,range=M_range)
    bin_means_s, bin_edges_s, binnumber_s = stats.binned_statistic(final_M[indx], final_W[indx], 'std', bins=n_bins,range=M_range)
    bin_means_x, bin_edges_x, binnumber_x = stats.binned_statistic(final_M[indx], final_M[indx], 'mean', bins=n_bins,range=M_range)

    ind_c = np.where(bin_means_c > 1000)
    M_cesm2 = np.ma.masked_invalid(bin_means_x[ind_c])
    W_cesm2 = np.ma.masked_invalid(bin_means[ind_c])
    b_cesm2 = len(M_cesm2)

#################################################################################
enn = (np.arange(0,262)).tolist()
enn.remove(175)

u = 20
v = u+20
print(len(enn),v)

M_plot = []
W_plot = []
b_coun  = []

for en in enn[u:v]:
    print(en)
    for i in ['U10', 'PSL', 'TREFHT', 'T850']:
        d_path = '/glade/campaign/cgd/projects/ppe/cam_ppe/rerun_PPE_250/PD/PD_timeseries/PPE_250_ensemble_PD.'+f'{en:03d}'+'/atm/hist/cc_PPE_250_ensemble_PD.'+f'{en:03d}'+'.h1.'+str(i)+'.nc'
        data =xr.open_dataset(d_path)
        
        lon  = data.variables['lon'][:]  #(lon: 288) [0.0, 1.25, 2.5, ... 356.25, 357.5, 358.75]
        lat  = data.variables['lat'][:]  #(lat: 192) [-90.0 , -89.057592, -88.115183, ... 88.115183,  89.057592, 90.0]
        time = data.variables['time'][:] #(time: 36)
#             
        locals()[str(en)+'_'+i] = data.variables[i][:]
    
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
        
        tmp  = locals()[str(en)+'_'+i]
        tmp2 = tmp[:,lat_ind1[0]:lat_ind2[0],:]
        MID  = np.multiply(maskm,tmp2)
        lats = np.array(lats)
        lon  = np.array(lon)
        MID  = np.array(MID)
        locals()['MID'+i+'_'+str(en)] = regrid_wght_wnans(lats,lon,MID,lats_edges,lons_edges)[0]
        
    theta_850_en = np.multiply(locals()['MIDT850_'+str(en)],(100000/85000)**con)
    theta_T2M_en = np.multiply(locals()['MIDTREFHT_'+str(en)],(100000/locals()['MIDPSL_'+str(en)])**con)

    M_en   = np.array(np.subtract(theta_T2M_en,theta_850_en)).reshape(-1)
    U10_en = np.array(locals()['MIDU10_'+str(en)]).reshape(-1)
    
    indx = np.isnan(M_en*U10_en)==False

    bin_means, bin_edges, binnumber       = stats.binned_statistic(M_en[indx], U10_en[indx], 'mean', bins=n_bins,range=M_range)
    bin_means_c, bin_edges_c, binnumber_c = stats.binned_statistic(M_en[indx], U10_en[indx], 'count', bins=n_bins,range=M_range)
    #bin_means_s, bin_edges_s, binnumber_s = stats.binned_statistic(final_M[indx], final_W[indx], 'std', bins=n_bins,range=M_range)
    bin_means_x, bin_edges_x, binnumber_x = stats.binned_statistic(M_en[indx], M_en[indx], 'mean', bins=n_bins,range=M_range)

    ind_c = np.where(bin_means_c > 1000)
    # print(bin_means_c)
    # for x in bin_means_c:
    #     if x
    #std_err = bin_means_s/np.sqrt(bin_means_c)
    M_plot.append(np.ma.masked_invalid(bin_means_x[ind_c])) #[ind_c]
    W_plot.append(np.ma.masked_invalid(bin_means[ind_c]))
    #W_erro.append(np.ma.masked_invalid(std_err[ind_c]))
    b_coun.append(len(M_plot[en-u]))
print(v)
    
bin_count = min(np.min(b_coun),b_cesm2)
plt.clf()
plt.rcParams['figure.figsize'] = (15.0/2.5, 15.0/2.5)

for i in range(0,len(M_plot)):
    plt.plot(M_plot[i][0:bin_count], W_plot[i][0:bin_count])
    plt.annotate(xy=(M_plot[i][bin_count-1],W_plot[i][bin_count-1]), text=str(i+u),fontsize=6,color='blue')
        
plt.plot(M_cesm2[0:bin_count], W_cesm2[0:bin_count], label='CESM2', color='black', linestyle='--', linewidth=2)
plt.annotate(xy=(M_cesm2[bin_count-1]+1,W_cesm2[bin_count-1]), text='CESM2',color='black',fontsize=7)
        
plt.fill_between([-0.7,2.5],[5,5],[14,14],color='grey',alpha=0.2)
plt.annotate(xy=(-0.5,6), text='Unstable\nCAO',color='grey',fontsize=9)
        
plt.ylabel('U10 [m/s]')
#plt.xlim(-20,0)
yti = '850'
plt.xlabel(r"M ($\Theta_{t2m}$ - $\Theta_{"+yti+"})$ [K]")
plt.title('U10 vs M for oceans between '+str(latr1)+'N to '+str(latr2)+'N\nCESM2 and CAM6 PPE ('+str(u)+'-'+str(u+20)+')')
plt.savefig('../figures/parts'+str(u)+'_PPE_U10vsM.png')