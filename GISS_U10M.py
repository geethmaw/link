# @Author: Geethma Werapitiya <wgeethma>
# @Date:   2022-08-16T13:38:14-06:00
# @Email:  wgeethma@uwyo.edu
# @Last modified by:   wgeethma
# @Last modified time: 2022-08-18T09:53:14-06:00

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
import calendar
from global_land_mask import globe
import glob
import math
from scipy import stats
import os
import netCDF4 as nc

from gcm_loading.highres_read import read_var_hires
from gcm_loading.myReadGCMsDaily import read_var_mod
from gcm_loading.read_amip import read_amip_var
from regrid_wght_3d import regrid_wght_wnans

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 100
plt.clf()
plt.rcParams['figure.figsize'] = (10, 10)

from con_models import get_cons
con, use_colors, varname, pvarname, modname, warming_modname, hiresmd, amip_md = get_cons()
#latitude range
latr1 = 30
latr2 = 80

#UL pressure and unstable range
UL_pres = 850
M_range = (-30,20)

time1=[2010, 1, 1]
time2=[2012, 12, 30]

lats_edges = np.arange(latr1,latr2+1,5)
lons_edges = np.arange(-180,181,5)

#binning
n_bins  = 20
max_c   = 500
bin_count = 1000

j = 2
print('\n',amip_md[j])
# def read_amip_var(level, modn, exper, varnm, time1, time2):
for i in varname:
    if i!='ts':
        locals()[i+'__GISS'] = read_amip_var('surface', amip_md[j], 'amip', i, time1, time2)

for k in pvarname:
    locals()[k+'__GISS'] = read_amip_var('p_level', amip_md[j], 'amip', k, time1, time2)

print('amip done')

mod_time = sfcWind__GISS[2]

#####OBSERVATIONS PROCESSING###############
from obs_data_function import obs
merwind, macwind, temp, sfctemp, sfcpres, p_lev_obs, p_mer_lat, merlon, merlev, obs_time = obs('surface','TS',  UL_pres, latr1, latr2)

for i in varname:
    if i!='ts':
        locals()[i+'_mod'] = []

for i in pvarname:
    locals()[i+'_mod'] = []

macwind_new = []
temp_new    = []
sfctemp_new = []
sfcpres_new = []

ot = 0
mt = 0

while (ot < len(obs_time) and mt < len(mod_time)):
    date_o = obs_time[ot]
    date_m = str(mod_time[mt][0])+str(mod_time[mt][1]).zfill(2)+str(mod_time[mt][2]).zfill(2)

    if date_o==date_m:
        for i in varname:
            if i!='ts':
                locals()[i+'_mod'].append(locals()[i+'__GISS'][4][mt])

        for i in pvarname:
            locals()[i+'_mod'].append(locals()[i+'__GISS'][4][mt])
        macwind_new.append(macwind[ot,:,:])
        temp_new.append(temp[ot,:,:])
        sfctemp_new.append(sfctemp[ot,:,:])
        sfcpres_new.append(sfcpres[ot,:,:])

        ot = ot+1
        mt = mt+1

    elif date_o<date_m:
        ot = ot+1

    elif date_o>date_m:
        mt = mt+1

macwind_new = np.array(macwind_new)
temp_new    = np.array(temp_new)
sfctemp_new = np.array(sfctemp_new)
sfcpres_new = np.array(sfcpres_new)

grid_obs_wind     = regrid_wght_wnans(p_mer_lat,merlon,macwind_new,lats_edges,lons_edges)[0]
grid_obs_temp_700 = regrid_wght_wnans(p_mer_lat,merlon,temp_new,lats_edges,lons_edges)[0]
grid_obs_temp_sfc = regrid_wght_wnans(p_mer_lat,merlon,sfctemp_new,lats_edges,lons_edges)[0]
grid_obs_pres_sfc = regrid_wght_wnans(p_mer_lat,merlon,sfcpres_new,lats_edges,lons_edges)[0]

lat_n = regrid_wght_wnans(p_mer_lat,merlon,sfcpres_new,lats_edges,lons_edges)[2][:,0]
lon_n = regrid_wght_wnans(p_mer_lat,merlon,sfcpres_new,lats_edges,lons_edges)[1][0,:]

theta_700 = np.array(np.multiply(grid_obs_temp_700, (100000/(merlev[p_lev_obs]*100))**(con)))
theta_sfc = np.array(np.multiply(grid_obs_temp_sfc, (100000/grid_obs_pres_sfc)**(con)))
print('theta_800', np.shape(theta_700))
print('theta_sfc', np.shape(theta_sfc))


p_CAOI = np.array(np.subtract(theta_sfc,theta_700))

#Mask for the ocean
maskm = np.ones((len(temp),len(lat_n),len(lon_n)))

for a in range(len(lat_n)):
    for b in range(len(lon_n)):
        if globe.is_land(lat_n[a], lon_n[b])==True:
            maskm[:,a,b] = math.nan
##############################

#ocean only mask
plot_CAOI = np.array(np.multiply(maskm,p_CAOI))
plot_wind = np.array(np.multiply(maskm,grid_obs_wind))

plot_indx = np.isnan(plot_CAOI*plot_wind)==False
plot_mer_theta = plot_CAOI[plot_indx]
plot_mac_wind  = plot_wind[plot_indx]

w_sfc = plot_mac_wind[plot_mac_wind>0]
m_700 = plot_mer_theta[plot_mac_wind>0]
###################################

from scipy import stats
bin_means, bin_edges, binnumber       = stats.binned_statistic(m_700, w_sfc, 'mean', bins=n_bins, range=M_range)
bin_means_c, bin_edges_c, binnumber_c = stats.binned_statistic(m_700, w_sfc, 'count', bins=n_bins, range=M_range)
bin_means_s, bin_edges_s, binnumber_s = stats.binned_statistic(m_700, w_sfc, 'std', bins=n_bins, range=M_range)
bin_means_x, bin_edges_x, binnumber_x = stats.binned_statistic(m_700, m_700, 'mean', bins=n_bins, range=M_range)

W_SFC = np.ma.masked_invalid(bin_means[bin_means_c>max_c])
M_700 = np.ma.masked_invalid(bin_means_x[bin_means_c>max_c])
s_err = bin_means_s[bin_means_c>max_c]/np.sqrt(bin_means_c[bin_means_c>max_c])
W_err = np.ma.masked_invalid(s_err)

bin_count = min(bin_count, len(W_SFC))

##########GISS PROCESSING############
G = 2
print(amip_md[G])

M_plot = []
W_plot = []
W_erro = []
bin_count = 1000
UL_pres = 850
M_range = (-30,20)

for j in varname:
    if j!='ts':
        lat  = locals()[j+'__GISS'][0]
        lon  = locals()[j+'__GISS'][1]
        time = locals()[j+'__GISS'][2]

        x_lat = np.array(lat)
        lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
        lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
        lats = lat[lat_ind1[0]:lat_ind2[0]]

        x_lon = lon[:]
        lon = np.array(lon)
        lon[lon > 180] = lon[lon > 180]-360

        maskm = np.ones((len(obs_time),len(lats),len(lon)))

        for a in range(len(lats)):
            for b in range(len(lon)):
                if globe.is_land(lats[a], lon[b])==True:
                    maskm[:,a,b] = math.nan
        #print(j)
        locals()[j+'_GISS'] = locals()[j+'_mod']
        locals()[j+'_GISS'] = np.ma.filled(locals()[j+'_GISS'], fill_value=np.nan)
        locals()['plot_'+j+'_GISS'] = np.array(np.multiply(maskm,locals()[j+'_GISS'][:,lat_ind1[0]:lat_ind2[0],:]))
        locals()['grid_'+j+'_GISS'] = regrid_wght_wnans(lats,lon,locals()['plot_'+j+'_GISS'],lats_edges,lons_edges)[0]

for k in pvarname:
    #print(k)
    lat  = locals()[k+'__GISS'][0]
    lon  = locals()[k+'__GISS'][1]
    time = locals()[k+'__GISS'][2]

    x_lat = np.array(lat)
    lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
    lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
    lats = lat[lat_ind1[0]:lat_ind2[0]]

    x_lon = lon
    lon = np.array(lon)
    lon[lon > 180] = lon[lon > 180]-360

    maskm = np.ones((len(obs_time),len(lats),len(lon)))

    for a in range(len(lats)):
        for b in range(len(lon)):
            if globe.is_land(lats[a], lon[b])==True:
                maskm[:,a,b] = math.nan
    locals()['plot_levels_GISS'] = ta__GISS[3]
    locals()['grid_'+k+'_GISS'] = []

    levels = plot_levels_GISS

    for p in range(len(levels)):
        if levels[p] == UL_pres*100:
            print(levels[p])
            locals()[k+'_GISS'] = locals()[k+'_mod']
            locals()[k+'_GISS'] = np.ma.filled(locals()[k+'_GISS'], fill_value=np.nan)
            temp_700   = np.array(np.multiply(maskm,locals()[k+'_GISS'][:,p,lat_ind1[0]:lat_ind2[0],:]))
            grid_t_700 = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)[0]
            break;

theta_700 = grid_t_700*(100000/ (UL_pres*100))**con
theta_t2m = grid_obs_temp_sfc*(100000/grid_psl_GISS)**con

t = min(len(theta_t2m),len(theta_700))
M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
plot_M = M_700.flatten()
try_W  = grid_sfcWind_GISS[0:t,:,:]
plot_W = try_W.flatten()

ind = np.argsort(plot_M)

final_M = np.sort(plot_M)
final_W = plot_W[ind]

indx = np.isnan(final_M*final_W)==False

bin_means, bin_edges, binnumber       = stats.binned_statistic(final_M[indx], final_W[indx], 'mean', bins=n_bins,range=M_range)
bin_means_c, bin_edges_c, binnumber_c = stats.binned_statistic(final_M[indx], final_W[indx], 'count', bins=n_bins,range=M_range)
bin_means_s, bin_edges_s, binnumber_s = stats.binned_statistic(final_M[indx], final_W[indx], 'std', bins=n_bins,range=M_range)
bin_means_x, bin_edges_x, binnumber_x = stats.binned_statistic(final_M[indx], final_M[indx], 'mean', bins=n_bins,range=M_range)

ind_c = np.where(bin_means_c > max_c)

std_err = bin_means_s/np.sqrt(bin_means_c)
M_plot.append(np.ma.masked_invalid(bin_means_x[ind_c])) #[ind_c]
W_plot.append(np.ma.masked_invalid(bin_means[ind_c]))
W_erro.append(np.ma.masked_invalid(std_err[ind_c]))
bin_count = min(bin_count,len(bin_means_c[ind_c]))

####PLOTS#####
plt.clf()
plt.plot(M_700[0:bin_count], W_SFC[0:bin_count], color='black' ,linestyle='dashed',
     markersize=5,linewidth=2, label='Observations')
# plt.fill_between(M_700[0:bin_count], W_SFC[0:bin_count]-W_err[0:bin_count], W_SFC[0:bin_count]+W_err[0:bin_count], color='black' ,alpha=0.2)
plt.annotate(xy=(M_700[bin_count-1],W_SFC[bin_count-1]), text='Observations', color='black',fontsize=7)

plt.plot(M_plot[0][0:bin_count], W_plot[0][0:bin_count], label='GISS-E3-G', color=use_colors[i])
# plt.fill_between(M_plot[i][0:bin_count], W_plot[i][0:bin_count]-W_erro[i][0:bin_count], W_plot[i][0:bin_count]+W_erro[i][0:bin_count], color=use_colors[i], alpha=0.2)
plt.annotate(xy=(M_plot[0][bin_count-1],W_plot[0][bin_count-1]), text='GISS-E3-G',color=use_colors[i],fontsize=7)
    # print(i)

# plt.fill_between([-8.6,-5.3],[5,5],[13,13],color='grey',alpha=0.2)
# plt.annotate(xy=(-8.3,12.5), text='Unstable CAOI',color='grey',fontsize=9)

plt.ylabel('U10 [m/s]')
# plt.xlim(-20,0)
yti = str(UL_pres)
plt.xlabel(r"M ($\Theta_{SST}$ - $\Theta_{"+yti+"})$ [K]")
plt.title('U10 vs M for oceans between '+str(latr1)+'N to '+str(latr2)+'N')
plt.savefig('../figures/GISS-E3-G_'+str(UL_pres)+'_sst_U10vsM.png')
