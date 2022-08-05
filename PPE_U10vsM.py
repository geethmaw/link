# @Author: Geethma Werapitiya <wgeethma>
# @Date:   2022-07-08T15:00:59-06:00
# @Email:  wgeethma@uwyo.edu
# @Last modified by:   wgeethma
# @Last modified time: 2022-07-19T04:07:38-06:00

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
plt.clf()
mpl.rcParams['figure.dpi'] = 100
plt.gcf().set_size_inches(6.4, 4.8)

from con_models import get_cons
con, use_colors, varname, pvarname, modname, warming_modname, hiresmd, amip = get_cons()

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
max_c   = 500
M_range = (-20,5)

################################################################################################
t = 0
for i in varname:
    locals()[i] = read_var_mod('surface', modname[t], 'historical', i, time1, time2)

for k in pvarname:
    locals()[k] = read_var_mod('p_level', modname[t], 'historical', k, time1, time2)

print('done')


for j in varname:
    lat  = locals()[j][0]
    lon  = locals()[j][1]
    time = locals()[j][2]

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
    # print(j)
    locals()[j+str(1)] = locals()[j][4]
    locals()[j+str(1)] = np.ma.filled(locals()[j+str(1)], fill_value=np.nan)
    locals()['plot_'+j] = np.array(np.multiply(maskm,locals()[j+str(1)][:,lat_ind1[0]:lat_ind2[0],:]))
    locals()['grid_'+j] = regrid_wght_wnans(lats,lon,locals()['plot_'+j],lats_edges,lons_edges)[0]

for k in pvarname:
    # print(k)
    lat  = locals()[k][0]
    lon  = locals()[k][1]
    time = locals()[k][2]

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
    locals()['plot_levels'+str(1)] = ta[3]
    locals()['grid_'+k+str(1)] = []

    levels = locals()['plot_levels'+str(1)]

    for p in range(len(levels)):
        if levels[p] == 85000:
            # print(levels[p])
            locals()[k+str(1)] = locals()[k][4]
            locals()[k+str(1)] = np.ma.filled(locals()[k+str(1)], fill_value=np.nan)
            temp_700   = np.array(np.multiply(maskm,locals()[k+str(1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
            grid_t_700 = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)[0]
            break;

theta_700 = grid_t_700*(100000/85000)**con
theta_t2m = grid_tas*(100000/grid_psl)**con

M_700  = theta_t2m - theta_700
plot_M = M_700.flatten()
plot_W = grid_sfcWind.flatten()

ind = np.argsort(plot_M)

final_M = np.sort(plot_M)
final_W = plot_W[ind]

indx = np.isnan(final_M*final_W)==False

bin_means, bin_edges, binnumber       = stats.binned_statistic(final_M[indx], final_W[indx], 'mean', bins=n_bins,range=M_range)
bin_means_c, bin_edges_c, binnumber_c = stats.binned_statistic(final_M[indx], final_W[indx], 'count',bins=n_bins,range=M_range)
bin_means_s, bin_edges_s, binnumber_s = stats.binned_statistic(final_M[indx], final_W[indx], 'std',  bins=n_bins,range=M_range)
bin_means_x, bin_edges_x, binnumber_x = stats.binned_statistic(final_M[indx], final_M[indx], 'mean', bins=n_bins,range=M_range)

ind_c     = np.where(bin_means_c > max_c)
M_cesm2   = np.ma.masked_invalid(bin_means_x[ind_c])
W_cesm2   = np.ma.masked_invalid(bin_means[ind_c])
b_cesm2   = len(M_cesm2)
bin_count = b_cesm2

UM_range = (-10, 5)

bin_means, bin_edges, binnumber       = stats.binned_statistic(final_M[indx], final_W[indx], 'mean', bins=n_bins,range=UM_range)
bin_means_c, bin_edges_c, binnumber_c = stats.binned_statistic(final_M[indx], final_W[indx], 'count',bins=n_bins,range=UM_range)
bin_means_s, bin_edges_s, binnumber_s = stats.binned_statistic(final_M[indx], final_W[indx], 'std',  bins=n_bins,range=UM_range)
bin_means_x, bin_edges_x, binnumber_x = stats.binned_statistic(final_M[indx], final_M[indx], 'mean', bins=n_bins,range=UM_range)

ind_c     = np.where(bin_means_c > max_c)
UM_cesm2  = np.ma.masked_invalid(bin_means_x[ind_c])
UW_cesm2  = np.ma.masked_invalid(bin_means[ind_c])

#################################################################################
enn = (np.arange(0,262)).tolist()
enn.remove(175)

u = 0
v = len(enn)
# print(len(enn),v)

M_ppe    = []
W_ppe    = []
UM_ppe   = []
UW_ppe   = []

for en in enn[u:v]:
    # print(en)
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

    bin_means, bin_edges, binnumber       = stats.binned_statistic(M_en[indx], U10_en[indx], 'mean',  bins=n_bins,range=M_range)
    bin_means_c, bin_edges_c, binnumber_c = stats.binned_statistic(M_en[indx], U10_en[indx], 'count', bins=n_bins,range=M_range)
    bin_means_x, bin_edges_x, binnumber_x = stats.binned_statistic(M_en[indx], M_en[indx], 'mean',    bins=n_bins,range=M_range)

    ind_c = np.where(bin_means_c > max_c)

    M_ppe.append(np.ma.masked_invalid(bin_means_x[ind_c])) #[ind_c]
    W_ppe.append(np.ma.masked_invalid(bin_means[ind_c]))

    bin_count = min(bin_count, len(bin_means_c[ind_c]))

    bin_means, bin_edges, binnumber       = stats.binned_statistic(M_en[indx], U10_en[indx], 'mean',  bins=n_bins,range=UM_range)
    bin_means_c, bin_edges_c, binnumber_c = stats.binned_statistic(M_en[indx], U10_en[indx], 'count', bins=n_bins,range=UM_range)
    bin_means_x, bin_edges_x, binnumber_x = stats.binned_statistic(M_en[indx], M_en[indx], 'mean',    bins=n_bins,range=UM_range)

    ind_c = np.where(bin_means_c > max_c)

    UM_ppe.append(np.ma.masked_invalid(bin_means_x[ind_c])) #[ind_c]
    UW_ppe.append(np.ma.masked_invalid(bin_means[ind_c]))

header = ['M_ppe','W_ppe','UM_ppe','UW_ppe']

import csv
with open('csv/PPE_M_W.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerow(M_ppe)
    writer.writerow(W_ppe)
    writer.writerow(UM_ppe)
    writer.writerow(UW_ppe)

## OBSERVATIONS
from obs_data_function import obs
merwind, macwind, temp, sfctemp, sfcpres, p_lev_obs, p_mer_lat, merlon, merlev = obs('surface','T2M', 850, latr1, latr2)

grid_obs_wind     = regrid_wght_wnans(p_mer_lat,merlon,macwind,lats_edges,lons_edges)[0]
grid_obs_temp_700 = regrid_wght_wnans(p_mer_lat,merlon,temp,lats_edges,lons_edges)[0]
grid_obs_temp_sfc = regrid_wght_wnans(p_mer_lat,merlon,sfctemp,lats_edges,lons_edges)[0]
grid_obs_pres_sfc = regrid_wght_wnans(p_mer_lat,merlon,sfcpres,lats_edges,lons_edges)[0]

lat_n = regrid_wght_wnans(p_mer_lat,merlon,sfcpres,lats_edges,lons_edges)[2][:,0]
lon_n = regrid_wght_wnans(p_mer_lat,merlon,sfcpres,lats_edges,lons_edges)[1][0,:]

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
M_OBS = np.ma.masked_invalid(bin_means_x[bin_means_c>max_c])

bin_count = min(bin_count, len(W_SFC))

bin_means, bin_edges, binnumber       = stats.binned_statistic(m_700, w_sfc, 'mean',  bins=n_bins, range=UM_range)
bin_means_c, bin_edges_c, binnumber_c = stats.binned_statistic(m_700, w_sfc, 'count', bins=n_bins, range=UM_range)
bin_means_s, bin_edges_s, binnumber_s = stats.binned_statistic(m_700, w_sfc, 'std',   bins=n_bins, range=UM_range)
bin_means_x, bin_edges_x, binnumber_x = stats.binned_statistic(m_700, m_700, 'mean',  bins=n_bins, range=UM_range)

UW_SFC = np.ma.masked_invalid(bin_means[bin_means_c>max_c])
UM_OBS = np.ma.masked_invalid(bin_means_x[bin_means_c>max_c])


plt.clf()
mpl.rcParams['figure.dpi'] = 100
plt.gcf().set_size_inches(6.4, 4.8)

# unstable CAOs plot
plt.fill_between([-10,0],[5,5],[14,14],color='grey',alpha=0.2)
plt.annotate(xy=(-7.5,11), text='Unstable CAO',color='grey',fontsize=9)

#PPE plot
for i in np.arange(0,len(M_ppe)): #,20
    plt.plot(M_ppe[i][0:bin_count], W_ppe[i][0:bin_count], color='red', alpha=0.05)

plt.annotate(xy=(M_cesm2[bin_count-1]+1,11.5), text='PPE',color='red',fontsize=9)

# CESM2 plot
plt.plot(M_cesm2[0:bin_count], W_cesm2[0:bin_count], label=modname[t], color='darkblue', linestyle='--', linewidth=2)
plt.scatter(M_cesm2[0:bin_count], W_cesm2[0:bin_count], color='darkblue', marker='x')
plt.annotate(xy=(M_cesm2[bin_count-1]+1,W_cesm2[bin_count-1]), text=modname[t],color='darkblue',fontsize=9)

# observations plot
plt.plot(M_OBS[0:bin_count], W_SFC[0:bin_count], color='black',linestyle='dashed',linewidth=3) #, color='black' ,markersize=5, label='Observations')
plt.annotate(xy=(M_OBS[bin_count-1],W_SFC[bin_count-1]), text='Observations', color='black',fontsize=9)

plt.ylabel('U10 [m/s]')
#plt.xlim(-20,0)
yti = '850'
plt.xlabel(r"M ($\Theta_{t2m}$ - $\Theta_{"+yti+"})$ [K]")
plt.title('U10 vs M for CAM6 PPE,\nCESM2 and obs')
plt.savefig('../figures/final/PPE.png')


ppe_bias = []

for i in range(0,len(UM_ppe)): #len(modname)
    bias      = UW_SFC - UW_ppe[i]
    mean_bias = np.nanmean(bias)
    ppe_bias.append(mean_bias)

import csv
# with open('csv/PPE_bias.csv', 'w', encoding='UTF8') as f:
#     writer = csv.writer(f)
#     writer.writecolum(ppe_bias)

with open('csv/PPE_bias.csv', 'w', ) as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for word in ppe_bias:
        wr.writerow([word])

headerList = [g_res[0]]

# open CSV file and assign header
with open("csv/PPE_bias.csv", 'w') as file:
    dw = csv.DictWriter(file, delimiter=',',
                        fieldnames=headerList)
    dw.writeheader()
    wr = csv.writer(file, quoting=csv.QUOTE_ALL)
    for word in ppe_bias:
        wr.writerow([word])

dataframe = pd.DataFrame(ppe_bias, columns=[g_res[0]])
dataframe.boxplot(grid='false', color='blue',fontsize=10, rot=30 )

df = pd.DataFrame(ppe_bias, columns=[g_res[0]])
df.head()

df.plot.box()

PPE_std  = np.std(ppe_bias)
PPE_bias = np.mean(ppe_bias)
