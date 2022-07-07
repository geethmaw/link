# @Author: Geethma Werapitiya <wgeethma>
# @Date:   2022-06-16T14:41:09-06:00
# @Email:  wgeethma@uwyo.edu
# @Last modified by:   wgeethma
# @Last modified time: 2022-06-16T16:31:45-06:00

import netCDF4 as nc
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from myReadGCMsDaily import read_var_mod
import calendar
from global_land_mask import globe
import glob
import math
from scipy import stats
import glob

#####Constants
Cp = 1004           #J/kg/K
Rd = 287            #J/kg/K
con= Rd/Cp

latr1 = 30
latr2 = 80

#pressure levels
p_level = 1

####OBSERVATIONS############################################
# merlist = np.sort(glob.glob('../data_merra/all_lat_lon/levels/MERRA2_*.nc'))
# sfclist = np.sort(glob.glob('../data_merra/all_lat_lon/surface_old/MERRA2_*.nc'))
# maclist = np.sort(glob.glob('../MACLWP_dailymean/take/wind1deg*.nc4'))
#
# p_mer_T   = []
# p_mac_w   = []
# sfc_mer_T = []
# sfc_mer_P = []
#
# for i in range(len(merlist)): #len(merlist)
#     d_path = merlist[i]
#     data   = nc.Dataset(d_path)
#     # print(d_path)
#
#     if i==0:
#         merlat = data.variables['lat'][:]
#         merlon = data.variables['lon'][:]
#         merlev = data.variables['lev'][:]
#         #shape latitude
#         mer_lat = np.flip(merlat)
#         mer_lat = np.array(mer_lat)
#         mlat_ind1 = np.where(mer_lat == mer_lat.flat[np.abs(mer_lat - (latr1)).argmin()])[0]
#         mlat_ind2 = np.where(mer_lat == mer_lat.flat[np.abs(mer_lat - (latr2)).argmin()])[0]
#         p_mer_lat  = np.array(mer_lat[mlat_ind1[0]:mlat_ind2[0]])
#         #shape longitude
#         merlon[merlon > 180] = merlon[merlon > 180]-360
#         # mer_lon = np.array(merlon)
#
#     merT   = data.variables['T'][:] #(time, lev, lat, lon)
#     mer_T = np.array(merT[:,:,::-1,:])
#     p_mer_T.extend(mer_T[:,:,mlat_ind1[0]:mlat_ind2[0],:])
#
# temp = np.array(p_mer_T)
#
# for i in range(len(sfclist)): #len(merlist)
#     s_path = sfclist[i]
#     sdata  = nc.Dataset(s_path)
#     # print(d_path)
#
#     if i==0:
#         sfclat = sdata.variables['lat'][:]
#         sfclon = sdata.variables['lon'][:]
#         #shape latitude
#         sfc_lat = np.flip(sfclat)
#         sfc_lat = np.array(sfc_lat)
#         flat_ind1 = np.where(sfc_lat == sfc_lat.flat[np.abs(sfc_lat - (latr1)).argmin()])[0]
#         flat_ind2 = np.where(sfc_lat == sfc_lat.flat[np.abs(sfc_lat - (latr2)).argmin()])[0]
#         p_sfc_lat = np.array(sfc_lat[flat_ind1[0]:flat_ind2[0]])
#         #shape longitude
#         sfclon[sfclon > 180] = sfclon[sfclon > 180]-360
#         # sfc_lon = np.array(sfclon)
#
#     sfcT   = sdata.variables['TS'][:]
#     sfc_T = np.array(sfcT[:,::-1,:])
#     sfc_mer_T.extend(sfc_T[:,flat_ind1[0]:flat_ind2[0],:])
#
#     sfcP   = sdata.variables['SLP'][:]
#     sfc_P  = np.array(sfcP[:,::-1,:])
#     sfc_mer_P.extend(sfc_P[:,flat_ind1[0]:flat_ind2[0],:])
#
# sfctemp = np.array(sfc_mer_T)
# sfcpres = np.array(sfc_mer_P)
#
# for i in range(len(maclist)): #len(maclist)
#     ddpath = maclist[i]
#     ddata  = nc.Dataset(ddpath)
#     macw   = ddata.variables['sfcwind'][:] #(time,lat,lon)
#     # print(ddpath)
#
#     if i==0:
#         maclat = ddata.variables['lat'][:]
#         maclon = ddata.variables['lon'][:]
#         #shape latitude
#         mac_lat = np.array(maclat)
#         slat_ind1 = np.where(mac_lat == mac_lat.flat[np.abs(mac_lat - (latr1)).argmin()])[0]
#         slat_ind2 = np.where(mac_lat == mac_lat.flat[np.abs(mac_lat - (latr2)).argmin()])[0]
#         p_mac_lat  = np.array(mac_lat[slat_ind1[0]:slat_ind2[0]])
#         #shape longitude
#         maclon[maclon > 180] = maclon[maclon > 180]-360
#         # mac_lon = np.array(maclon)
#
#     n_w = macw[:,slat_ind1[0]:slat_ind2[0],:]
#     p_mac_w.extend(n_w)
#
# #reshaping longitudes
# mer_lon = []
# mer_lon.extend(merlon[180:360])
# mer_lon.extend(merlon[0:180])
#
# sfc_lon = []
# sfc_lon.extend(sfclon[180:360])
# sfc_lon.extend(sfclon[0:180])
#
# mac_lon = []
# mac_lon.extend(maclon[180:360])
# mac_lon.extend(maclon[0:180])
#
# wind   = np.array(p_mac_w)
#
# theta_800 = np.array(np.multiply(temp[:,p_level,:,:], (100000/(merlev[p_level]*100))**(Rd/Cp)))
# theta_sfc = np.array(np.multiply(sfctemp, (100000/sfcpres)**(Rd/Cp)))
#
# p_CAOI = np.array(np.subtract(theta_sfc,theta_800))
#
#
# #Mask for the ocean
# maskm = np.ones((len(temp),len(p_mer_lat),len(mer_lon)))
#
# for a in range(len(p_mer_lat)):
#     for b in range(len(mer_lon)):
#         if globe.is_land(p_mer_lat[a], mer_lon[b])==True:
#             maskm[:,a,b] = math.nan
# ##############################
# #reshaping M and wind
# caoi_test = p_CAOI
# wind_test = wind
#
# plot_CAOI = np.ones((len(temp),len(p_mer_lat),len(mer_lon)))
# plot_CAOI[:,:,180:360] = caoi_test[:,:,0:180]
# plot_CAOI[:,:,0:180]   = caoi_test[:,:,180:360]
# plot_CAOI = np.array(plot_CAOI)
#
# plot_wind = np.ones((len(temp),len(p_mer_lat),len(mer_lon)))
# plot_wind[:,:,180:360] = wind_test[:,:,0:180]
# plot_wind[:,:,0:180]   = wind_test[:,:,180:360]
# plot_wind = np.array(plot_wind)
#
# #ocean only mask
# plot_CAOI = np.array(np.multiply(maskm,plot_CAOI))
# plot_wind = np.array(np.multiply(maskm,plot_wind))
#
# plot_indx = np.isnan(plot_CAOI*plot_CAOI)==False
# plot_mer_theta = plot_CAOI[plot_indx]
# plot_mac_wind  = plot_wind[plot_indx]
# ###################################
#
# #Sort and removing nan values
# ind = np.argsort(plot_mer_theta)
# xx  = np.sort(plot_mer_theta)
# yy  = plot_mac_wind[ind]
#
# xx_new = xx[yy>0]
# yy_new = yy[yy>0]
#
# o_indx = np.isnan(xx_new*yy_new)==False
#
# o_M = xx_new[o_indx]
# o_W = yy_new[o_indx]
#
# o_W_bin_means, o_W_bin_edges, o_W_binnumber = stats.binned_statistic(o_M, o_W, 'mean', bins=100, range=(0,8))
# o_final_W = o_W_bin_means
#
#
# #############################################################
# ####GCMs#####################################################
modname = ['CESM2','CESM2-WACCM','HadGEM3-GC31-LL','HadGEM3-GC31-MM','IPSL-CM6A-LR','CNRM-ESM2-1','INM-CM5-0','MPI-ESM1-2-HR','UKESM1-0-LL','MPI-ESM1-2-LR','MPI-ESM-1-2-HAM','CMCC-CM2-SR5','CMCC-CM2-HR4','CMCC-ESM2','CNRM-CM6-1','CNRM-ESM2-1','IPSL-CM5A2-INCA']
varname = ['sfcWind', 'ts','psl'] #'sfcWind', 'hfss', 'hfls', 'tas', 'ps', 'psl',,'pr'
pvarname= ['ta']
#
l=0
m=len(modname)
#
time1=[2010, 1, 1]
time2=[2012, 12, 30]

for j in range(l,m): #
    print(modname[j])
    for i in varname:
        locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)
        # print(i)
    for k in pvarname:
        locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

# except IndexError:
#     print(str(j+1)+' not available')
print('done')

for i in range(l,m):
    lat  = locals()['sfcWind__'+str(i+1)][0]
    lon  = locals()['sfcWind__'+str(i+1)][1]
    time = locals()['sfcWind__'+str(i+1)][2]
#
#     for j in varname:
#         locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
#         locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
#         print(np.shape(locals()[j+str(i+1)]))
# #
#     for k in pvarname:
#         locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
#         locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
#         print(np.shape(locals()[k+str(i+1)]))
#
#         locals()['lev'+str(i+1)] = locals()['ta__'+str(i+1)][3]
#
    x_lat = np.array(lat)
#     lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
#     lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
#     lats = lat[lat_ind1[0]:lat_ind2[0]]
#
    x_lon = lon
#     lon = np.array(lon)
#     lon[lon > 180] = lon[lon > 180]-360
#
#     maskm = np.ones((len(time),len(lats),len(lon)))
#
#     for a in range(len(lats)):
#         for b in range(len(lon)):
#             if globe.is_land(lats[a], lon[b])==True:
#                 maskm[:,a,b] = math.nan
#
#  ###  averaged theta at 800hPa and surface
#     theta_850 = locals()['ta'+str(i+1)][:,1,:,:]*(100000/85000)**con
#     theta_700 = locals()['ta'+str(i+1)][:,2,:,:]*(100000/70000)**con
#     theta_800 = theta_700 + ((2/3) * (theta_850 - theta_700))
#
#     theta_sfc = locals()['ts'+str(i+1)]*(100000/locals()['psl'+str(i+1)])**con
#
# ### CAOI at 800hPa
#     M = theta_sfc - theta_800
#
#     x_sfcWind = locals()['sfcWind'+str(i+1)]
#     m_sfcWind = x_sfcWind[:,lat_ind1[0]:lat_ind2[0],:]
#     lats = lat[lat_ind1[0]:lat_ind2[0]]
#
#     x_M = M
#     m_M = x_M[:,lat_ind1[0]:lat_ind2[0],:]
#
#     cao = np.array(m_M)
#     sw  = np.array(m_sfcWind)
#
#     plot_CAOI = np.array(np.multiply(maskm,cao))
#     wind      = np.array(np.multiply(maskm,sw))
#
#     pl_theta  = plot_CAOI.reshape(-1)
#     pl_wind   = wind.reshape(-1)
#
#     plo_theta = pl_theta[pl_theta>-40]
#     plo_wind  = pl_wind[pl_theta>-40]
#
#     plot_theta = plo_theta[plo_theta<40]
#     plot_wind  = plo_wind[plo_theta<40]
#
#     ind = np.argsort(plot_theta)
#     xx = np.sort(plot_theta)
#     yy = plot_wind[ind]
#
#     g_indx = np.isnan(xx*yy)==False
#
#     g_M = xx[g_indx]
#     g_W = yy[g_indx]
#
#     #g_bin_means, g_bin_edges, g_binnumber = stats.binned_statistic(g_M, g_W, 'mean', bins=1000)
#     g_W_bin_means, g_W_bin_edges, g_W_binnumber = stats.binned_statistic(g_M, g_W, 'mean', bins=100, range=(0,8))
#
#     # g_index   = np.isnan(g_M_bin_means*g_W_bin_means)==False
#     g_final_W = g_W_bin_means #[g_index]
#
#     bias      = o_final_W - g_final_W
#     mean_bias = np.mean(bias)

    g_lat_diff = np.abs(x_lat[1]-x_lat[0])
    g_lon_diff = np.abs(x_lon[1]-x_lon[0])
    g_res      = (np.sqrt(g_lat_diff**2 + g_lon_diff**2)) * 110.574

    # plt.scatter(g_res,mean_bias)
    print(modname[i])
    print(g_res)

#############PLOT###############
# plt.ylabel('U10 Bias [m/s]',fontsize='15')
# plt.xlabel('Resolution [km]')
# plt.title('for oceans between '+str(latr1)+'N to '+str(latr2)+'N/nfor 0-8K M')
# plt.savefig('../figures/BiasVsRes'+str(latr1)+'N to '+str(latr2)+'N_800theta.png')
