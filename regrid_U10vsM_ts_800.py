# @Author: Geethma Werapitiya <wgeethma>
# @Date:   2022-07-01T12:00:32-06:00
# @Email:  wgeethma@uwyo.edu
# @Last modified by:   geethmawerapitiya
# @Last modified time: 2022-07-06T00:31:53-06:00


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
from regrid_wght_3d import regrid_wght_wnans
from scipy import stats

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

lats_edges = np.arange(latr1,latr2+1,5)
lons_edges = np.arange(-180,181,5)

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

for i in range(l,1):

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
        locals()['grid_'+j+str(i+1)] = regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]



    for k in pvarname:
        locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
        locals()['grid_'+k+str(i+1)] = []

        levels = locals()['plot_levels'+str(i+1)]

        for p in range(len(levels)):
            locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
            locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
            plev  = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
            grid_plev = regrid_wght_wnans(lats,lon,plev,lats_edges,lons_edges)
            locals()['grid_'+k+str(i+1)].append(grid_plev[0])


    temp_800  = log_interpolate_1d(80000, levels,np.array(locals()['grid_ta'+str(i+1)]), axis=0, fill_value=np.nan)
    theta_800 = temp_800*(100000/80000)**con
    theta_sfc = locals()['grid_ts'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

    M_800  = theta_sfc - theta_800[0,:,:,:]
    plot_M = M_800.flatten()
    plot_W = locals()['grid_sfcWind'+str(i+1)].flatten()

    ind = np.argsort(plot_M)

    final_M = np.sort(plot_M)
    final_W = plot_W[ind]

    indx = np.isnan(final_M*final_W)==False

    bin_means, bin_edges, binnumber       = stats.binned_statistic(final_M[indx], final_W[indx], 'mean', bins=100,range=(-20,40))
    # bin_means, bin_edges, binnumber       = stats.binned_statistic(final_M[indx], final_W[indx], 'mean', bins=100,range=(-20,40))
    bin_means_x, bin_edges_x, binnumber_x = stats.binned_statistic(final_M[indx], final_M[indx], 'mean', bins=100,range=(-20,40))

    plt.plot(bin_means_x, bin_means, label=modname[i])

    std_err = stats.sem(final_W[indx])
    plt.fill_between(bin_means_x, bin_means-std_err, bin_means+std_err, alpha=0.2)


## OBSERVATIONS
import glob
merlist = np.sort(glob.glob('../data_merra/all_lat_lon/levels/MERRA2_*.nc'))
sfclist = np.sort(glob.glob('../data_merra/all_lat_lon/surface_old/MERRA2_*.nc'))
maclist = np.sort(glob.glob('../MACLWP_dailymean/take/wind1deg*.nc4'))

p_level = 1

import netCDF4 as nc
import xarray as xr
p_mer_T   = []
p_mac_w   = []
sfc_mer_T = []
sfc_mer_P = []

for i in range(len(sfclist)): #len(merlist)
    d_path = merlist[i]
    data   = nc.Dataset(d_path)
    # print(d_path)

    if i==0:
        merlat = data.variables['lat'][:]
        merlon = data.variables['lon'][:]
        merlev = data.variables['lev'][:]
        print(merlev[p_level])
        #shape latitude
        mer_lat = np.flip(merlat)
        mer_lat = np.array(mer_lat)
        mlat_ind1 = np.where(mer_lat == mer_lat.flat[np.abs(mer_lat - (latr1)).argmin()])[0]
        mlat_ind2 = np.where(mer_lat == mer_lat.flat[np.abs(mer_lat - (latr2)).argmin()])[0]
        p_mer_lat  = np.array(mer_lat[mlat_ind1[0]:mlat_ind2[0]])
        #shape longitude
        merlon[merlon > 180] = merlon[merlon > 180]-360
        # mer_lon = np.array(merlon)

    merT   = data.variables['T'][:] #(time, lev, lat, lon)
    mer_T = np.array(merT[:,:,::-1,:])
    p_mer_T.extend(mer_T[:,:,mlat_ind1[0]:mlat_ind2[0],:])
print('p_mer_T', np.shape(p_mer_T))

temp = np.array(p_mer_T)

for i in range(len(sfclist)): #len(merlist)
    s_path = sfclist[i]
    sdata  = nc.Dataset(s_path)
    # print(d_path)

    if i==0:
        sfclat = sdata.variables['lat'][:]
        sfclon = sdata.variables['lon'][:]
        #shape latitude
        sfc_lat = np.flip(sfclat)
        sfc_lat = np.array(sfc_lat)
        flat_ind1 = np.where(sfc_lat == sfc_lat.flat[np.abs(sfc_lat - (latr1)).argmin()])[0]
        flat_ind2 = np.where(sfc_lat == sfc_lat.flat[np.abs(sfc_lat - (latr2)).argmin()])[0]
        p_sfc_lat = np.array(sfc_lat[flat_ind1[0]:flat_ind2[0]])
        #shape longitude
        sfclon[sfclon > 180] = sfclon[sfclon > 180]-360
        # sfc_lon = np.array(sfclon)

    sfcT   = sdata.variables['TS'][:]
    sfc_T = np.array(sfcT[:,::-1,:])
    sfc_mer_T.extend(sfc_T[:,flat_ind1[0]:flat_ind2[0],:])


    sfcP   = sdata.variables['SLP'][:]
    sfc_P  = np.array(sfcP[:,::-1,:])
    sfc_mer_P.extend(sfc_P[:,flat_ind1[0]:flat_ind2[0],:])

print('sfc_mer_T', np.shape(sfc_mer_T))
print('sfc_mer_P', np.shape(sfc_mer_P))

sfctemp = np.array(sfc_mer_T)
sfcpres = np.array(sfc_mer_P)

for i in range(len(maclist)): #len(maclist)
    ddpath = maclist[i]
    ddata  = nc.Dataset(ddpath)
    macw   = ddata.variables['sfcwind'][:] #(time,lat,lon)
    # print(ddpath)

    if i==0:
        maclat = ddata.variables['lat'][:]
        maclon = ddata.variables['lon'][:]
        #shape latitude
        mac_lat = np.array(maclat)
        slat_ind1 = np.where(mac_lat == mac_lat.flat[np.abs(mac_lat - (latr1)).argmin()])[0]
        slat_ind2 = np.where(mac_lat == mac_lat.flat[np.abs(mac_lat - (latr2)).argmin()])[0]
        p_mac_lat  = np.array(mac_lat[slat_ind1[0]:slat_ind2[0]])
        #shape longitude
        maclon[maclon > 180] = maclon[maclon > 180]-360
        # mac_lon = np.array(maclon)

    n_w = macw[:,slat_ind1[0]:slat_ind2[0],:]
    p_mac_w.extend(n_w)
print('p_mac_w', np.shape(p_mac_w))

#reshaping longitudes
mer_lon = []
mer_lon.extend(merlon[180:360])
mer_lon.extend(merlon[0:180])

sfc_lon = []
sfc_lon.extend(sfclon[180:360])
sfc_lon.extend(sfclon[0:180])

mac_lon = []
mac_lon.extend(maclon[180:360])
mac_lon.extend(maclon[0:180])

wind   = np.array(p_mac_w)

grid_obs_wind     = regrid_wght_wnans(p_sfc_lat,sfc_lon,wind,lats_edges,lons_edges)[0]
grid_obs_temp_800 = regrid_wght_wnans(p_sfc_lat,sfc_lon,temp[:,p_level,:,:],lats_edges,lons_edges)[0]
grid_obs_temp_sfc = regrid_wght_wnans(p_sfc_lat,sfc_lon,sfctemp,lats_edges,lons_edges)[0]
grid_obs_pres_sfc = regrid_wght_wnans(p_sfc_lat,sfc_lon,sfcpres,lats_edges,lons_edges)[0]

lat_n = regrid_wght_wnans(p_sfc_lat,sfc_lon,sfcpres,lats_edges,lons_edges)[2][:,0]
lon_n = regrid_wght_wnans(p_sfc_lat,sfc_lon,sfcpres,lats_edges,lons_edges)[1][0,:]

theta_800 = np.array(np.multiply(grid_obs_temp_800, (100000/(merlev[p_level]*100))**(Rd/Cp)))
theta_sfc = np.array(np.multiply(grid_obs_temp_sfc, (100000/grid_obs_pres_sfc)**(Rd/Cp)))
print('theta_800', np.shape(theta_800))
print('theta_sfc', np.shape(theta_sfc))


p_CAOI = np.array(np.subtract(theta_sfc,theta_800))


#Mask for the ocean
maskm = np.ones((len(temp),len(lat_n),len(lon_n)))

for a in range(len(lat_n)):
    for b in range(len(lon_n)):
        if globe.is_land(lat_n[a], lon_n[b])==True:
            maskm[:,a,b] = math.nan
##############################
#reshaping M and wind
plot_CAOI = p_CAOI
grid_obs_wind = np.array(grid_obs_wind)
plot_wind = grid_obs_wind[0:len(sfclist), :, :]

# plot_CAOI = np.ones((len(temp),len(lat_n),len(lon_n)))
# plot_CAOI[:,:,180:360] = caoi_test[:,:,0:180]
# plot_CAOI[:,:,0:180]   = caoi_test[:,:,180:360]
# plot_CAOI = np.array(plot_CAOI)
#
# plot_wind = np.ones((len(temp),len(lat_n),len(lon_n)))
# plot_wind[:,:,180:360] = wind_test[:,:,0:180]
# plot_wind[:,:,0:180]   = wind_test[:,:,180:360]
# plot_wind = np.array(plot_wind)

#ocean only mask
plot_CAOI = np.array(np.multiply(maskm,plot_CAOI))
plot_wind = np.array(np.multiply(maskm,plot_wind))

plot_indx = np.isnan(plot_CAOI*plot_CAOI)==False
plot_mer_theta = plot_CAOI[plot_indx]
plot_mac_wind  = plot_wind[plot_indx]
###################################

#Sort and removing nan values
ind = np.argsort(plot_mer_theta)
xx  = np.sort(plot_mer_theta)
yy  = plot_mac_wind[ind]

xx_new = xx[yy>0]
yy_new = yy[yy>0]

indx = np.isnan(xx_new*yy_new)==False

from scipy import stats
# M_range = [np.percentile(xx_new[indx],2.5),np.percentile(xx_new[indx],97.5)]
bin_means, bin_edges, binnumber = stats.binned_statistic(xx_new[indx], yy_new[indx], 'mean', bins=100)
bin_means_x, bin_edges_x, binnumber_x = stats.binned_statistic(xx_new[indx], xx_new[indx], 'mean', bins=100)

index = np.isnan(bin_means_x*bin_means)==False
##############################################

#plot observations
plt.plot(bin_means_x[index], bin_means[index], color='black' ,marker='*', linestyle='dashed',
     markersize=5,linewidth=1, label='Observations')



plt.legend()
plt.ylabel('U10 [m/s]',fontsize='15')
plt.xlim(-20,20)
yti = '800'
plt.xlabel(r"M ($\Theta_{SST}$ - $\Theta_{"+yti+"})$ [K]",fontsize='15')
plt.title('U10 vs M for oceans between '+str(latr1)+'N to '+str(latr2)+'N')
plt.savefig('../figures/regrid_SST_U10vsM_'+str(latr1)+'N to '+str(latr2)+'N_800theta.png')
