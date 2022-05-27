# @Author: Geethma Werapitiya <wgeethma>
# @Date:   2022-05-27T14:27:18-06:00
# @Email:  wgeethma@uwyo.edu
# @Last modified by:   wgeethma
# @Last modified time: 2022-05-27T14:47:35-06:00

# Get gcm data by using the link

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from readGCMs import read_var_mod
import calendar
from global_land_mask import globe
import glob
import math

#####Constants
Cp = 1004           #J/kg/K
Rd = 287            #J/kg/K
con= Rd/Cp

#latitude range
latr1 = 30
latr2 = 70

#pressure levels
p_level = 0

##### Retrieve GCM data ###########
###### HadGEM3 ####################
varname = ['sfcWind', 'tas','psl'] #'sfcWind', 'hfss', 'hfls', 'tas', 'ps', 'psl',
pvarname= ['ta']

end = 0
for i in varname:
    ncname = i+'_day_HadGEM3-GC31-LL_historical_r1i1p1f3_gn_19500101-20141230.nc'
    d_path = '/glade/collections/cmip/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/historical/r1i1p1f3/day/'+i+'/gn/v20190624/'+i+'/'+ncname
    data =xr.open_dataset(d_path)

    if end == 0:
        lon  = data.variables['lon'][:]  #(lon: 288) [0.0, 1.25, 2.5, ... 356.25, 357.5, 358.75]
        lat  = data.variables['lat'][:]  #(lat: 192) [-90.0 , -89.057592, -88.115183, ... 88.115183,  89.057592, 90.0]
        time = data.variables['time'][:] #(time: 36)

    if i == 'ta':
        lev  = data.variables['plev'][:]

    locals()[i+'_0'] = data.variables[i][:]
    # print(i+'_0', np.shape(locals()[i+'_0']))


for i in pvarname:
    ncname = i+'_day_HadGEM3-GC31-LL_historical_r1i1p1f3_gn_20000101-20141230.nc'
    d_path = '/glade/collections/cmip/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/historical/r1i1p1f3/day/'+i+'/gn/v20190624/'+i+'/'+ncname
    data =xr.open_dataset(d_path)

    if end == 0:
        lon_ta  = data.variables['lon'][:]  #(lon: 288) [0.0, 1.25, 2.5, ... 356.25, 357.5, 358.75]
        lat_ta  = data.variables['lat'][:]  #(lat: 192) [-90.0 , -89.057592, -88.115183, ... 88.115183,  89.057592, 90.0]
        time_ta = data.variables['time'][:] #(time: 36)
        lev     = data.variables['plev'][:]

    locals()[i+'_0'] = data.variables[i][:]
    # print(i+'_0', np.shape(locals()[i+'_0']))

for j in varname:
    locals()[j+str(0)] = locals()[j+'_'+str(0)][21600:22680,:,:]
    # print(j+'0', np.shape(locals()[j+str(0)]))

for j in pvarname:
    locals()[j+str(0)] = locals()[j+'_'+str(0)][3600:4680,:,:]
    # print(j+'0', np.shape(locals()[j+str(0)]))


theta_850 = ta0[:,1,:,:]*(100000/85000)**con
theta_700 = ta0[:,2,:,:]*(100000/70000)**con
theta_800 = theta_700 + ((2/3) * (theta_850 - theta_700))
theta_sfc = tas0*(100000/psl0)**con

M = theta_sfc - theta_800

x_lat = np.array(lat)
lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (30)).argmin()])[0]
lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (70)).argmin()])[0]

x_lon = lon
# lon_ind1 = np.where(x_lon == x_lon.flat[np.abs(x_lon - (-180)).argmin()])[0]
# lon_ind2 = np.where(x_lon == x_lon.flat[np.abs(x_lon - (180)).argmin()])[0]

x_sfcWind = sfcWind0
m_sfcWind = x_sfcWind[0:1094,lat_ind1[0]:lat_ind2[0],:]
lats = lat[lat_ind1[0]:lat_ind2[0]]

x_M = M
m_M = x_M[0:1094,lat_ind1[0]:lat_ind2[0],:]

cao = np.array(m_M)
sw  = np.array(m_sfcWind)

maskm = np.ones((len(sw),len(lats),len(lon)))

for a in range(len(lats)):
    for b in range(len(lon)):
        if globe.is_land(lats[a], lon[b]-180.)==True:
            maskm[:,a,b] = math.nan

plot_CAOI = np.array(np.multiply(maskm,cao))
wind      = np.array(np.multiply(maskm,sw))

plot_theta = plot_CAOI.ravel()
plot_wind = wind.ravel()

ind = np.argsort(plot_theta)
xx = np.sort(plot_theta)
yy = plot_wind[ind]

indx = np.isnan(xx*yy)==False

from scipy import stats
bin_means, bin_edges, binnumber = stats.binned_statistic(xx[indx], yy[indx], 'mean', bins=1000)
bin_means_x, bin_edges_x, binnumber_x = stats.binned_statistic(xx[indx], xx[indx], 'mean', bins=1000)

from skmisc.loess import loess
index = np.isnan(bin_means_x*bin_means)==False

plt.plot(bin_means_x[index], bin_means[index], label='HadGEM3')



########### OBSERVATIONS #########

import glob
merlist = np.sort(glob.glob('../data_merra/all_lat_lon/levels/MERRA2_*.nc'))
sfclist = np.sort(glob.glob('../data_merra/all_lat_lon/surface/MERRA2_*.nc'))
maclist = np.sort(glob.glob('../MACLWP_dailymean/take/wind1deg*.nc4'))

import netCDF4 as nc
import xarray as xr
p_mer_T   = []
p_mac_w   = []
sfc_mer_T = []
sfc_mer_P = []

for i in range(len(merlist)): #len(merlist)
    d_path = merlist[i]
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

    merT   = data.variables['T'][:] #(time, lev, lat, lon)
    mer_T = np.array(merT[:,:,::-1,:])
    p_mer_T.extend(mer_T[:,:,mlat_ind1[0]:mlat_ind2[0],:])

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

theta_800 = np.array(np.multiply(temp[:,p_level,:,:], (100000/(merlev[p_level]*100))**(Rd/Cp)))
theta_sfc = np.array(np.multiply(sfctemp, (100000/sfcpres)**(Rd/Cp)))

p_CAOI = np.array(np.subtract(theta_sfc,theta_800))


#Mask for the ocean
maskm = np.ones((len(temp),len(p_mer_lat),len(mer_lon)))

for a in range(len(p_mer_lat)):
    for b in range(len(mer_lon)):
        if globe.is_land(p_mer_lat[a], mer_lon[b])==True:
            maskm[:,a,b] = math.nan
##############################
#reshaping M and wind
caoi_test = p_CAOI
wind_test = wind

plot_CAOI = np.ones((len(temp),len(p_mer_lat),len(mer_lon)))
plot_CAOI[:,:,180:360] = caoi_test[:,:,0:180]
plot_CAOI[:,:,0:180]   = caoi_test[:,:,180:360]
plot_CAOI = np.array(plot_CAOI)

plot_wind = np.ones((len(temp),len(p_mer_lat),len(mer_lon)))
plot_wind[:,:,180:360] = wind_test[:,:,0:180]
plot_wind[:,:,0:180]   = wind_test[:,:,180:360]
plot_wind = np.array(plot_wind)

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
bin_means, bin_edges, binnumber = stats.binned_statistic(xx_new[indx], yy_new[indx], 'mean', bins=500)
bin_means_x, bin_edges_x, binnumber_x = stats.binned_statistic(xx_new[indx], xx_new[indx], 'mean', bins=500)

index = np.isnan(bin_means_x*bin_means)==False
##############################################

#plot observations
plt.plot(bin_means_x[index], bin_means[index], color='black' ,marker='*', linestyle='dashed',
     markersize=5,linewidth=1, label='Observations')

plt.legend()
plt.ylabel('U10 [m/s]',fontsize='15')
yti = '800'
plt.xlabel(r"M ($\Theta_{SST}$ - $\Theta_{"+yti+"})$ [K]",fontsize='15')
plt.title('U10 vs M for oceans between 30N to 70N')
plt.savefig('../figures/HadGEM3_obs.png')
