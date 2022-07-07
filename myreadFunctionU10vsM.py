# @Author: Geethma Werapitiya <wgeethma>
# @Date:   2022-06-03T14:04:12-06:00
# @Email:  wgeethma@uwyo.edu
# @Last modified by:   geethmawerapitiya
# @Last modified time: 2022-06-28T01:10:45-06:00



### compare gcms, observations U10 vs M oceans 30N to 80N.
### Using myReadGCMsDaily.py to read from scratch
### Use all data percentile
### 30 to 80 latitudes
### use gcm ts
### use merra2 TS
### use UL 800hPa

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
from scipy import stats

#####JOB NUM
# job = '31'

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
modname = ['CESM2','CESM2-WACCM','HadGEM3-GC31-LL','HadGEM3-GC31-MM','IPSL-CM6A-LR','CNRM-ESM2-1','INM-CM5-0','MPI-ESM1-2-HR','UKESM1-0-LL','CMCC-CM2-SR5','CMCC-CM2-HR4','CNRM-CM6-1','CNRM-ESM2-1','IPSL-CM5A2-INCA']
varname = ['sfcWind', 'ts','psl'] #'sfcWind', 'hfss', 'hfls', 'tas', 'ps', 'psl',,'pr'
pvarname= ['ta']


l = 0
m = len(modname)   #l+1

time1=[2010, 1, 1]
time2=[2012, 12, 30]

fig, (ax1, ax2) = plt.subplots(1, 2)

for j in range(l,m):
    print(modname[j])
    for i in varname:
        locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)
        # print(i)
    for k in pvarname:
        locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)
        # print(k)
    print('done')

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
        print(np.shape(locals()[j+str(i+1)]))
#
    for k in pvarname:
        locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
        locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
        print(np.shape(locals()[k+str(i+1)]))

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

    theta_800 = theta_700 + (2/3) * (theta_850 - theta_700)

    theta_sfc = locals()['ts'+str(i+1)]*(100000/locals()['psl'+str(i+1)])**con

### CAOI at 700hPa
    M_800 = theta_sfc - theta_800

    x_sfcWind = locals()['sfcWind'+str(i+1)]
    m_sfcWind = x_sfcWind[:,lat_ind1[0]:lat_ind2[0],:]
    lats = lat[lat_ind1[0]:lat_ind2[0]]

    x_M_800 = M_800
    m_M_800 = x_M_800[:,lat_ind1[0]:lat_ind2[0],:]

    cao_800 = np.array(m_M_800)
    sw  = np.array(m_sfcWind)

    plot_CAOI_800 = np.array(np.multiply(maskm,cao_800))
    wind      = np.array(np.multiply(maskm,sw))

    pl_theta_800  = plot_CAOI_800.reshape(-1)
    pl_wind   = wind.reshape(-1)

    plo_theta_800 = pl_theta_800[pl_theta_800>-40]
    plo_wind_800  = pl_wind[pl_theta_800>-40]

    plot_theta_800 = plo_theta_800[plo_theta_800<40]
    plot_wind_800  = plo_wind_800[plo_theta_800<40]

    ind = np.argsort(plot_theta_800)
    xx  = np.sort(plot_theta_800)
    yy  = plot_wind_800[ind]

    indx = np.isnan(xx*yy)==False


    from scipy import stats
    # M_range = [np.percentile(xx[indx],2.5),np.percentile(xx[indx],97.5)]
    bin_means, bin_edges, binnumber = stats.binned_statistic(xx[indx], yy[indx], 'mean', bins=100)
    bin_means_x, bin_edges_x, binnumber_x = stats.binned_statistic(xx[indx], xx[indx], 'mean', bins=100)

    index = np.isnan(bin_means_x*bin_means)==False

    plt.plot(bin_means_x[index], bin_means[index], label=modname[i], alpha=0.6)

## OBSERVATIONS
import glob
merlist = np.sort(glob.glob('../data_merra/all_lat_lon/levels/MERRA2_*.nc'))
sfclist = np.sort(glob.glob('../data_merra/all_lat_lon/surface_old/MERRA2_*.nc'))
maclist = np.sort(glob.glob('../MACLWP_dailymean/take/wind1deg*.nc4'))

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

theta_800 = np.array(np.multiply(temp[:,p_level,:,:], (100000/(merlev[p_level]*100))**(Rd/Cp)))
theta_sfc = np.array(np.multiply(sfctemp, (100000/sfcpres)**(Rd/Cp)))
print('theta_800', np.shape(theta_800))
print('theta_sfc', np.shape(theta_sfc))


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
wind_test = wind[0:len(sfclist), :, :]

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
# M_range = [np.percentile(xx_new[indx],2.5),np.percentile(xx_new[indx],97.5)]
bin_means, bin_edges, binnumber = stats.binned_statistic(xx_new[indx], yy_new[indx], 'mean', bins=1000)
bin_means_x, bin_edges_x, binnumber_x = stats.binned_statistic(xx_new[indx], xx_new[indx], 'mean', bins=1000)

index = np.isnan(bin_means_x*bin_means)==False
##############################################

#plot observations
plt.plot(bin_means_x[index], bin_means[index], color='black' ,marker='*', linestyle='dashed',
     markersize=5,linewidth=1, label='Observations')


### PPE
# enn = np.arange(201,251)
# ppe_var = ['U10', 'PSL', 'T850','TREFHT']
# for en in enn:
#     if en != 175:
#         for i in ppe_var: #TREFHT was used since no TS. Should double check this.
#             d_path = '/glade/campaign/cgd/projects/ppe/cam_ppe/rerun_PPE_250/PD/PD_timeseries/PPE_250_ensemble_PD.'+f'{en:03d}'+\
#             '/atm/hist/cc_PPE_250_ensemble_PD.'+f'{en:03d}'+'.h1.'+str(i)+'.nc'
#             data =xr.open_dataset(d_path)
#
#             if en == enn[0]:
#                 lon  = data.variables['lon'][:]  #(lon: 288) [0.0, 1.25, 2.5, ... 356.25, 357.5, 358.75]
#                 lat  = data.variables['lat'][:]  #(lat: 192) [-90.0 , -89.057592, -88.115183, ... 88.115183,  89.057592, 90.0]
#                 time = data.variables['time'][:] #(time: 36)
#
#             locals()[str(en)+'_'+i] = data.variables[i][:]
#
# x_lat = np.array(lat)
# lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
# lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
# lat_new  = lat[lat_ind1[0]:lat_ind2[0]]
#
# mask = np.ones((1096,len(lat_new),len(lon)))
#
# for a in range(len(lat_new)):
#     for b in range(len(lon)):
#         if globe.is_land(lat_new[a], lon[b]-180.)==True:
#             mask[:,a,b] = math.nan
#
# for en in enn:
#     if en != 175:
#         # print(en)
#         for i in ppe_var: #
#             tmp  = locals()[str(en)+'_'+i]
#             tmp2 = tmp[0:1096,lat_ind1[0]:lat_ind2[0],:]
#             locals()['MID'+i+'_'+str(en)] = np.multiply(mask,tmp2)
#
#         locals()['theta_850_'+str(en)] = np.multiply(locals()['MIDT850_'+str(en)],
#                                                      (np.divide(locals()['MIDPSL_'+str(en)],85000))**(Rd/Cp))
#         locals()['M_'+str(en)]   = np.array(np.subtract(locals()['MIDTREFHT_'+str(en)],locals()['theta_850_'+str(en)])).reshape(-1)
#         locals()['U10_'+str(en)] = np.array(locals()['MIDU10_'+str(en)]).reshape(-1)
#
#         x = locals()['M_'+str(en)]
#         y = locals()['U10_'+str(en)]
#         ind = np.argsort(x)
#         xx = np.sort(x)
#         yy = y[ind]
#
#         indx = np.isnan(xx*yy)==False
#
#         bin_means, bin_edges, binnumber = stats.binned_statistic(xx[indx], yy[indx], 'mean', bins=1000)
#         bin_means_x, bin_edges_x, binnumber_x = stats.binned_statistic(xx[indx], xx[indx], 'mean', bins=1000)
#
#         index = np.isnan(bin_means_x*bin_means)==False
#
#
#         if en==250:
#             plt.plot(bin_means_x[index], bin_means[index], alpha=0.2,label='PPE')
#
#         else:
#             plt.plot(bin_means_x[index], bin_means[index], alpha=0.2)



plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.ylabel('U10 [m/s]',fontsize='15')
yti = str(merlev[p_level])
plt.xlabel(r"M ($\Theta_{SST}$ - $\Theta_{"+yti+"})$ [K]",fontsize='15')
plt.title('U10 vs M for oceans between '+str(latr1)+'N to '+str(latr2)+'N')
plt.savefig('../figures/SST_U10vsM_'+str(latr1)+'N to '+str(latr2)+'N_800theta.png')
