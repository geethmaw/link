# @Author: Geethma Werapitiya <wgeethma>
# @Date:   2022-05-25T12:30:57-06:00
# @Email:  wgeethma@uwyo.edu
# @Last modified by:   wgeethma
# @Last modified time: 2022-05-25T14:17:47-06:00



### Compare GCM and Obs M and U10 PDFs.

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

### GCM
modname = ['CESM2','CNRM-CM6-1','CESM2-WACCM',
           'HadGEM3-GC31-LL','NorESM2-LM']
varname = ['sfcWind', 'tas','psl'] #'sfcWind', 'hfss', 'hfls', 'tas', 'ps', 'psl',,'pr'
pvarname= ['ta']
conname = ['NCAR', 'CNRM-CERFACS','NCAR', 'MOHC', 'NCC']
ensname = ['r11i1p1f1', 'r1i1p1f2', 'r1i1p1f1',
           'r1i1p1f3', 'r1i1p1f1']
ggname  = ['gn', 'gr','gn', 'gn', 'gn']

l=0

#marker size
s = 120

for j in range(l,len(modname)): #

    for i in varname:
        locals()[i+'__'+str(j+1)] = read_var_mod(modn=modname[j], consort=conname[j], varnm=i, cmip='cmip6', exper='historical', ensmem=ensname[j], typevar='day', gg=ggname[j], read_p=False, time1=[2009, 12, 1], time2=[2012, 12, 31])
        # print(i)
    for k in pvarname:
        locals()[k+'__'+str(j+1)] = read_var_mod(modn=modname[j], consort=conname[j], varnm=k, cmip='cmip6', exper='historical', ensmem=ensname[j], typevar='day', gg=ggname[j], read_p=True, time1=[2009, 12, 1], time2=[2012, 12, 31])
        # print(k)
    # print(modname[j])

for i in range(l,len(modname)):
    lat  = locals()['sfcWind__'+str(i+1)][2]
    lon  = locals()['sfcWind__'+str(i+1)][3]
    time = locals()['sfcWind__'+str(i+1)][4]

    for j in varname:
        locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][0]

    for k in pvarname:
        locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][0]
        locals()['lev'+str(i+1)] = locals()['ta__'+str(i+1)][1]

    x_lat = np.array(lat)
    lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
    lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
    lats = lat[lat_ind1[0]:lat_ind2[0]]

    x_lon = lon

    maskm = np.ones((len(time),len(lats),len(lon)))

    for a in range(len(lats)):
        for b in range(len(lon)):
            if globe.is_land(lats[a], lon[b])==True:
                maskm[:,a,b] = math.nan


 ###  averaged theta at 800hPa and surface
    theta_850 = locals()['ta'+str(i+1)][:,1,:,:]*(100000/85000)**con
    theta_700 = locals()['ta'+str(i+1)][:,2,:,:]*(100000/70000)**con
    theta_800 = theta_700 + ((2/3) * (theta_850 - theta_700))

    theta_sfc = locals()['tas'+str(i+1)]*(100000/locals()['psl'+str(i+1)])**con

### CAOI at 800hPa
    M = theta_sfc - theta_800

    x_sfcWind = locals()['sfcWind'+str(i+1)]
    m_sfcWind = x_sfcWind[:,lat_ind1[0]:lat_ind2[0],:]
    lats = lat[lat_ind1[0]:lat_ind2[0]]

    x_M = M
    m_M = x_M[:,lat_ind1[0]:lat_ind2[0],:]

    cao = np.array(m_M)
    sw  = np.array(m_sfcWind)

    plot_CAOI = np.array(np.multiply(maskm,cao))
    wind      = np.array(np.multiply(maskm,sw))

    pl_theta  = plot_CAOI.reshape(-1)
    pl_wind   = wind.reshape(-1)

    plot_theta = pl_theta[pl_theta>-40]
    plot_wind  = pl_wind[pl_theta>-40]

    ind = np.argsort(plot_theta)
    xx = np.sort(plot_theta)
    yy = plot_wind[ind]

    indx = np.isnan(xx*yy)==False

    #######Plot U10 vs M################
    # from scipy import stats
    # bin_means, bin_edges, binnumber = stats.binned_statistic(xx[indx], yy[indx], 'mean', bins=1000)
    # bin_means_x, bin_edges_x, binnumber_x = stats.binned_statistic(xx[indx], xx[indx], 'mean', bins=1000)
    #
    # index = np.isnan(bin_means_x*bin_means)==False
    #
    # plt.plot(bin_means_x[index], bin_means[index], label=modname[i], alpha=0.6)
    ####################################

    ######## U10 PDF ##################
    from scipy.stats import norm
    # fig = plt.figure()
    # plt.scatter(np.sort(yy[indx]), norm.pdf(np.sort(yy[indx])), s=s, label=modname[i])
    # plt.legend()
    # plt.title(modname[i]+' - U10 PDF')
    #Change marker size
    # s = s-20
    ######################################

    # ######## M PDF ##################
    # from scipy.stats import norm
    # fig = plt.figure()
    plt.scatter(xx[indx], norm.pdf(xx[indx]), s=s, label=modname[i])
    s = s-20
    # # plt.legend()
    # plt.title(modname[i]+' - M PDF')
    # ######################################


## OBSERVATIONS
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

# indx = np.isnan(xx_new*yy_new)==False
#
# from scipy import stats
# bin_means, bin_edges, binnumber = stats.binned_statistic(xx_new[indx], yy_new[indx], 'mean', bins=500)
# bin_means_x, bin_edges_x, binnumber_x = stats.binned_statistic(xx_new[indx], xx_new[indx], 'mean', bins=500)
#
# index = np.isnan(bin_means_x*bin_means)==False
##############################################

# ############plot observations
# plt.plot(bin_means_x[index], bin_means[index], color='black' ,marker='*', linestyle='dashed',
#      markersize=5,linewidth=1, label='Observations')
# #################################################

############observations U10 PDF############
# from scipy.stats import norm
# import matplotlib as mpl
# mpl.rcParams['agg.path.chunksize'] = 10000
# test_yy_new = np.sort(yy_new)
# plt.scatter(test_yy_new, norm.pdf(test_yy_new), s=s, label='observations')
##############################################

############observations M PDF############
from scipy.stats import norm
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
test_xx_new = np.sort(xx_new)
plt.scatter(test_xx_new, norm.pdf(test_xx_new), s=s, label='observations')
##############################################

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



plt.legend()
plt.title('sfcWind PDF')
plt.savefig('../figures/caoiPDF.png')
# plt.ylabel('U10 [m/s]',fontsize='15')
# yti = '800'
# plt.xlabel(r"M ($\Theta_{SST}$ - $\Theta_{"+yti+"})$ [K]",fontsize='15')
# plt.title('U10 vs M for oceans between '+str(latr1)+'N to '+str(latr2)+'N')
# plt.savefig('../figures/all_U10vsM_'+str(latr1)+'N to '+str(latr2)+'N_800theta.png')
