# @Author: geethmawerapitiya
# @Date:   2022-07-06T08:53:28-06:00
# @Project: Research
# @Filename: regridded_BiasVsResolution.py
# @Last modified by:   wgeethma
# @Last modified time: 2022-07-06T14:19:53-06:00

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
plt.clf()
plt.rcParams['figure.figsize'] = (12.0/2.5, 12.0/2.5)


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
p_level_700 = 3  ### 700hPa

### GCM
# modname = ['CESM2','CESM2-WACCM','HadGEM3-GC31-LL','HadGEM3-GC31-MM','IPSL-CM6A-LR','CNRM-ESM2-1','INM-CM5-0','MPI-ESM1-2-HR','UKESM1-0-LL','CMCC-CM2-SR5','CMCC-CM2-HR4','CNRM-CM6-1','CNRM-ESM2-1','IPSL-CM5A2-INCA','MPI-ESM1-2-LR','MPI-ESM-1-2-HAM']

modname = ['CESM2','CESM2-FV2','CESM2-WACCM','CMCC-CM2-HR4','CMCC-CM2-SR5','CMCC-ESM2','CNRM-CM6-1','CNRM-ESM2-1',
'HadGEM3-GC31-LL','HadGEM3-GC31-MM','INM-CM4-8','INM-CM5-0','IPSL-CM5A2-INCA','IPSL-CM6A-LR','MPI-ESM-1-2-HAM',
'MPI-ESM1-2-HR','MPI-ESM1-2-LR','NorESM2-MM','UKESM1-0-LL']

varname = ['sfcWind','tas','psl'] #'sfcWind', 'hfss', 'hfls', 'tas', 'ps', 'psl',,'pr'
pvarname= ['ta']

use_colors = ['rosybrown','goldenrod','teal','blue','hotpink','green','red','cyan','magenta','cornflowerblue','mediumpurple','blueviolet',
'deeppink','lawngreen','coral','peru','salmon','burlywood','rosybrown','goldenrod','teal','blue','hotpink','green','red','cyan','magenta','yellow','cornflowerblue','mediumpurple','blueviolet',
'deeppink','lawngreen','coral','peru','salmon','burlywood']

l = 0
m = len(modname)   #l+1

time1=[2010, 1, 1]
time2=[2012, 12, 30]

#binning
n_bins  = 20
M_range = (-10,-6)

lats_edges = np.arange(latr1,latr2+1,5)
lons_edges = np.arange(-180,181,5)

############## OBS ##################################
import glob
merlist = np.sort(glob.glob('../data_merra/all_lat_lon/level/MERRA2_*.nc'))
sfclist = np.sort(glob.glob('../data_merra/all_lat_lon/surface/MERRA2_*.nc'))
maclist = np.sort(glob.glob('../MACLWP_dailymean/take/wind1deg*.nc4'))

new_list_s = []
new_list_m = []
# new_list_c = []

s = 0
m = 0
length = max(len(merlist), len(sfclist))

while m != length:
    print(s,m)
    name_s = os.path.basename(sfclist[s])
    date_s = name_s.split(".")[2]

    name_m = os.path.basename(merlist[m])
    date_m = name_m.split(".")[2]
    print(sfclist[s],date_s)
    print(merlist[m],date_m)

    if date_s==date_m:
        new_list_s.append(sfclist[s])
        new_list_m.append(merlist[m])

        s = s+1
        m = m+1

    elif date_s<date_m:
        s = s+1

    elif date_s>date_m:
        m = m+1

macwind = []
new_s   = []
new_m   = []
macdate = []
for i in range(len(new_list_s)):
    flag = 0
    name_s = os.path.basename(new_list_s[i])
    date_s = name_s.split(".")[2]

    for k in range(len(maclist)):

        name_mac = os.path.basename(maclist[k])
        date_mac = name_mac.split(".")[1]
        ddata    = nc.Dataset(maclist[k])
        mactime  = ddata.variables['time'][:]
        macw     = ddata.variables['sfcwind'][:]

        if date_mac==date_s[0:-2]:
            for r in range(len(mactime)):
                if str(mactime[r]). zfill(2) == date_s[-2::]:
                    macwind.append(macw[r,:,:])
                    # print(maclist[k],str(mactime[r]). zfill(2))
                    # print(new_list_s[i],date_s)
                    new_m.append(new_list_m[i])
                    new_s.append(new_list_s[i])
                    macdate.append(date_mac+str(mactime[r]). zfill(2))

                    flag = 1
                    break;

        if flag == 1:
            break;


macwind = np.array(macwind)

#####################

p_mer_T   = []
p_mac_w   = []
sfc_mer_T = []
sfc_mer_P = []

for i in range(len(new_s)):
    d_path = new_m[i]
    data   = nc.Dataset(d_path)

    if i==0:
        merlat = data.variables['lat'][:]
        merlon = data.variables['lon'][:]
        merlev = data.variables['lev'][:]
        print(merlev[p_level_700])
        #shape latitude
        mer_lat = np.flip(merlat)
        mer_lat = np.array(mer_lat)
        mlat_ind1 = np.where(mer_lat == mer_lat.flat[np.abs(mer_lat - (latr1)).argmin()])[0]
        mlat_ind2 = np.where(mer_lat == mer_lat.flat[np.abs(mer_lat - (latr2)).argmin()])[0]
        p_mer_lat  = np.array(mer_lat[mlat_ind1[0]:mlat_ind2[0]])
        #shape longitude
        merlon[merlon > 180] = merlon[merlon > 180]-360
        # mer_lon = np.array(merlon)

    merT      = data.variables['T'][:] #(time, lev, lat, lon)
    mer_T     = np.array(merT[:,:,::-1,:])
    mer_T_700 = mer_T[:,p_level_700,mlat_ind1[0]:mlat_ind2[0],:]
    p_mer_T.extend(mer_T_700)

    s_path = new_s[i]
    sdata  = nc.Dataset(s_path)

    sfcT   = sdata.variables['T2M'][:]
    sfc_T = np.array(sfcT[:,::-1,:])
    sfc_mer_T.extend(sfc_T[:,mlat_ind1[0]:mlat_ind2[0],:])


    sfcP   = sdata.variables['SLP'][:]
    sfc_P  = np.array(sfcP[:,::-1,:])
    sfc_mer_P.extend(sfc_P[:,mlat_ind1[0]:mlat_ind2[0],:])

p_mac_w = macwind[:,mlat_ind1[0]:mlat_ind2[0],:]

wind    = np.array(p_mac_w)
temp    = np.array(p_mer_T)
sfctemp = np.array(sfc_mer_T)
sfcpres = np.array(sfc_mer_P)

grid_obs_wind     = regrid_wght_wnans(p_mer_lat,merlon,wind,lats_edges,lons_edges)[0]
grid_obs_temp_700 = regrid_wght_wnans(p_mer_lat,merlon,temp,lats_edges,lons_edges)[0]
grid_obs_temp_sfc = regrid_wght_wnans(p_mer_lat,merlon,sfctemp,lats_edges,lons_edges)[0]
grid_obs_pres_sfc = regrid_wght_wnans(p_mer_lat,merlon,sfcpres,lats_edges,lons_edges)[0]

lat_n = regrid_wght_wnans(p_mer_lat,merlon,sfcpres,lats_edges,lons_edges)[2][:,0]
lon_n = regrid_wght_wnans(p_mer_lat,merlon,sfcpres,lats_edges,lons_edges)[1][0,:]

theta_700 = np.array(np.multiply(grid_obs_temp_700, (100000/(merlev[p_level_700]*100))**(Rd/Cp)))
theta_sfc = np.array(np.multiply(grid_obs_temp_sfc, (100000/grid_obs_pres_sfc)**(Rd/Cp)))
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
bin_means_WO, bin_edges_WO, binnumber_WO = stats.binned_statistic(m_700, w_sfc, 'mean', bins=n_bins,range=M_range)
bin_means_MO, bin_edges_MO, binnumber_MO = stats.binned_statistic(m_700, m_700, 'mean', bins=n_bins,range=M_range)
bin_means_CO, bin_edges_CO, binnumber_CO = stats.binned_statistic(m_700, w_sfc, 'count',bins=n_bins,range=M_range)

W_SFC_O = np.ma.masked_invalid(bin_means_WO[bin_means_CO>1000])
M_700_O = np.ma.masked_invalid(bin_means_MO[bin_means_CO>1000])


############## GCMs #################################
m = len(modname)

for j in range(l,m):
    print(modname[j])
    for i in varname:
        locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)
        # print(i)
    for k in pvarname:
        locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)
        # print(k)
    print('done')

M_700_G = []
W_SFC_G = []
b_coun  = []
g_res   = []

for i in range(l,m):

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
        locals()['grid_'+j+str(i+1)] = regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]



    for k in pvarname:
        locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
        locals()['grid_'+k+str(i+1)] = []

        levels = locals()['plot_levels'+str(i+1)]

        for p in range(len(levels)):
            print(levels[p])
            if levels[p] == 70000:
                print('if ',levels[p])
                locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                grid_t_700 = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)[0]
                break;

    theta_700 = grid_t_700*(100000/70000)**con
    theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

    M_700  = theta_t2m - theta_700
    plot_M = M_700.flatten()
    plot_W = locals()['grid_sfcWind'+str(i+1)].flatten()

    ind = np.argsort(plot_M)

    final_M = np.sort(plot_M)
    final_W = plot_W[ind]

    indx = np.isnan(final_M*final_W)==False
    
    bin_means_WG, bin_edges_WG, binnumber_WG = stats.binned_statistic(final_M[indx], final_W[indx], 'mean', bins=n_bins,range=M_range)
    bin_means_CG, bin_edges_CG, binnumber_CG = stats.binned_statistic(final_M[indx], final_W[indx], 'count', bins=n_bins,range=M_range)
    bin_means_MG, bin_edges_MG, binnumber_MG = stats.binned_statistic(final_M[indx], final_M[indx], 'mean', bins=n_bins,range=M_range)

    ind_c = np.where(bin_means_CG > 1000)
    print(bin_means_CG)
    # for x in bin_means_c:
    #     if x

    M_700_G.append(np.ma.masked_invalid(bin_means_MG[ind_c])) #[ind_c]
    W_SFC_G.append(np.ma.masked_invalid(bin_means_WG[ind_c]))
    b_coun.append(len(M_700_G[i]))
    
    g_lat_diff = np.abs(x_lat[1]-x_lat[0])
    g_lon_diff = np.abs(x_lon[1]-x_lon[0])
    g_res.append((np.sqrt(g_lat_diff**2 + g_lon_diff**2)) * 110.574)

bin_count = np.min(b_coun)
bin_count = min(bin_count, len(W_SFC_O))
for i in range(l,m):
    bias      = W_SFC_O[0:bin_count] - W_SFC_G[i][0:bin_count]
    mean_bias = np.nanmean(bias)

    plt.scatter(g_res[i],mean_bias,label=modname[i],color=use_colors[i])
    plt.annotate(xy=(g_res[i]+0.03,mean_bias+0.03), text=modname[i], color=use_colors[i],fontsize=9)
    print(modname[i])
    print(g_res[i])

#############PLOT###############
#plt.legend()
plt.ylabel('U10 Bias [m/s]')
plt.xlabel('Resolution [km]')
plt.title('for oceans between '+str(latr1)+'N to '+str(latr2)+'N/nfor M>0')
plt.savefig('../figures/new_final_BiasVsRes'+str(latr1)+'N to '+str(latr2)+'N.png')
