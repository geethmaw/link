# @Author: Geethma Werapitiya <wgeethma>
# @Date:   2022-07-01T12:00:32-06:00
# @Email:  wgeethma@uwyo.edu
# @Last modified by:   wgeethma
# @Last modified time: 2022-07-12T09:14:33-06:00


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

from highres_read import read_var_hires
from myReadGCMsDaily import read_var_mod
from regrid_wght_3d import regrid_wght_wnans

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 100
plt.clf()
plt.rcParams['figure.figsize'] = (15.0/2.5, 15.0/2.5)


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

# use_colors = ['rosybrown','goldenrod','teal','blue','hotpink','green','red','cyan','magenta','cornflowerblue','mediumpurple','blueviolet',
# 'deeppink','lawngreen','coral','peru','salmon','burlywood','rosybrown','goldenrod','teal','blue','hotpink','green','red','cyan','magenta','yellow','cornflowerblue','mediumpurple','blueviolet',
# 'deeppink','lawngreen','coral','peru','salmon','burlywood']

use_colors = ['#88CCEE','#CC6677','#117733','#332288','#AA4499','#44AA99','#999933','#882255','#661100','#6699CC','#888888','#e6194b',
'#3cb44b','#0082c8','#f58231','#911eb4','#46f0f0','#f032e6','#d2f53c','#fabebe','#008080','#e6beff','#aa6e28','#fffac8','#800000','#aaffc3',
'#808000','#ffd8b1','#000080','#808080','#ffffff','#000000'] #'#ffe119', ,'#DDCC77'

### GCM
# modname = ['CESM2','CESM2-WACCM','HadGEM3-GC31-LL','HadGEM3-GC31-MM','IPSL-CM6A-LR','CNRM-ESM2-1','INM-CM5-0','MPI-ESM1-2-HR','UKESM1-0-LL','CMCC-CM2-SR5','CMCC-CM2-HR4','CNRM-CM6-1','CNRM-ESM2-1','IPSL-CM5A2-INCA','MPI-ESM1-2-LR','MPI-ESM-1-2-HAM']

modname = ['CESM2','CESM2-FV2','CESM2-WACCM','CMCC-CM2-HR4','CMCC-CM2-SR5','CMCC-ESM2','CNRM-CM6-1','CNRM-ESM2-1',
'HadGEM3-GC31-LL','HadGEM3-GC31-MM','INM-CM4-8','INM-CM5-0','IPSL-CM5A2-INCA','IPSL-CM6A-LR','MPI-ESM-1-2-HAM',
'MPI-ESM1-2-HR','MPI-ESM1-2-LR','NorESM2-MM','UKESM1-0-LL']   #'NorESM2-LM','CanESM5','GFDL-CM4','CESM2-WACCM-FV2',

varname = ['sfcWind', 'tas','psl'] #'sfcWind', 'hfss', 'hfls', 'tas', 'ps', 'psl',,'pr'
pvarname= ['ta']


l = 0
m = len(modname)   #l+1

time1=[2010, 1, 1]
time2=[2012, 12, 30]

lats_edges = np.arange(latr1,latr2+1,5)
lons_edges = np.arange(-180,181,5)

#binning
n_bins  = 20
M_range = (-20,5)

for j in range(l,m):
    print(modname[j])
    for i in varname:
        locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

    for k in pvarname:
        locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

M_plot = []
W_plot = []
W_erro = []
b_coun = []
m = len(modname)

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
            if levels[p] == 70000:
                print(levels[p])
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

    bin_means, bin_edges, binnumber       = stats.binned_statistic(final_M[indx], final_W[indx], 'mean', bins=n_bins,range=M_range)
    bin_means_c, bin_edges_c, binnumber_c = stats.binned_statistic(final_M[indx], final_W[indx], 'count', bins=n_bins,range=M_range)
    bin_means_s, bin_edges_s, binnumber_s = stats.binned_statistic(final_M[indx], final_W[indx], 'std', bins=n_bins,range=M_range)
    bin_means_x, bin_edges_x, binnumber_x = stats.binned_statistic(final_M[indx], final_M[indx], 'mean', bins=n_bins,range=M_range)

    ind_c = np.where(bin_means_c > 1000)
    # print(bin_means_c)
    # for x in bin_means_c:
    #     if x
    std_err = bin_means_s/np.sqrt(bin_means_c)
    M_plot.append(np.ma.masked_invalid(bin_means_x[ind_c])) #[ind_c]
    W_plot.append(np.ma.masked_invalid(bin_means[ind_c]))
    W_erro.append(np.ma.masked_invalid(std_err[ind_c]))
    b_coun.append(len(M_plot[i]))
    #

bin_count = np.min(b_coun)
for i in range(len(modname)):
    plt.plot(M_plot[i][0:bin_count], W_plot[i][0:bin_count], label=modname[i], color=use_colors[i])
    plt.fill_between(M_plot[i][0:bin_count], W_plot[i][0:bin_count]-W_erro[i][0:bin_count], W_plot[i][0:bin_count]+W_erro[i][0:bin_count], color=use_colors[i], alpha=0.2)
    plt.annotate(xy=(M_plot[i][bin_count-1],W_plot[i][bin_count-1]), text=modname[i],color=use_colors[i],fontsize=5)
    print(i)




## OBSERVATIONS
import glob
merlist = np.sort(glob.glob('../data_merra/all_lat_lon/level/MERRA2_*.nc'))
sfclist = np.sort(glob.glob('../data_merra/all_lat_lon/surface/MERRA2_*.nc'))
maclist = np.sort(glob.glob('../MACLWP_dailymean/take/wind1deg*.nc4'))

new_list_s = []
new_list_m = []
new_list_c = []

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

for i in range(len(new_s)): #len(merlist)
    d_path = new_m[i]
    data   = nc.Dataset(d_path)
    # print(d_path)

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
bin_means, bin_edges, binnumber       = stats.binned_statistic(m_700, w_sfc, 'mean', bins=n_bins, range=M_range)
bin_means_c, bin_edges_c, binnumber_c = stats.binned_statistic(m_700, w_sfc, 'count', bins=n_bins, range=M_range)
bin_means_s, bin_edges_s, binnumber_s = stats.binned_statistic(m_700, w_sfc, 'std', bins=n_bins, range=M_range)
bin_means_x, bin_edges_x, binnumber_x = stats.binned_statistic(m_700, m_700, 'mean', bins=n_bins, range=M_range)

W_SFC = np.ma.masked_invalid(bin_means)
M_700 = np.ma.masked_invalid(bin_means_x)
s_err = bin_means_s/np.sqrt(bin_means_c)
W_err = np.ma.masked_invalid(s_err)

plt.plot(M_700[0:bin_count], W_SFC[0:bin_count], color='black' ,linestyle='dashed',
     markersize=5,linewidth=1, label='Observations')
plt.fill_between(M_700[0:bin_count], W_SFC[0:bin_count]-W_err[0:bin_count], W_SFC[0:bin_count]+W_err[0:bin_count], color='black' ,alpha=0.2)
plt.annotate(xy=(M_700[bin_count-1],W_SFC[bin_count-1]), text='Observations', color='black',fontsize=7)


plt.fill_between([-10,-6],[5,5],[13,13],color='grey',alpha=0.2)
plt.annotate(xy=(-9,12.5), xytext=(3,0), textcoords='offset points', text='Unstable CAOI', va='center',color='grey',fontsize=5)

# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.ylabel('U10 [m/s]')
plt.xlim(-20,0)
yti = '700'
plt.xlabel(r"M ($\Theta_{t2m}$ - $\Theta_{"+yti+"})$ [K]")
plt.title('U10 vs M for oceans between '+str(latr1)+'N to '+str(latr2)+'N')
plt.savefig('../figures/final_U10vsM_'+str(latr1)+'N to '+str(latr2)+'N.png')
