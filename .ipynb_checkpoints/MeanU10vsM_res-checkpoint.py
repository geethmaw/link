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
plt.rcParams['figure.figsize'] = (10, 10)

from con_models import get_cons
con, use_colors, varname, pvarname, modname, warming_modname, hiresmd, amip_md = get_cons()

#latitude range
latr1 = 30
latr2 = 80

gcm_p = 70000

time1=[2010, 1, 1]
time2=[2012, 12, 30]

lats_edges = np.arange(latr1,latr2+1,5)
lons_edges = np.arange(-180,181,5)

#binning
n_bins    = 20
M_range   = (-20,-5.3)
UM_range  = (-8.6,-5.3)
max_c     = 500
bin_count = 1000
bin_coun  = 1000

from obs_data_function import obs
merwind, macwind, temp, sfctemp, sfcpres, p_lev_obs, p_mer_lat, merlon, merlev = obs('T2M', 700, latr1, latr2)

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
M_700 = np.ma.masked_invalid(bin_means_x[bin_means_c>max_c])
s_err = bin_means_s[bin_means_c>max_c]/np.sqrt(bin_means_c[bin_means_c>max_c])
W_err = np.ma.masked_invalid(s_err)

bin_count = min(bin_count, len(W_SFC))

from scipy import stats
bin_means_WO, bin_edges_WO, binnumber_WO = stats.binned_statistic(m_700, w_sfc, 'mean', bins=n_bins,range=UM_range)
bin_means_MO, bin_edges_MO, binnumber_MO = stats.binned_statistic(m_700, m_700, 'mean', bins=n_bins,range=UM_range)
bin_means_CO, bin_edges_CO, binnumber_CO = stats.binned_statistic(m_700, w_sfc, 'count',bins=n_bins,range=UM_range)

W_SFC_O = np.ma.masked_invalid(bin_means_WO[bin_means_CO>max_c])
M_700_O = np.ma.masked_invalid(bin_means_MO[bin_means_CO>max_c])

bin_coun = min(bin_coun, len(W_SFC_O))

for j in range(0,len(modname)):
    print(modname[j])
    for i in varname:
        locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

    for k in pvarname:
        locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

print('modname done')

mm = len(modname)
for j in range(0,len(hiresmd)):
    print(hiresmd[j],' ', str(j))
    for i in varname:
        locals()[i+'__'+str(j+1+mm)] = read_var_hires('surface', hiresmd[j], 'highresSST-present', i, time1, time2)

    for k in pvarname:
        locals()[k+'__'+str(j+1+mm)] = read_var_hires('p_level', hiresmd[j], 'highresSST-present', k, time1, time2)

print('hires done')

M_plot  = []
W_plot  = []
W_erro  = []
g_res   = []
 
M_700_G = []
W_SFC_G = []
 
mm      = len(modname)


test_num = 0
for i in range(test_num,mm+len(hiresmd)): #l,mm+len(hiresmd)
    if i<len(modname):
        print(modname[i],str(i))
    else:
        print(hiresmd[i-mm],str(i))

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
        #print(j)
        locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
        locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
        locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
        locals()['grid_'+j+str(i+1)] = regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]
    
    g_lat_diff = np.abs(x_lat[1]-x_lat[0])
    g_lon_diff = np.abs(x_lon[1]-x_lon[0])
    g_res.append((np.sqrt(g_lat_diff**2 + g_lon_diff**2)) * 110.574)

    for k in pvarname:
        #print(k)
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
            if levels[p] == gcm_p:
                #print(levels[p])
                locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                grid_t_700 = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)[0]
                break;

    
    theta_700 = grid_t_700*(100000/gcm_p)**con
    theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

    t = min(len(theta_t2m),len(theta_700))
    M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
    plot_M = M_700.flatten()
    try_W  = locals()['grid_sfcWind'+str(i+1)][0:t,:,:]
    plot_W = try_W.flatten()

    ind = np.argsort(plot_M)

    final_M = np.sort(plot_M)
    final_W = plot_W[ind]

    indx = np.isnan(final_M*final_W)==False
    
    #for U10 vs M:
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
    
    #for the bias:
    bin_means_WG, bin_edges_WG, binnumber_WG = stats.binned_statistic(final_M[indx], final_W[indx], 'mean', bins=n_bins,range=M_range)
    bin_means_CG, bin_edges_CG, binnumber_CG = stats.binned_statistic(final_M[indx], final_W[indx], 'count', bins=n_bins,range=M_range)
    bin_means_MG, bin_edges_MG, binnumber_MG = stats.binned_statistic(final_M[indx], final_M[indx], 'mean', bins=n_bins,range=M_range)

    ind_c = np.where(bin_means_CG > max_c)

    M_700_G.append(np.ma.masked_invalid(bin_means_MG[ind_c]))
    W_SFC_G.append(np.ma.masked_invalid(bin_means_WG[ind_c]))
    bin_coun = min(bin_coun,len(bin_means_CG[ind_c]))


        
yy = []
xx = g_res
plt.clf()

y1  = [4.5,4.5]
y2  = [1,1]
x   = [40,100]
plt.fill_between(x, y1, y2, color ='red', alpha=0.1)
plt.annotate(xy=(x[0]+6,3.8), text='H Res', color='red',fontsize=9)

y1  = [4.5,4.5]
y2  = [1,1]
x   = [100,300]
plt.fill_between(x, y1, y2, color ='grey', alpha=0.1)
plt.annotate(xy=(x[0]+6,3.8), text='M Res', color='black',fontsize=9)

y1  = [4.5,4.5]
y2  = [1,1]
x   = [300,480]
plt.fill_between(x, y1, y2, color ='violet', alpha=0.1)
plt.annotate(xy=(x[0]+6,3.8), text='L Res', color='darkviolet',fontsize=9)

# plt.clf()
for i in range(1,len(modname)): #len(modname)
    bias      = W_SFC_O[0:bin_count] - W_SFC_G[i][0:bin_count]
    mean_bias = np.nanmean(bias)
    yy.append(mean_bias)
    
    e = np.std(bias)
    
    # plt.scatter(g_res[i],mean_bias,label=modname[i],color=use_colors[i])
    plt.errorbar(g_res[i],mean_bias, e,fmt='-o',color=use_colors[i])
    plt.annotate(xy=(g_res[i]-20,mean_bias+0.03), text=modname[i], color=use_colors[i],fontsize=9)

    
for i in range(len(modname),len(modname)+len(hiresmd)):
    bias      = W_SFC_O[0:bin_count] - W_SFC_G[i][0:bin_count]
    mean_bias = np.nanmean(bias)
    yy.append(mean_bias)
    
    e = np.std(bias)

    # plt.scatter(g_res[i],mean_bias,label=hiresmd[i-len(modname)],color=use_colors[i])
    plt.errorbar(g_res[i],mean_bias, e,fmt='-o',color=use_colors[i])
    plt.annotate(xy=(g_res[i]-20,mean_bias+0.03), text=hiresmd[i-len(modname)], color=use_colors[i],fontsize=9)

for i in range(0,1): #len(modname)
    bias      = W_SFC_O[0:bin_count] - W_SFC_G[i][0:bin_count]
    mean_bias = np.nanmean(bias)
    yy.append(mean_bias)
    
    e = np.std(bias)
    
    # plt.scatter(g_res[i],mean_bias,label=modname[i],color=use_colors[i])
    plt.errorbar(g_res[i],mean_bias, e,fmt='-o',color=use_colors[i])
    plt.annotate(xy=(g_res[i]-20,mean_bias+0.03), text=modname[i], color=use_colors[i],fontsize=9)

xx  = np.array(xx)
yy  = np.array(yy)
ind = np.argsort(xx)
xx  = np.sort(xx)
yy  = yy[ind]

from skmisc.loess import loess
l = loess(xx,yy)
l.fit()
pred = l.predict(xx, stderror=True)
conf = pred.confidence()

lowess = pred.values
ll = conf.lower
ul = conf.upper

# plt.plot(xx, yy, '+')
plt.plot(xx, lowess)
plt.fill_between(xx,ll,ul,alpha=.33)

plt.ylabel('U10 Bias [m/s]')
plt.xticks(np.arange(80,470,40))
plt.xlabel('Resolution [km]')
plt.title('for oceans between '+str(latr1)+'N to '+str(latr2)+'N\nfor unstable CAOs')
plt.savefig('../figures/final/LMH_BiasVsRes.png')