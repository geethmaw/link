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

#####Constants
Cp = 1004           #J/kg/K
Rd = 287            #J/kg/K
con= Rd/Cp

#latitude range
latr1 = 30
latr2 = 80


### GCM
modname = ['CESM2','CESM2-WACCM','HadGEM3-GC31-LL','HadGEM3-GC31-MM','IPSL-CM6A-LR','CNRM-ESM2-1','INM-CM5-0','MPI-ESM1-2-HR','UKESM1-0-LL','CMCC-CM2-SR5','CMCC-CM2-HR4','CNRM-CM6-1','CNRM-ESM2-1','IPSL-CM5A2-INCA']
varname = ['sfcWind', 'ts','psl'] #, 'hfss', 'hfls', 'tas', 'ps', 'psl',,'pr'
pvarname= ['ta']

l = 0
m = len(modname)   #l+1

time1=[2010, 1, 1]
time2=[2012, 12, 30]



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
fig = plt.figure(figsize=(12,10))


for i in range(l,m):

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



    for k in pvarname:
        levels = locals()['ta__'+str(i+1)][3]
        locals()['plot_'+k+str(i+1)] = []

        for p in range(len(levels)):
            locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
            locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
            plev  = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
            locals()['plot_'+k+str(i+1)].append(plev)
            # x   = np.array(np.repeat(levels[p],len(y)))

    temp_800  = log_interpolate_1d(80000, levels,np.array(locals()['plot_ta'+str(i+1)]), axis=0, fill_value=np.nan)
    theta_800 = temp_800*(100000/80000)**con
    theta_sfc = locals()['plot_ts'+str(i+1)]*(100000/locals()['plot_psl'+str(i+1)])**con

    M_800  = theta_sfc - theta_800[0,:,:,:]
    
    M_800  = np.nanmean(np.nanmean(M_800,axis=0),axis=1)
    
    indx   = np.isnan(M_800) == False
    
    from scipy.stats import norm
    import seaborn as sns
    ##############################################
    variableG = M_800[indx]
    hx = np.histogram(variableG,len(lats))
    xvals = hx[1][:-1]
    #
    xvalsSmooth = np.linspace(np.min(xvals),np.max(xvals),len(lats))
    kernel = stats.gaussian_kde(variableG)
    plt.plot(lats,kernel.pdf(xvalsSmooth),label = modname[i])
    
    
    
    
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('M PDF with latitudes')
plt.savefig('../figures/MPDFwithLats.png')