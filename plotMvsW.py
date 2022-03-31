import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import xarray as xr
from global_land_mask import globe
from scipy import stats
from skmisc.loess import loess
import glob

#####Constants
Cp = 1004           #J/kg/K
Rd = 287            #J/kg/K

merlist = np.sort(glob.glob('../data_merra/lat_30_70/new/MERRA2_*.nc'))
maclist = np.sort(glob.glob('../MACLWP_dailymean/take/wind1deg*.nc4'))

p_mer_T = []
p_mac_w = []

for i in range(31): #len(merlist)
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
        #shape longitude
        mer_lon = np.array(merlon)
        
        
    merT   = data.variables['T'][:] #(time, lev, lat, lon)
    mer_T = np.array(merT[:,:,::-1,:])
    p_mer_T.extend(mer_T)
    
p_mer_T = np.array(p_mer_T)
temp = np.ma.masked_where(p_mer_T == np.max(p_mer_T), p_mer_T)
    
for i in range(1): #len(maclist)
    ddpath = maclist[i]
    ddata  = nc.Dataset(ddpath)
    macw   = ddata.variables['sfcwind'][:] #(time,lat,lon)
    # print(ddpath)
    
    if i==0:
        maclat = ddata.variables['lat'][:]
        maclon = ddata.variables['lon'][:]
        #shape latitude
        mac_lat = np.array(maclat)
        slat_ind1 = np.where(mac_lat == mac_lat.flat[np.abs(mac_lat - (31)).argmin()])[0]
        slat_ind2 = np.where(mac_lat == mac_lat.flat[np.abs(mac_lat - (71)).argmin()])[0]
        p_mac_lat  = np.array(mac_lat[slat_ind1[0]:slat_ind2[0]])
        #shape longitude
        maclon[maclon > 180] = maclon[maclon > 180]-360
        mac_lon = np.array(maclon)
        p_mac_lon = []
        p_mac_lon.extend(mac_lon[180::])
        p_mac_lon.extend(mac_lon[0:180])
        p_mac_lon = np.array(p_mac_lon)
    n_w = macw[:,slat_ind1[0]:slat_ind2[0],180::]
    new_w = np.append(n_w,macw[:,slat_ind1[0]:slat_ind2[0],0:180],2)
    p_mac_w.extend(new_w)

p_mac_w = np.array(p_mac_w)
wind = np.ma.masked_where(p_mac_w == np.min(p_mac_w), p_mac_w)
wind = np.ma.masked_where(wind < 5, wind)

theta_850 = np.array(np.multiply(temp[:,1,:,:], (100/85)**(Rd/Cp)))
p_theta_850 = np.ma.masked_where(theta_850 == np.max(theta_850), theta_850)

CAOI = np.array(np.subtract(temp[:,0,:,:],p_theta_850))
p_CAOI = np.ma.masked_where(CAOI == np.max(CAOI), CAOI)
p_CAOI = np.ma.masked_where(p_CAOI < -20, p_CAOI)

maskm = np.ones((len(mer_lat),len(mer_lon)))

for a in range(len(mer_lat)):
    for b in range(len(mer_lon)):
        if globe.is_land(mer_lat[a], mer_lon[b])==True:
            maskm[a,b] = 0
            
plot_CAOI = np.array(np.multiply(maskm,p_CAOI))
plot_CAOI = np.ma.masked_where(plot_CAOI == np.max(plot_CAOI), plot_CAOI)
plot_CAOI = np.ma.masked_where(plot_CAOI < -20, plot_CAOI)

plot_mer_theta = plot_CAOI.reshape(-1)
plot_mac_wind = wind.reshape(-1)

bin_means, bin_edges, binnumber = stats.binned_statistic(plot_mer_theta, plot_mac_wind, 'mean', bins=200)
bin_width = (bin_edges[1] - bin_edges[0])
bin_centers = bin_edges[1:] - bin_width/2

mac_bin_means = np.ma.masked_where(bin_means < -20, bin_means)
mac_bin_centers = bin_centers[~np.isnan(bin_means)]

#ind = np.argsort(plot_mer_theta)
#xx = np.sort(plot_mer_theta)
#yy = plot_mac_wind[ind]

#l = loess(xx,yy)
#l.fit()
#pred = l.predict(xx, stderror=True)
#conf = pred.confidence()

#macloess = pred.values
#ll = conf.lower
#ul = conf.upper
#finaltheta = xx


#plt.plot(finaltheta, macloess)
plt.scatter(mac_bin_centers, mac_bin_means)
plt.ylabel('U10 [m/s]',fontsize='15')
plt.xlabel('CAOI [K]',fontsize='15')
plt.title('MACLWP wind vs MERRA2 M\nfor 30N to 70N ocean only')
plt.savefig('try.png')

