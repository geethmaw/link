import netCDF4 as nc
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from mpl_toolkits.basemap import Basemap, addcyclic
import array
from mpl_toolkits import mplot3d
from scipy.interpolate import RegularGridInterpolator
###########
###########
fn        = '/netdata/R1/data/wgeethma/data_cygnss/2018/cyg.ddmi.s20180802-003000-e20180802-233000.l3.grid-wind.a30.d31.nc'
ds        = nc.Dataset(fn)
lons	  = ds.variables['lon'][:]           #size=1800 #Range is 0.1 .. 359.9
lats   	  = ds.variables['lat'][:]        #size=400 #Range is -39.9 .. 39.9
u		  = ds.variables['wind_speed'][:,:,:]    #float wind_speed(time, lat, lon) #units = "m s-1"
time      = ds.variables['time'][:]          #size=24
ds.close
##############
fna       = '/netdata/R1/data/wgeethma/data_amsr2/2018/RSS_AMSR2_ocean_L3_daily_2018-08-02_v08.2.nc'
dsa       = nc.Dataset(fna)
lonsa	  = dsa.variables['lon'][:]          #size=1440  #Range is 0.125 .. 359.875
latsa	  = dsa.variables['lat'][:]          #size=720 #Range is -89.875 .. 89.875
ua        = dsa.variables['wind_speed_LF'][:]  #float32 wind_speed_LF(pass, lat, lon)
timea     = dsa.variables['time'][:]            #(pass, lat, lon) shape=(2,720,1440)
passa     = dsa.variables['pass'][:]
dsa.close
###############figure
fig, ax = plt.subplots(2,1)
fig.set_size_inches(10,8)
###############cygnss Basemap
ax[1].set_title('cygnss data on 02nd August 2018')
X, Y = np.meshgrid(lons, lats)
m = Basemap(projection='mill',llcrnrlat=-50,urcrnrlat=50,llcrnrlon=0.,urcrnrlon=360.,resolution='c',ax=ax[1])
x, y = m(X, Y)
m.drawcountries()
m.drawcoastlines()
m.drawparallels(np.arange(-90,90,30),labels=[1,0,0,0])
m.drawmeridians(np.arange(m.lonmin,m.lonmax+30,30),labels=[0,0,0,1])
clevsw = np.arange(2., 10., 0.5)
for i in range(len(time)):
    w = u[i,::-1,:]
    cntr_w   = m.contourf(x, y, w, clevsw, cmap='BrBG')
cb     = plt.colorbar(cntr_w, orientation='vertical', extendrect=True, ticks=clevsw, ax=ax[1])
cb.set_ticks(clevsw)
cb.set_ticklabels(clevsw)
cb.set_label('m/s')
#########amsr Basemap
ax[0].set_title('AMSR data on 02nd August 2018')
X, Y = np.meshgrid(lonsa, latsa)
m = Basemap(projection='mill',llcrnrlat=-50,urcrnrlat=50,llcrnrlon=0.,urcrnrlon=360,resolution='c',ax=ax[0],anchor='C')
x, y = m(X, Y)
m.drawcountries()
m.drawcoastlines()
m.drawparallels(np.arange(-90,90,30),labels=[1,0,0,0])
m.drawmeridians(np.arange(m.lonmin,m.lonmax+30,30),labels=[0,0,0,1])
clevsw = np.arange(2., 10., 0.5)
for i in range(1):
    wa = ua[i,::-1,:]
    cntr_w   = m.contourf(x, y, wa, clevsw, cmap='bwr')
cb     = plt.colorbar(cntr_w, orientation='vertical', extendrect=True, ticks=clevsw, ax=ax[0])
cb.set_ticks(clevsw)
cb.set_ticklabels(clevsw)
cb.set_label('m/s')
plt.savefig('/netdata/R1/data/wgeethma/output/cyg_amsr_test10.png')
plt.show()
