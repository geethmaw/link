import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from mpl_toolkits.basemap import Basemap, addcyclic
import array
from mpl_toolkits import mplot3d
###########
###########
fn        = '/netdata/R1/data/wgeethma/data_cygnss/2018/cyg.ddmi.s20180802-003000-e20180802-233000.l3.grid-wind.a30.d31.nc'
ds        = nc.Dataset(fn)
lons			= ds.variables['lon'][:]           #size=1800 #Range is 0.1 .. 359.9
lats			= ds.variables['lat'][:]        #size=400 #Range is -39.9 .. 39.9
u		      = ds.variables['wind_speed'][:,:,:]    #float wind_speed(time, lat, lon) #units = "m s-1" 
time      = ds.variables['time'][:]          #size=24
ds.close
############
##############
mean_time = np.mean(u,axis=0)  #shape = 400,1800
#########contour plot
fig1 = plt.figure(figsize=(12,6))
cs = plt.contour(lons, lats, mean_time, np.arange(0,10), cmap='rainbow')
plt.title("daily mean of cygnss wind speed on 2018 August 2nd")
plt.xlabel("longitude (deg)")
plt.ylabel("latitude (deg)")
cbar =plt.colorbar()
cbar.set_label('wind speed (ms-1)', rotation=270)
#################
#########3D surface plot
fig2 = plt.figure()
ax = fig2.add_subplot(projection='3d')
x = lons
y = lats
c = mean_time
#data = np.random.random((10,20))
#print(data)
#print(data.shape)
X, Y = np.meshgrid(x, y)
ax.plot_trisurf(X, Y, c)
#ax.contour3D(X, Y, c)
#ax.set_title("daily mean of cygnss wind speed on 2018 August 2nd")
#ax.set_xlabel("longitude (deg)")
#ax.set_ylabel("latitude (deg)")
#ax.set_zlabel("wind speed (ms-1)")
plt.show()
  