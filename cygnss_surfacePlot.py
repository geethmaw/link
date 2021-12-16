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
fn      = '/netdata/R1/data/wgeethma/data_cygnss/2018/cyg.ddmi.s20180802-003000-e20180802-233000.l3.grid-wind.a30.d31.nc'
ds      = nc.Dataset(fn)
lons	= ds.variables['lon'][:]           #size=1800 #Range is 0.1 .. 359.9
lats	= ds.variables['lat'][:]        #size=400 #Range is -39.9 .. 39.9
u		= ds.variables['wind_speed'][:,:,:]    #float wind_speed(time, lat, lon) #units = "m s-1"
time    = ds.variables['time'][:]          #size=24
ds.close
##############
x = np.array(lons)
y = np.array(lats)
X, Y = np.meshgrid(x,y)
Z = u[10,:,:]
#####plot the surface
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, Z)
fig.colorbar(surf)
plt.savefig('sfc_cygnss.png')
plt.show()
