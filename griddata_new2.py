import netCDF4 as nc
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from mpl_toolkits.basemap import Basemap, addcyclic
import array
from mpl_toolkits import mplot3d
from scipy.interpolate import RegularGridInterpolator
from scipy import interpolate
###########
###########
fn        = '/netdata/R1/data/wgeethma/data_cygnss/2018/cyg.ddmi.s20180802-003000-e20180802-233000.l3.grid-wind.a30.d31.nc'
ds        = nc.Dataset(fn)
lons			= ds.variables['lon'][:]           #size=1800 #Range is 0.1 .. 359.9
lats			= ds.variables['lat'][:]        #size=400 #Range is -39.9 .. 39.9
u		      = ds.variables['wind_speed'][:,:,:]    #float wind_speed(time, lat, lon) #units = "m s-1"
time      = ds.variables['time'][:]          #size=24
ds.close
##############
fna       = '/netdata/R1/data/wgeethma/data_amsr2/2018/RSS_AMSR2_ocean_L3_daily_2018-01-01_v08.2.nc'
dsa       = nc.Dataset(fna)
lonsa			= dsa.variables['lon'][:]          #size=1440  #Range is 0.125 .. 359.875
latsa			= dsa.variables['lat'][:]          #size=720 #Range is -89.875 .. 89.875
ua        = dsa.variables['wind_speed_LF'][:]  #float32 wind_speed_LF(pass, lat, lon)
timea     = dsa.variables['time'][:]            #(pass, lat, lon) shape=(2,720,1440)
passa     = dsa.variables['pass'][:]
dsa.close
############
##############
x = lons
y = lats
z = time
zg, yg, xg = np.meshgrid(z,y, x)
data = u
pts = (z,y,x)
my_interpolate = interpolate.interpn(pts,data,(zg,yg,xg))
print(my_interpolate[3,3,3])
#############
xa = lonsa
ya = latsa
za = passa
zga, yga, xga = np.meshgrid(za, ya, xa, indexing='ij', sparse=True)
dataa = np.array(ua)
my_interpolate_a = RegularGridInterpolator((za,ya,xa), ua)
#################
