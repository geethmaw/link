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
fna     = '/netdata/R1/data/wgeethma/data_amsr2/2018/RSS_AMSR2_ocean_L3_daily_2018-08-02_v08.2.nc'
dsa     = nc.Dataset(fna)
lonsa   = dsa.variables['lon'][:]          #size=1440  #Range is 0.125 .. 359.875
latsa	= dsa.variables['lat'][:]          #size=720 #Range is -89.875 .. 89.875
ua      = dsa.variables['wind_speed_LF'][:]  #float32 wind_speed_LF(pass, lat, lon)
timea   = dsa.variables['time'][:]            #(pass, lat, lon) shape=(2,720,1440)
passa   = dsa.variables['pass'][:]
dsa.close
############
# print(timea[1,103:106,103:106])
# print(ua[1,103:106,103:106])
##############
x = np.array(lons)
y = np.array(lats)
z = np.array(time)
zg, yg, xg = np.meshgrid(z,y, x, indexing='ij', sparse=True)
data = u
my_interpolate = RegularGridInterpolator((z,y,x), data)  #a function [0:23,-39.9:39.9,0.1:359]
#############
xa = np.array(lonsa)
ya = np.array(latsa)
za = np.array(passa)
zga, yga, xga = np.meshgrid(za, ya, xa, indexing='ij', sparse=True)
dataa = np.array(ua)
my_interpolate_a = RegularGridInterpolator((za,ya,xa), ua)#a function [1:2,-89.9:89.9,0.1:359]
#################subplots
fig,ax = plt.subplots(2,2,figsize=(10,16))
#################CYGNSS plot
for i in range(0,23):
    for j in range(-39,39,1):
        for k in range(90,91):
            pt = [i,j,k]
            cygn = my_interpolate(pt)
            if cygn>0:
                ax[0,0].plot(j,cygn,'bo')
#################AMSR plot
for i in range(1,2):
    for j in range(-39,39,1):
        for k in range(90,91):
            pt = [i,j,k]
            amsr = my_interpolate_a(pt)
            if amsr>0:
                ax[0,0].plot(j,amsr,'ro',label='AMSR2')
# plt.text(0.5,0.5,'AMSR2', color='r')
ax[0,0].set_xlabel('latitudes')
ax[0,0].set_ylabel('wind speed (m/s)')
ax[0,0].set_title('AMSR2 winds(red) and CYGNSS winds(blue)\n on 2nd August, 2018\n at 90deg longitude')
######subplot2
#################CYGNSS plot
for i in range(0,23):
    for j in range(-39,39,1):
        for k in range(180,181):
            pt = [i,j,k]
            cygn = my_interpolate(pt)
            if cygn>0:
                ax[0,1].plot(j,cygn,'bo')
#################AMSR plot
for i in range(1,2):
    for j in range(-39,39,1):
        for k in range(180,181):
            pt = [i,j,k]
            amsr = my_interpolate_a(pt)
            if amsr>0:
                ax[0,1].plot(j,amsr,'ro',label='AMSR2')
# plt.text(0.5,0.5,'AMSR2', color='r')
ax[0,1].set_xlabel('latitudes')
ax[0,1].set_ylabel('wind speed (m/s)')
ax[0,1].set_title('AMSR2 winds(red) and CYGNSS winds(blue)\n on 2nd August, 2018\n at 180deg longitude')
######subplot4
#################CYGNSS plot
for i in range(0,23):
    for j in range(-39,39,1):
        for k in range(359,360):
            pt = [i,j,k]
            cygn = my_interpolate(pt)
            if cygn>0:
                ax[1,1].plot(j,cygn,'bo')
#################AMSR plot
for i in range(1,2):
    for j in range(-39,39,1):
        for k in range(359,360):
            pt = [i,j,k]
            amsr = my_interpolate_a(pt)
            if amsr>0:
                ax[1,1].plot(j,amsr,'ro',label='AMSR2')
# plt.text(0.5,0.5,'AMSR2', color='r')
ax[1,1].set_xlabel('latitudes')
ax[1,1].set_ylabel('wind speed (m/s)')
ax[1,1].set_title('AMSR2 winds(red) and CYGNSS winds(blue)\n on 2nd August, 2018\n at 359deg longitude')
######subplot3
#################CYGNSS plot
for i in range(0,23):
    for j in range(-39,39,1):
        for k in range(270,271):
            pt = [i,j,k]
            cygn = my_interpolate(pt)
            if cygn>0:
                ax[1,0].plot(j,cygn,'bo')
#################AMSR plot
for i in range(1,2):
    for j in range(-39,39,1):
        for k in range(270,271):
            pt = [i,j,k]
            amsr = my_interpolate_a(pt)
            if amsr>0:
                ax[1,0].plot(j,amsr,'ro',label='AMSR2')
# plt.text(0.5,0.5,'AMSR2', color='r')
ax[1,0].set_xlabel('latitudes')
ax[1,0].set_ylabel('wind speed (m/s)')
ax[1,0].set_title('AMSR2 winds(red) and CYGNSS winds(blue)\n on 2nd August, 2018\n at 270deg longitude')
plt.savefig('/netdata/R1/data/wgeethma/output/08-02-18_AMSRandCYGNSS_Scatter_long_subplot.png')
