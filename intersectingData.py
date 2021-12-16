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
##########
# print(timea[:,10,10])
# print(my_interpolate_a([2,1,1]))
count = 1
am = []
cyg = []
for i in range(0,24): #time_cyg
    for j in range(1,3): #pass
        for k in range(-39,40): #lats
            for l in range(1,360): #lons
                pt  = [i,k,l]
                pta = [j,k,l]
                cygn = my_interpolate(pt)
                amsr = my_interpolate_a(pta)
                #print(cygn)
                if cygn>0 and amsr>0:
                    #print('pass '+str(j))
#                    print('count = '+str(count))
#                    print('amsr time = '+str(timea[j-1,k,l]))
#                    print('cygnss time = '+str(time[i])+'\n')
                    t = abs(amsr - cygn)
                    if t<=1:
                      am.append(amsr[0])  
                      cyg.append(cygn[0])
                      #print(t)
                      #print(count)
                    count = count+1
#am = np.array(am)
#cyg = np.array(cyg)
#print(am)
#print(cyg)
fig = plt.figure()
plt.scatter(cyg,am)
plt.title('AMSR vs CYGNSS')
plt.xlabel('CYGNSS (m/s)')
plt.ylabel('AMSR (m/s)')
plt.show()