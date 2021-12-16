import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from mpl_toolkits.basemap import Basemap, addcyclic
import array
###########
###########
fn        = '/netdata/R1/data/wgeethma/data_cygnss/2018/cyg.ddmi.s20180802-003000-e20180802-233000.l3.grid-wind.a30.d31.nc'
ds        = nc.Dataset(fn)
lons			= ds.variables['lon'][:]           #size=1800 #Range is 0.1 .. 359.9
lats			= ds.variables['lat'][:]        #size=400 #Range is -39.9 .. 39.9
u		      = ds.variables['wind_speed'][:,:,:]    #float wind_speed(time, lat, lon) #units = "m s-1" 
time      = ds.variables['time'][:]          #size=24
ds.close
fna       = '/netdata/R1/data/wgeethma/data_amsr2/2018/RSS_AMSR2_ocean_L3_daily_2018-01-01_v08.2.nc'
dsa       = nc.Dataset(fna)
lonsa			= dsa.variables['lon'][:]          #size=1440  #Range is 0.125 .. 359.875
latsa			= dsa.variables['lat'][:]          #size=720 #Range is -89.875 .. 89.875
ua        = dsa.variables['wind_speed_LF'][:]  #float32 wind_speed_LF(pass, lat, lon)
timea     = dsa.variables['time'][:]            #shape=(2,720,1440)
passa     = dsa.variables['pass'][:]
dsa.close
#############
#############
print(u[0,1,2])
x = u[0,1,2]
print(f"It's np.isnan  : {np.isnan(x)}")
#print(lonsa[100:110])
#print(latsa.shape)
#print(timea.shape)
#print(passa)
#print(lonsa[0])
#print(lonsa[-1])
#for i in range(1):
#  fig = plt.figure(figsize=(12,6))
#  x = np.arange(-39.9,40.,1.)
#  w        = u[10, ::-1, 899]
#  wa       = ua[1,::-1,719]
#  print(lons[899]) #179.9
#  print(lonsa[719]) #179.875
#  plt.plot(np.arange(-39.9,40.,0.2),w[::-1],'ro')
#  plt.plot(wa[::-1],'go')
#  plt.title("Cygnss wind speed (surface wind) \n 2018-August-"+str(2)+" on longitude "+str(lons[899])+" degree")
#  plt.xlabel("Latitude(deg)")
#  plt.ylabel("wind speed(ms-1)")
##  #plt.savefig('/netdata/R1/data/wgeethma/output_microwindSpeed/2014-dec/2014-Dec-'+str(i+1)+'.png')
##  #plt.clf()
#  plt.show()

# 
