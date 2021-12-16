import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from mpl_toolkits.basemap import Basemap, addcyclic
###########
###########
#fn = '/netdata/R1/data/wgeethma/data_cygnss/2021/cyg.ddmi.s20210101-003000-e20210101-233000.l3.grid-wind.a30.d31.nc'
fn = '/netdata/R1/data/wgeethma/data_amsr2/2018/RSS_AMSR2_ocean_L3_daily_2018-01-01_v08.2.nc'
ds = nc.Dataset(fn)
#print (ds)
#print(ds.__dict__)
#for dim in ds.dimensions.values():
#    print(dim)
#for var in ds.variables.values():
#    print(var)
#
lons			= ds.variables['lon'][:]
lats			= ds.variables['lat'][:]    
##levs			= ds.variables['lev'][:]
##hgt			  = ds.variables['PHIS'][:, ::-1, :].squeeze()
#u		      = ds.variables['wind_speed'][:]
#time      = ds.variables['time'][:]
##temp      = ds.variables['T'][:, ::-1, :].squeeze() 
#ds.close
print("lons")
print (lons[0:10])
print("lats")
print (lats[0:10])
##
##potential temperature
#pressure = np.zeros((42, 39, 68))
#Rd = 287. 
#Cpd = 1004. 
#for i in range(0, 42):
#    pressure[i,:,:]=levs[i]
#theta = temp*((100000./pressure)**(Rd/Cpd)) 
##
#u = [[1.,2.,3.],[1.,2.,3.],[1.,2.,3.]]
#i = 10
#j = 10
##print(time)
#print(np.max(u))
#z_lat = np.nanmean(np.where(u>2.,u,np.nan),axis=1) #shape=24,1800
#z_lon = np.nanmean(np.where(u>2.,u,np.nan),axis=2) #shape=24,400
#print(z_lat.shape)
#print(z_lon.shape)
#print(np.nanmean(z_lat))
##print(levs[0],levs[1],levs[-2],levs[-1])
##
#fig, ax = plt.subplots()
##
#m = Basemap(projection='merc', llcrnrlat = 18, urcrnrlat =50, llcrnrlon = 250, urcrnrlon = 300, lat_ts = 5, resolution = 'c',ax = ax)

