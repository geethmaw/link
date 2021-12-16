import netCDF4 as nc
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
# from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs
import array
from mpl_toolkits import mplot3d
from scipy.interpolate import RegularGridInterpolator
###########
###########
# fn        = '/netdata/R1/data/wgeethma/data_cygnss/2018/cyg.ddmi.s20180802-003000-e20180802-233000.l3.grid-wind.a30.d31.nc'
# ds        = nc.Dataset(fn)
# lons	  = ds.variables['lon'][:]           #size=1800 #Range is 0.1 .. 359.9
# lats   	  = ds.variables['lat'][:]        #size=400 #Range is -39.9 .. 39.9
# u		  = ds.variables['wind_speed'][:,:,:]    #float wind_speed(time, lat, lon) #units = "m s-1"
# time      = ds.variables['time'][:]          #size=24
# ds.close
# ##############
# fna       = '/netdata/R1/data/wgeethma/data_amsr2/2018/RSS_AMSR2_ocean_L3_daily_2018-08-02_v08.2.nc'
# dsa       = nc.Dataset(fna)
# lonsa	  = dsa.variables['lon'][:]          #size=1440  #Range is 0.125 .. 359.875
# latsa	  = dsa.variables['lat'][:]          #size=720 #Range is -89.875 .. 89.875
# ua        = dsa.variables['wind_speed_LF'][:]  #float32 wind_speed_LF(pass, lat, lon)
# timea     = dsa.variables['time'][:]            #(pass, lat, lon) shape=(2,720,1440)
# passa     = dsa.variables['pass'][:]
# dsa.close
# ###############figure
# fig, ax = plt.subplots(2,1)
# fig.set_size_inches(10,8)
# ###############cygnss Basemap
# ax[1].set_title('cygnss data on 02nd August 2018')
# m = Basemap(projection='mill',llcrnrlat=-50,urcrnrlat=50,llcrnrlon=0.,urcrnrlon=360.,resolution='c',ax=ax[1])
# m.drawcountries()
# m.drawcoastlines()
# m.drawparallels(np.arange(-90,90,30),labels=[1,0,0,0])
# m.drawmeridians(np.arange(m.lonmin,m.lonmax+30,30),labels=[0,0,0,1])
#
# count = 0
# ind_lat = []
# ind_lon = []
# #################CYGNSS plot
# for i in range(0,23):
#     for j in range(0,399):
#         for k in range(0,1799):
#             cygn = u[i,j,k]
#             if cygn>0:
#                 ind_lat.append(j)
#                 ind_lon.append(k)
#                 count = count + 1
#
# lons_n = lons[ind_lon]
# lats_n = lats[ind_lat]
#
# x, y = m(lons_n, lats_n)
#
# m.scatter(x, y, marker='D',color='m')
#
# #########amsr Basemap
# ax[0].set_title('AMSR data on 02nd August 2018')
# X, Y = np.meshgrid(lonsa, latsa)
# m = Basemap(projection='mill',llcrnrlat=-50,urcrnrlat=50,llcrnrlon=0.,urcrnrlon=360,resolution='c',ax=ax[0],anchor='C')
# x, y = m(X, Y)
# m.drawcountries()
# m.drawcoastlines()
# m.drawparallels(np.arange(-90,90,30),labels=[1,0,0,0])
# m.drawmeridians(np.arange(m.lonmin,m.lonmax+30,30),labels=[0,0,0,1])
# count = 0
# ind_lat = []
# ind_lon = []
# for i in range(0,1):
#     for j in range(0,719):
#         for k in range(0,1439):
#             amsr = ua[i,j,k]
#             if amsr>0:
#                 ind_lat.append(j)
#                 ind_lon.append(k)
#                 count = count + 1
#
# lons_n = lonsa[ind_lon]
# lats_n = latsa[ind_lat]
#
# x, y = m(lons_n, lats_n)
#
# m.scatter(x, y, color='m')
#
# plt.show()
