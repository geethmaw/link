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
#fn      = '/netdata/R1/data/wgeethma/data_cygnss/2018/cyg.ddmi.s20180803-003000-e20180803-233000.l3.grid-wind.a30.d31.nc'
amf = []
cygf = []
coem = []
coec = []
#############2018 august data##########
for x in range(2,3):
    aaa = x
    for i in range(5,7):
        bbb = i

        fn      = '/netdata/R1/data/wgeethma/data_cygnss/2019/cyg.ddmi.s2019'+f'{aaa:02d}'+f'{bbb:02d}'+'-003000-e2019'+f'{aaa:02d}'+f'{bbb:02d}'+'-233000.l3.grid-wind.a30.d31.nc'
        dsi     = nc.Dataset(fn)
        # ds.append(dsi)
        lons	= dsi.variables['lon'][:]           #size=1800 #Range is 0.1 .. 359.9
        lats	= dsi.variables['lat'][:]        #size=400 #Range is -39.9 .. 39.9
        u		= dsi.variables['wind_speed'][:,:,:]    #float wind_speed(time, lat, lon) #units = "m s-1"
        time    = dsi.variables['time'][:]          #size=24
        dsi.close
    ##############
    #     fna     = '/netdata/R1/data/wgeethma/data_amsr2/2019/RSS_AMSR2_ocean_L3_daily_2019-'+f'{x:02d}'+'-'+f'{i:02d}'+'_v08.2.nc'
    #     dsa     = nc.Dataset(fna)
    #     lonsa   = dsa.variables['lon'][:]          #size=1440  #Range is 0.125 .. 359.875
    #     latsa	= dsa.variables['lat'][:]          #size=720 #Range is -89.875 .. 89.875
    #     ua      = dsa.variables['wind_speed_LF'][:]  #float32 wind_speed_LF(pass, lat, lon)
    #     timea   = dsa.variables['time'][:]            #(pass, lat, lon) shape=(2,720,1440)
    #     passa   = dsa.variables['pass'][:]
    #     dsa.close
    # # ############
    # ##############
        # x = lons
        x = np.array(lons)
        print(fn)
        # y = np.array(lats)
        # z = np.array(time)
