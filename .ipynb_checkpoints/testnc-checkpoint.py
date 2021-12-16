import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import array
from scipy.interpolate import RegularGridInterpolator
import sys
import math
###########
###########
#fn      = '/netdata/R1/data/wgeethma/data_cygnss/2018/cyg.ddmi.s20180803-003000-e20180803-233000.l3.grid-wind.a30.d31.nc'
# amf = []
# cygf = []
# coem = []
# coec = []
#############2019 Jan data##########

    # print('in')
# fn      = '/glade/work/geethma/research/data_cygnss/2019/cyg.ddmi.s201901'+f'{day:02d}'+'-003000-e201901'+f'{day:02d}'+'-233000.l3.grid-wind.a30.d31.nc'

fn      ='/glade/work/geethma/research/MACLWP_dailymean/wind1deg_maclwpv1.201601.nc4'
dsi     = nc.Dataset(fn)
# ds.append(dsi)
lons	= dsi.variables['lon'][:]-180           #size=1800 #Range is 0.1 .. 359.9
lats	= dsi.variables['lat'][:]        #size=400 #Range is -39.9 .. 39.9
# u		= dsi.variables['wind_speed'][:,:,:]    #float wind_speed(time, lat, lon) #units = "m s-1"
time    = dsi.variables['time'][:]          #size=24
dsi.close
# print(lons[0])
# print(lats[0])
# print(lons[-1])
# print(lats[-1])
# print(time[0])
print(time)
