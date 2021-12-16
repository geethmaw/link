import netCDF4 as nc
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
# from mpl_toolkits.basemap import Basemap, addcyclic
import array
from mpl_toolkits import mplot3d
from scipy.interpolate import RegularGridInterpolator
import math
from tempfile import TemporaryFile
###########
###########
#fn      = '/netdata/R1/data/wgeethma/data_cygnss/2018/cyg.ddmi.s20180803-003000-e20180803-233000.l3.grid-wind.a30.d31.nc'
r = []
bias = []
n = 1 #month
m = 11 #date consider FEBRUARY
# tmp_file = TemporaryFile()
#############2019 data##########
for x in range(n,n+1):  #month
    x_temp = x
    for i in range(m,m+10): #date
        i_temp = i
        fn      = '/glade/work/geethma/research/data_cygnss/2019/cyg.ddmi.s2019'+f'{x_temp:02d}'+f'{i_temp:02d}'+'-003000-e2019'+f'{x_temp:02d}'+f'{i_temp:02d}'+'-233000.l3.grid-wind.a30.d31.nc'
        # fn      = '/glade/work/geethma/research/data_cygnss/2019/cyg.ddmi.s2019'+'01'+'01'+'-003000-e2019'+'01'+'01'+'-233000.l3.grid-wind.a30.d31.nc'
        # sftp://cheyenne.ucar.edu/glade/work/geethma/research/data_cygnss/2019/cyg.ddmi.s20190101-003000-e20190101-233000.l3.grid-wind.a30.d31.nc
        dsi     = nc.Dataset(fn)
        # ds.append(dsi)
        lons	= dsi.variables['lon'][:]           #size=1800 #Range is 0.1 .. 359.9
        lats	= dsi.variables['lat'][:]        #size=400 #Range is -39.9 .. 39.9
        u		= dsi.variables['wind_speed'][:,:,:]    #float wind_speed(time, lat, lon) #units = "m s-1"
        time    = dsi.variables['time'][:]          #size=24
        dsi.close
    ##############
        fna     = '/glade/work/geethma/research/data_amsr2/2019/RSS_AMSR2_ocean_L3_daily_2019-'+f'{x_temp:02d}'+'-'+f'{i_temp:02d}'+'_v08.2.nc'
        # sftp://cheyenne.ucar.edu/glade/work/geethma/research/data_amsr2/2014
        dsa     = nc.Dataset(fna)
        lonsa   = dsa.variables['lon'][:]          #size=1440  #Range is 0.125 .. 359.875
        latsa	= dsa.variables['lat'][:]          #size=720 #Range is -89.875 .. 89.875
        ua      = dsa.variables['wind_speed_LF'][:]  #float32 wind_speed_LF(pass, lat, lon)
        timea   = dsa.variables['time'][:]            #(pass, lat, lon) shape=(2,720,1440)
        passa   = dsa.variables['pass'][:]
        raina   = dsa.variables['rain_rate'][:]
        dsa.close
    # ############
    # ##############
        x = lons
        x = np.array(lons)
        y = np.array(lats)
        z = np.array(time)
        zg, yg, xg = np.meshgrid(z, y, x, indexing='ij', sparse=True)
        my_interpolate = RegularGridInterpolator((z,y,x), u)
        # #############
        xa = np.array(lonsa)
        ya = np.array(latsa)
        za = np.array(passa)
        zga, yga, xga = np.meshgrid(za, ya, xa, indexing='ij', sparse=True)
        dataa = np.array(raina)
        my_interpolate_a = RegularGridInterpolator((za,ya,xa), ua) #wind as a function [1:2,-89.9:89.9,0.1:359]
        my_interpolate_r = RegularGridInterpolator((za,ya,xa), raina) #rain_rate
        ##########
        count = 1
        am = []
        cyg = []
        for i in range(0,24): #time_cyg (0,24)
            for j in range(1,3): #pass (1,3)
                for k in range(-39,40): #lats(-39,40)
                    for l in range(1,360): #lons(1,360)
                        pt  = [i,k,l]
                        pta = [j,k,l]
                        cygn = my_interpolate(pt)
                        amsr = my_interpolate_a(pta)
                        rr = my_interpolate_r(pta)
                        #print(cygn)
                        if cygn>0 and amsr>0:
                            t = abs(amsr - cygn)
                            if t<=1 and rr>0:
                              b = math.sqrt(abs((cygn**2)-(amsr**2)))
                              bias.append(b)
                              r.append(rr)
                              cyg.append(cygn[0])
                              am.append(amsr[0])
                              np.savez_compressed('/glade/work/geethma/research/npzfiles/u10_'+f'{x_temp:02d}'+f'{i_temp:02d}'+'2019', cw=cyg, aw=am, b=bias, ar=r) #u10_monthdateyear.npz
# np.savez_compressed('/glade/work/geethma/research/tmp', cyg, am, bias, r)
# fig = plt.figure(figsize=(12,6))
# plt.scatter(r,bias)
# plt.xlabel('Rain rate (mm h-1)')
# plt.ylabel('Bias of cygnss and asmsr2 winds (ms-1)')
# plt.title('Bias Vs Rain rate')
# plt.show()
