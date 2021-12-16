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
def main(day):
    # print('in')
    fn      = '/glade/work/geethma/research/data_cygnss/2019/cyg.ddmi.s201901'+f'{day:02d}'+'-003000-e201901'+f'{day:02d}'+'-233000.l3.grid-wind.a30.d31.nc'
    dsi     = nc.Dataset(fn)
    # ds.append(dsi)
    lons	= dsi.variables['lon'][:]           #size=1800 #Range is 0.1 .. 359.9
    lats	= dsi.variables['lat'][:]        #size=400 #Range is -39.9 .. 39.9
    u		= dsi.variables['wind_speed'][:,:,:]    #float wind_speed(time, lat, lon) #units = "m s-1"
    time    = dsi.variables['time'][:]          #size=24
    dsi.close
##############
    fna     = '/glade/work/geethma/research/data_amsr2/2019/RSS_AMSR2_ocean_L3_daily_2019-01-'+f'{day:02d}'+'_v08.2.nc'
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
    # x = lons
    x = np.array(lons)
    y = np.array(lats)
    z = np.array(time)
    # zg, yg, xg = np.meshgrid(z,y, x, indexing='ij', sparse=True)
    # data = u
    my_interpolate = RegularGridInterpolator((z,y,x), u)  #a function [0:23,-39.9:39.9,0.1:359]
    # #############
    xa = np.array(lonsa)
    ya = np.array(latsa)
    za = np.array(passa)
    # zga, yga, xga = np.meshgrid(za, ya, xa, indexing='ij', sparse=True)
    # dataa = np.array(ua)
    my_interpolate_a = RegularGridInterpolator((za,ya,xa), ua)#a function [1:2,-89.9:89.9,0.1:359]
    my_interpolate_t = RegularGridInterpolator((za,ya,xa), timea)
    my_interpolate_r = RegularGridInterpolator((za,ya,xa), raina) #rain_rate
    ##########
    # count = 1
    am = []
    cyg = []
    r = []
    bias = []
    cT = []
    aT = []
    longt = []
    latit = []
    for i in range(0,24): #time_cyg (0,24)
        for j in range(1,3): #pass (1,3)
            for k in np.arange(-39,40,0.5): #lats(-39,40)
                for l in np.range(1,360,0.5): #lons(1,360)
                    pt  = [i,k,l]
                    pta = [j,k,l]
                    cygn = my_interpolate(pt)
                    amsr = my_interpolate_a(pta)
                    amT = my_interpolate_a(pta)
                    rr = my_interpolate_r(pta)
                    #print(cygn)
                    if cygn>0 and amsr>0:
                        t = abs(time[i] - amT)
                        if t<=0.5:
                          b = (cygn[0]-amsr[0])
                          bias.append(b)
                          r.append(rr[0])
                          cyg.append(cygn[0])
                          am.append(amsr[0])
                          cT.append(time[i])
                          aT.append(amT[0])
                          longt.append(l)
                          latit.append(k)
                        # count = count+1
    # print(len(cyg))
    # print(len(am))
    # print(len(bias))
    # print(len(r))
    # print(len(cT))
    # print(len(aT))
    np.savez_compressed('/glade/work/geethma/research/npzfilesn/2019_/january/u10_01'+f'{day:02d}'+'2019', cw=cyg, aw=am, b=bias, ar=r, cT=cT, aT=aT, lon=longt, lat=latit) #u10_monthdateyear.npz
    # am = np.array(am)
    # cyg = np.array(cyg)
    # coefficients = np.polyfit(cyg, am, 1)
    # coem.append(coefficients[0])
    # coec.append(coefficients[1])
    # print(count)
# print('not in')

if __name__ == "__main__":
    day = int(sys.argv[1])
    # b = int(sys.argv[2])
    main(day)
# m = np.mean(coem) #mean value of gradient
# c = np.mean(coec) #mean value of intercept
# print(round(m,2))
# print(round(c,2))
