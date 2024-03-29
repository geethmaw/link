# @Author: Geethma Werapitiya <wgeethma>
# @Date:   2022-06-06T16:07:59-06:00
# @Email:  wgeethma@uwyo.edu
# @Last modified by:   geethmawerapitiya
# @Last modified time: 2022-07-08T02:44:15-06:00



import netCDF4 as nc
import glob
import numpy as np
import xarray as xr
import cftime

pp_path_scratch='/glade/scratch/geethma/cmip6'

#example of arguments
# level = 'surface'/'p_level'
# modn='CESM2', exper='historical', varnm='sfcWind',time1=[2010, 1, 1],time2=[2012, 12, 30]

def read_var_mod(level, modn, exper, varnm, time1, time2):
    path   = pp_path_scratch+'/'+level+'/'

    ncname = 'CMIP6.*'+modn+'.*'+exper+'*'+varnm

    fn     = np.sort(glob.glob(path+ncname+'*nc*'))
    print(len(fn))

    times = []
    data  = []

    for i in range(len(fn)):
        print(fn[i])
        f      = nc.Dataset(fn[i])
        time   = f.variables['time']

        timeout = []
        lats    = []
        lons    = []
        lev     = []

        if level=='p_level':
            lev = f.variables['plev']

        for j in range(len(time[:])):
            tt1 = nc.num2date(time[j], f.variables['time'].units,
                              calendar=f.variables['time'].calendar)
            timeout.append([int(tt1.year), int(tt1.month), int(tt1.day)])

        ind1  = -1
        ind2  = -1

        for k in range(len(timeout)):
            if timeout[k]==time1:
                ind1 = k

        for k in range(len(timeout)):
            if timeout[k]==time2:
                ind2 = k

        #print('ind1 ',ind1, 'ind2 ',ind2)

        if ind1>=0 and ind2>=0:
            #print('1')
            lats    = f.variables['lat']
            lons    = f.variables['lon']
            times.extend(timeout[ind1:ind2+1])
            #print('start day:')
            #print(timeout[ind1])
            #print('last day:')
            #print(timeout[ind2])
            datai = f.variables[varnm]
            data.extend(np.array(datai[ind1:ind2+1,:,:]))
            break


        elif ind1>=0 and ind2<0:
            #print('2')
            lats    = f.variables['lat']
            lons    = f.variables['lon']
            times.extend(timeout[ind1:])
            #print('start day:')
            #print(timeout[ind1])
            datai = f.variables[varnm]
            data.extend(np.array(f.variables[varnm][ind1:,:,:]))


        elif ind1<0 and ind2<0:
            #print('3')
            if data:
                #print('4')
                lats    = f.variables['lat']
                lons    = f.variables['lon']
                times.extend(timeout[:])
                datai = f.variables[varnm]
                data.extend(np.array(f.variables[varnm][:,:,:]))


        elif ind2>=0 and ind1<0:
            #print('5')
            lats    = f.variables['lat']
            lons    = f.variables['lon']
            times.extend(timeout[:ind2+1])
            #print('last day:')
            #print(timeout[ind2])
            datai = f.variables[varnm]
            data.extend(np.array(f.variables[varnm][:ind2+1,:,:]))
            break


        if times:
            #print('t')
            if times[0]<time1:
                print('invalid start date')

            if times[-1]>time2:
                print('invalid end date')

    return(lats,lons,times,lev,np.array(data))
