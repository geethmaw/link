# @Author: Geethma Werapitiya <wgeethma>
# @Date:   2022-07-07T15:39:03-06:00
# @Email:  wgeethma@uwyo.edu
# @Last modified by:   wgeethma
# @Last modified time: 2022-07-07T16:09:59-06:00

import netCDF4 as nc
import glob
import numpy as np
import xarray as xr
import cftime

pp_path_scratch='/glade/scratch/geethma/cmip6'

def read_warming(level, modn, exper, varnm, n_yr):
    path    = pp_path_scratch+'/'+level+'/'

    ncname  = 'CMIP6.CMIP.*'+modn+'.'+exper+'.*'+varnm

    fn      = np.sort(glob.glob(path+ncname+'*nc*'))

    n       = -1

    data     = []

    num_yr = n_yr

    for i in range(len(fn)):
        f       = nc.Dataset(fn[n])
        time    = f.variables['time']
        if level=='p_level':
            lev = f.variables['plev']
        else:
            lev = []

        if n == -1:
            if len(time) > n_yr:
                # print(fn[n])
                f       = nc.Dataset(fn[n])
                lats    = f.variables['lat']
                lons    = f.variables['lon']
                datai   = f.variables[varnm]
                data.extend(np.array(datai[-n_yr::,:,:]))
                break

        if len(time) <= num_yr:
            # print(fn[n])
            f       = nc.Dataset(fn[n])
            lats    = f.variables['lat']
            lons    = f.variables['lon']
            datai   = f.variables[varnm]
            data.extend(np.array(datai))
            n       = n-1
            num_yr  = num_yr - len(time)

        if len(time) > num_yr:
            # print(fn[n])
            f       = nc.Dataset(fn[n])
            lats    = f.variables['lat']
            lons    = f.variables['lon']
            datai   = f.variables[varnm]
            data.extend(np.array(datai[-num_yr::,:,:]))
            break


    return(lats,lons,lev,np.array(data))
