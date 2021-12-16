#import nctoolkit as nc
import netCDF4 as nc
import numpy as np

infile = '/netdata/R1/data/wgeethma/data_cygnss/2021/cyg.ddmi.s20210101-003000-e20210101-233000.l3.grid-wind.a30.d31.nc'
data = nc.Dataset(infile)

data.to_latlon(lon = [0.1,359.9], lat = [-39.9,39.9], res = [1., 1.])
