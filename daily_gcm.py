from __future__ import absolute_import
from __future__ import print_function
from numpy import *
from six.moves import range
import numpy.ma as ma
pp_path_cisl='/glade/collections/cmip/CMIP6/CMIP/'

gcm_name = ['HadGEM3-GC31-LL']


###gcm='HadGEM3-GC31-LL', exper='historical',
end = 0
for i in varname: 
    import glob
    ncname = i+'_day_'+gcm+'_historical_r1i1p1f3_gn_19500101-20141230.nc'
    d_path = '/glade/collections/cmip/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/historical/r1i1p1f3/day/'+i+'/gn/v20190624/'+i+'/'+ncname
    data =xr.open_dataset(d_path)

    if end == 0:
        lon  = data.variables['lon'][:]  #(lon: 288) [0.0, 1.25, 2.5, ... 356.25, 357.5, 358.75]
        lat  = data.variables['lat'][:]  #(lat: 192) [-90.0 , -89.057592, -88.115183, ... 88.115183,  89.057592, 90.0]
        time = data.variables['time'][:] #(time: 36)

    if i == 'ta':
        lev  = data.variables['plev'][:]
    
    locals()[i+'_0'] = data.variables[i][:]