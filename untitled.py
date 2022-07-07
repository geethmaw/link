from __future__ import absolute_import
from __future__ import print_function
from numpy import *
from six.moves import range
import numpy.ma as ma
pp_path_cisl='/glade/collections/cmip/'

def read_var_mod(modn='CNRM-CM6-1', consort='CNRM-CERFACS', varnm='cli', cmip='cmip6', exper='historical', ensmem='r1i1p1f2', typevar='Amon', gg='gr', read_p=False, time1=[1980, 1, 15], time2=[2005, 12, 31]):
    if cmip == 'cmip6':
        MIP = 'CMIP'
        if 'ssp' in exper:
            MIP = 'ScenarioMIP'
        if exper=='amip-p4K':
            MIP = 'CFMIP'
        pth = pp_path_cisl+'/CMIP6/'+MIP+'/'+consort+'/'+modn + \
            '/'+exper+'/'+ensmem+'/'+typevar+'/'+varnm+'/'+gg+'/'
        
    data, P, lat, lon, time = read_hs(
        pth, varnm, read_p=read_p, time1=time1, time2=time2)
    
    