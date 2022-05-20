from readGCMs import read_var_mod
import numpy as np

modname  = ['CESM2','CNRM-CM6-1', 'CanESM5','CESM2-WACCM','E3SM-1-0',
       'HadGEM3-GC31-LL','IPSL-CM6A-LR','NorESM2-LM', 'SAM0-UNICON',
       'ACCESS-CM2','ACCESS-ESM1-5','CNRM-ESM2-1','EC-Earth3',
       'EC-Earth3-Veg','FGOALS-f3-L','GISS-E2-1-G-CC','HadGEM3-GC31-MM',
       'INM-CM4-8','INM-CM5-0','MPI-ESM1-2-HR','MRI-ESM2-0','NorCPM1',
       'UKESM1-0-LL']
conname  = ['NCAR', 'CNRM-CERFACS','CCCma', 'NCAR', 'E3SM-Project', 'MOHC', 
       'IPSL', 'NCC', 'SNU','CSIRO-ARCCSS','CSIRO','CNRM-CERFACS',
       'EC-Earth-Consortium','EC-Earth-Consortium','CAS','NASA-GISS','MOHC'
       ,'INM','INM','MPI-M','MRI','NCC','MOHC'] 
ensname  = ['r11i1p1f1', 'r1i1p1f2', 'r10i1p1f1', 'r1i1p1f1', 'r1i1p1f1', 
       'r1i1p1f3', 'r10i1p1f1', 'r1i1p1f1', 'r1i1p1f1','r1i1p1f1',
       'r10i1p1f1','r1i1p1f2','r101i1p1f1','r10i1p1f1','r1i1p1f1',
       'r1i1p1f1','r1i1p1f3','r1i1p1f1','r10i1p1f1','r10i1p1f1','r10i1p1f1',
       'r10i1p1f1','r10i1p1f2']
ggname   = ['gn', 'gr','gn', 'gn', 'gr', 'gn', 'gr', 'gn', 'gn','gn', 'gn', 'gr',
       'gr', 'gr', 'gr', 'gn', 'gn', 'gr1', 'gr1', 'gn', 'gr', 'gn', 'gn']

m = 0 #starting index
n = len(modname)
varname = ['sfcWind', 'tas','psl']#'sfcWind', 'hfss', 'hfls', 'tas', 'ps', 'psl',
pvarname = ['ta']


def my_gcm_op(m,n,varname,pvarname):    
    try:
        for j in range(m,n):
        
            for i in varname: 
                locals()[i+'__'+str(j+1)] = read_var_mod(modn=modname[j], consort=conname[j], varnm=i, cmip='cmip6', exper='historical', ensmem=ensname[j], typevar='day', gg=ggname[j], read_p=False, time1=[2010, 1, 1], time2=[2012, 12, 31])
            for k in pvarname: 
                locals()[k+'__'+str(j+1)] = read_var_mod(modn=modname[j], consort=conname[j], varnm=k, cmip='cmip6', exper='historical', ensmem=ensname[j], typevar='day', gg=ggname[j], read_p=True, time1=[2010, 1, 1], time2=[2012, 12, 31])
                
            print(modname[l])
        
    except IndexError:
        print(str(j+1)+' not available')

    for i in range(m,n):
        locals()['lat'+str(i+1)]  = locals()['sfcWind__'+str(i+1)][2]
        locals()['lon'+str(i+1)]  = locals()['sfcWind__'+str(i+1)][3]
        locals()['time'+str(i+1)] = locals()['sfcWind__'+str(i+1)][4]

        for j in varname:
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][0]

        for k in pvarname:
            locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][0]
            locals()['lev'+str(i+1)] = locals()['ta__'+str(i+1)][1]
            
        return 