import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
import cartopy.crs as ccrs
import calendar

#####Constants
Cp = 1004           #J/kg/K
Rd = 287            #J/kg/K

def mer(y):
    
    globals()['sfcT']     = []
    globals()['seaP']     = []
    globals()['sfcP']     = []
    globals()['theta']    = []
    globals()['theta_sfc']= []
    globals()['p700T']    = []

    var = ['PS', 'SLP', 'T']
    
    for m in range(1,13):
        for j in range(1,29):
            #####SURFACE
            for i in range(0,len(var)):
                d_path = 'link/potT/merra2_sfc/MERRA2_400.inst3_3d_asm_Np.'+str(y)+f'{m:02d}'+f'{j:02d}'+'.SUB.nc'
                data   = xr.open_dataset(d_path)

                if m==1:
                    globals()['lat'] = data.variables['lat'][:]
                    globals()['lon'] = data.variables['lon'][:]
                
                if var[i]=='T':
                    globals()[var[i]+'_mer_sfc'] = data.variables[var[i]][:,:,39:140,:] #T_mer_sfc
#                     globals()[var[i]+'_mer_sfc'] = np.nanmean(globals()[var[i]+'_mer_sfc'],axis=1)
#                     globals()[var[i]+'_mer_sfc'] = np.nanmean(globals()[var[i]+'_mer_sfc'],axis=2)
#                     globals()[var[i]+'_mer_sfc'] = np.nanmean(globals()[var[i]+'_mer_sfc'],axis=1)
                    
                else:
                    globals()[var[i]+'_mer_sfc'] = data.variables[var[i]][:,39:140,:] #PS_mer_sfc
#                     globals()[var[i]+'_mer_sfc'] = np.nanmean(globals()[var[i]+'_mer_sfc'],axis=2)
#                     globals()[var[i]+'_mer_sfc'] = np.nanmean(globals()[var[i]+'_mer_sfc'],axis=1)

            T_mer_sfc_mean = np.nanmean(T_mer_sfc,axis=1)

            sfcT.extend(T_mer_sfc_mean)
            seaP.extend(SLP_mer_sfc)
            sfcP.extend(PS_mer_sfc)

            ####700 hPa unit: K
            d_path = 'link/potT/merra2/MERRA2_400.inst3_3d_asm_Np.'+str(y)+f'{m:02d}'+f'{j:02d}'+'.SUB.nc'
            data   = xr.open_dataset(d_path)

            T = data.variables['T'][:,:,39:140,:]        
#             T = np.nanmean(np.nanmean(np.nanmean(T,axis=2),axis=2),axis=1)
            p700T.extend(T)
            
        
            #####potential temperature at 700hPa
    theta = np.multiply(p700T, ((np.divide(seaP,70000))**(Rd/Cp)))

            #####potential temperature at surface unit: K
    theta_sfc = np.multiply(sfcT , (np.divide(seaP,sfcP))**(Rd/Cp))
   

        #####COLD AIR OUTBREAK INDEX unit: K
    globals()['CAO_'+str(y)] = np.subtract(theta_sfc, theta)
    
    
mer(2016)
print(np.shape(CAO_2016))