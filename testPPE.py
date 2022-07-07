# @Author: Geethma Werapitiya <wgeethma>
# @Date:   2022-06-16T16:06:47-06:00
# @Email:  wgeethma@uwyo.edu
# @Last modified by:   wgeethma
# @Last modified time: 2022-06-16T16:20:45-06:00

import xarray as xr
import numpy as np
### PPE
enn = np.arange(201,251)
ppe_var = ['U10', 'PSL', 'T850','TREFHT']
for en in enn:
    if en != 175:
        for i in ppe_var: #TREFHT was used since no TS. Should double check this.
            d_path = '/glade/campaign/cgd/projects/ppe/cam_ppe/rerun_PPE_250/PD/PD_timeseries/PPE_250_ensemble_PD.'+f'{en:03d}'+\
            '/atm/hist/cc_PPE_250_ensemble_PD.'+f'{en:03d}'+'.h1.'+str(i)+'.nc'
            data =xr.open_dataset(d_path)
print(done)
