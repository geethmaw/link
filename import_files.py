# @Author: Geethma Werapitiya <wgeethma>
# @Date:   2022-08-18T09:13:44-06:00
# @Email:  wgeethma@uwyo.edu
# @Last modified by:   wgeethma
# @Last modified time: 2022-08-18T09:18:18-06:00

def import_files():
    import xarray as xr
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.cm import get_cmap
    import calendar
    from global_land_mask import globe
    import glob
    import math
    from scipy import stats
    import os
    import netCDF4 as nc

    from gcm_loading.highres_read import read_var_hires
    from gcm_loading.myReadGCMsDaily import read_var_mod
    from gcm_loading.read_amip import read_amip_var
    from regrid_wght_3d import regrid_wght_wnans

    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = 100
    plt.clf()
    plt.rcParams['figure.figsize'] = (10, 10)

    return
