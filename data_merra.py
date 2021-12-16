import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from mpl_toolkits.basemap import Basemap, addcyclic
import array
###########
###########
m = Basemap(projection='robin',lon_0=0,resolution='c')
m.drawcoastlines()
#m.fillcontinents(color='coral',lake_color='aqua')
# draw parallels and meridians.
m.drawparallels(np.arange(-90.,120.,30.))
m.drawmeridians(np.arange(0.,360.,60.))
m.drawmapboundary()
#
clevsw = np.arange(2, 14, 1)
for i in range(28):
  if i<9:
    fn = '/netdata/R1/data/wgeethma/data_merra/MERRA2_400.tavg1_2d_flx_Nx.2021020'+str(i+1)+'.nc4.nc4'
    print(fn)
  else:
    fn = '/netdata/R1/data/wgeethma/data_merra/MERRA2_400.tavg1_2d_flx_Nx.202102'+str(i+1)+'.nc4.nc4'
    print(fn)
  ds = nc.Dataset(fn)
  w_time = 0
  lons			= ds.variables['lon'][:]
  lats			= ds.variables['lat'][::-1]    
  w         = ds.variables['SPEED'][w_time, ::-1, :].squeeze()
  ds.close
  x, y      = m(*np.meshgrid(lons,lats))
  cntr_w    = m.contourf(x, y, w, clevsw, cmap='PiYG')
#levs			= ds.variables['lev'][:]
##hgt			  = ds.variables['PHIS'][:, ::-1, :].squeeze()
##u		      = ds.variables['U'][:, ::-1, :].squeeze()
##temp      = ds.variables['T'][:, ::-1, :].squeeze() 
#time      = ds.variables['time'][:]
#w         = ds.variables['SPEED'][w_time, ::-1, :].squeeze()
#ds.close
print ('done')
#print (np.amin(w))
#print (np.amax(w))
#print (len(w))
#
##

##cntrv = map.contourf(x, y, vort, clevsv, extend='both', cmap='PiYG' ) #vorticity
cb     = plt.colorbar(cntr_w, orientation='vertical', extendrect=True, ticks=clevsw)
cb.set_ticks(clevsw)
cb.set_ticklabels(clevsw)
cb.set_label('m/s')
##
plt.title("MERRA-2 wind speed \n 2021-February")
plt.savefig('/netdata/R1/data/wgeethma/data_merra/2021-February.png')
plt.show()
