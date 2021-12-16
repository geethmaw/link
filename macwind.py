import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from mpl_toolkits.basemap import Basemap, addcyclic
import array
###########
###########
fn = '/netdata/R1/data/dmccoy4/MACLWP_daily/wind1deg_maclwpv1.201412.nc4'
ds = nc.Dataset(fn)
lons			= ds.variables['lon'][:]
lats			= ds.variables['lat'][::-1]
u		      = ds.variables['sfcwind'][:]
time      = ds.variables['time'][:]
ds.close
print ('done')
#print(time)
#
#############
#############
for i in range(len(time)):
  print(i)
  plt.figure(figsize=(12,6))
  m = Basemap(projection='mill',lon_0=180) # plot coastlines, draw label meridians and parallels.
  m.drawcoastlines()
  m.drawparallels(np.arange(-90,90,10),labels=[1,0,0,0])
  m.drawmeridians(np.arange(m.lonmin,m.lonmax+30,30),labels=[0,0,0,1])
  x, y      = m(*np.meshgrid(lons,lats))
  w        = u[i, ::-1, :]
  clevsw = np.arange(2., 10., 0.5)
  cntr_w   = m.contourf(x, y, w, clevsw, cmap='rainbow')
  cb     = plt.colorbar(cntr_w, orientation='vertical', extendrect=True, ticks=clevsw)
  cb.set_ticks(clevsw)
  cb.set_ticklabels(clevsw)
  cb.set_label('m/s')
  plt.title("MicroWave wind speed (surface wind) \n 2014-December-"+str(i+1))
#  plt.savefig('/netdata/R1/data/wgeethma/output_microwindSpeed/2014-dec/2014-Dec-'+str(i+1)+'.png')
#  plt.clf()
#
# 
#plt.show()