import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from mpl_toolkits.basemap import Basemap, addcyclic
import array
###########
###########
fn = '/netdata/R1/data/wgeethma/data_cygnss/2018/cyg.ddmi.s20180802-003000-e20180802-233000.l3.grid-wind.a30.d31.nc'
ds = nc.Dataset(fn)
lons			= ds.variables['lon'][:]           #size=1800 #Range is 0.1 .. 359.9
lats			= ds.variables['lat'][:]        #size=400 #Range is -39.9 .. 39.9
u		      = ds.variables['wind_speed'][:]    #float wind_speed(time, lat, lon) #units = "m s-1"
time      = ds.variables['time'][:]          #size=24
ds.close
#############
#############
for i in range(1):
  #print(i)
  fig = plt.figure(figsize=(12,6))
  # syntax for 3-D projection
#  ax = plt.axes(projection ='3d')
#  # defining all 3 axes
#  z = u[10, ::-1, :]
#  x = lats
#  y =z * np.cos(25 * z)s
#
#  # plotting
#  ax.plot3D(x, y, z, 'green')
#  ax.set_title('3D line plot geeks for geeks')
#  plt.show()
#  m = Basemap(projection='mill',lon_0=180) # plot coastlines, draw label meridians and parallels.
#  m.drawcoastlines()
#  m.drawparallels(np.arange(-90,90,10),labels=[1,0,0,0])
#  m.drawmeridians(np.arange(m.lonmin,m.lonmax+30,30),labels=[0,0,0,1])
#  x, y      = m(*np.meshgrid(lons,lats))
  x = np.arange(-39.9,40.,1.)
  w        = u[10, ::-1, 899]
  print(lons[899])
  if w[300]==0:
    print('thats zero')
  else:
    print(w[300]*10000000000000000.)
  #print(lats[::-1])
  #print(lats[::-1].shape)
  #print(lons[1200])
  plt.plot(np.arange(-39.9,40.,0.2),w[::-1],'ro')
  #plt.plot(lats[::-1],w,'go')
##  clevsw = np.arange(2., 10., 0.5)
##  cntr_w   = m.contourf(x, y, w, clevsw, cmap='rainbow')
##  cb     = plt.colorbar(cntr_w, orientation='vertical', extendrect=True, ticks=clevsw)
##  cb.set_ticks(clevsw)
##  cb.set_ticklabels(clevsw)
##  cb.set_label('m/s')
  plt.title("Cygnss wind speed (surface wind) \n 2018-August-"+str(2)+" on longitude "+str(lons[899]))
  plt.xlabel("Latitude")
  plt.ylabel("wind speed(ms-1)")
#  #plt.savefig('/netdata/R1/data/wgeethma/output_microwindSpeed/2014-dec/2014-Dec-'+str(i+1)+'.png')
#  #plt.clf()
  plt.show()

#
