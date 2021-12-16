import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import cartopy.crs as ccrs
import array
import glob
#####cw=cyg, aw=am, b=bias, ar=r, cT=cT, aT=aT, lon=longt, lat=latit
fn = glob.glob('/glade/work/geethma/research/npzfilesn/2019/january2/u10_01062019.npz')
fn = np.array(fn)
lons=[]
lats=[]
# bf = np.zeros((5,5))
######
# print(bf)
for i in range(0,len(fn)):
    d = np.load(fn[i])
    lon = np.array(d['lon'])
    lat = np.array(d['lat'])
    b = np.array(d['b'])
    # bf = np.zeros((len(lon),len(lon)))
    bf = np.zeros((5,5))
    for i in range(0,5): #len(lon)
        # for j in range(0,len(lat)):
        lons.append(lon[i])
        lats.append(lat[i])
        # if i==j:
        bf[i][i]=b[i]
# #         # bff.append(b[i][j])
# ###########map
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
plt.contourf(lons, lats, bf, 60, transform=ccrs.PlateCarree())

plt.show()
