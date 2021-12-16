import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
import cartopy.crs as ccrs
import calendar
import glob
from scipy import stats

def plot(y):

    cao = []
    sw  = []

    fi = glob.glob('/glade/work/geethma/research/npzfilesn/cao/'+str(y)+'/M.npz')
    fi = np.array(fi)
    di = np.load(fi[0])
    cao.extend(di['cao'])

    fnn = glob.glob('/glade/work/geethma/research/npzfilesn/macsfcwind/'+str(y)+'/m*npz')
    fnn = np.array(fnn)

    for i in range(0,len(fnn)):
        dd = np.load(fnn[i])
        sw.extend(dd['sfcW'])

    globals()['cao'] = cao
    globals()['sw'] = sw


#     fig = plt.figure()
#
#     plt.scatter(cao, sw, marker='o', label='MAC-LWP + MERRA2')
#
# #     plt.hexbin(cao, sw, gridsize=(15,15), cmap=plt.cm.Purples_r )
#     plt.xlabel('CAO [K]')
#     plt.ylabel('surface wind (m/s)')
#     plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     # plt.show()
#     plt.savefig('/glade/work/geethma/research/output/windVsM.png')

plot(2016)

print(np.shape(cao),np.shape(sw))

caof = []
swf = []

for x in range(101):
    for y in range(360):
        for i in range(336):
            if np.isnan(cao[i][x][y]*sw[i][x][y]) == False:
                caof.append(cao[i][x][y])
                swf.append(sw[i][x][y])


print(np.shape(caof),np.shape(swf))


bin_means, bin_edges, binnumber = stats.binned_statistic(caof, swf, 'mean', bins=500)
bin_width = (bin_edges[1] - bin_edges[0])
bin_centers = bin_edges[1:] - bin_width/2
plt.figure()
# plt.hist(samples, bins=500, density=True, histtype='stepfilled',
         # alpha=0.2, label='histogram of data')
plt.plot(bin_centers[300:400], bin_means[300:400], 'r.', label='MAC-LWP+MERRA2')
plt.title('Surface wind Vs Cold Air Outbreak Index (50S to 50N)')
plt.xlabel('CAO (K)')
plt.ylabel('u10 (m/s)')
plt.legend(fontsize=10)
plt.savefig('/glade/work/geethma/research/output/windVsMmean.png')
