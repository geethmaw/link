# @Author: Daniel McCoy
# @Date:   2022-06-30T15:42:33-06:00
# @Email:  wgeethma@uwyo.edu
# @Last modified by:   wgeethma
# @Last modified time: 2022-07-07T16:03:46-06:00



from numpy import *
from scipy.stats import *
# def regrid_wght(lat,lon,Z,latn,lonn):
# 	### assumes Z=time,lat,lon
# 	[lon,lat]=meshgrid(lon,lat)
# 	wght=cos(lat/180*pi)
# 	zout=zeros((Z.shape[0],len(latn)-1,len(lonn)-1))
# 	print (wght.shape)
# 	print (Z[0].shape)
# 	LON=lon.flatten(); LAT=lat.flatten();
# 	for i in range(Z.shape[0]):
# 		zz=binned_statistic_2d(LAT,LON,(Z[i]*wght).flatten(),bins=(latn,lonn),statistic='sum')
# 		ww=binned_statistic_2d(LAT,LON,wght.flatten(),bins=(latn,lonn),statistic='sum')
# 		zout[i]=zz[0]/ww[0]
# 		#latn=(latn[1:]+latn[0:-1])/2
# 		#lonn=(latn[1:]+latn[0:-1])/2
# 	latno=binned_statistic_2d(LAT,LON,LAT,bins=(latn,lonn))
# 	lonno=binned_statistic_2d(LAT,LON,LON,bins=(latn,lonn))
#
# 	return zout,lonno[0],latno[0]


def regrid_wght_wnans(lat,lon,Z,latn,lonn):
	time=arange(0,Z.shape[0])
	[lon,lat]=meshgrid(lon,lat)
	wght=cos(lat/180*pi)
	zout=zeros((Z.shape[0],len(latn)-1,len(lonn)-1))
	# print (wght.shape)
	# print (Z[0].shape)
	LON=lon#.flatten();
	LAT=lat#.flatten();
	for i in range(Z.shape[0]):
		Zt=Z[i]
		ind=isnan(Zt)==False
		if len(where(ind)[0])>0:
			zz=binned_statistic_2d(LAT[ind],LON[ind],(Zt[ind]*wght[ind]).flatten(),bins=(latn,lonn),statistic='sum')
			ww=binned_statistic_2d(LAT[ind],LON[ind],wght[ind].flatten(),bins=(latn,lonn),statistic='sum')
			zout[i]=zz[0]/ww[0]
		else:
			zout[i]=NaN
	latno=binned_statistic_2d(LAT.flatten(),LON.flatten(),LAT.flatten(),bins=(latn,lonn))
	lonno=binned_statistic_2d(LAT.flatten(),LON.flatten(),LON.flatten(),bins=(latn,lonn))

	return zout,lonno[0],latno[0]
