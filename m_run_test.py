# @Author: Geethma Werapitiya <wgeethma>
# @Date:   2022-07-17T14:34:35-06:00
# @Email:  wgeethma@uwyo.edu
# @Last modified by:   wgeethma
# @Last modified time: 2022-07-18T13:47:07-06:00

# import xarray as xr
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.cm import get_cmap
# import calendar
# from global_land_mask import globe
# import glob
# import math
# from scipy import stats
# import os
# import netCDF4 as nc
#
# from highres_read import read_var_hires
# from myReadGCMsDaily import read_var_mod
# from regrid_wght_3d import regrid_wght_wnans
#
# import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 100
# plt.clf()
# plt.rcParams['figure.figsize'] = (10, 10)
#
# from con_models import get_cons
# con, use_colors, varname, pvarname, modname, warming_modname, hiresmd, amip_md = get_cons()
#
# #latitude range
# latr1 = 30
# latr2 = 80

modname = warming_modname

DJF = []
DJF_lats = []

test_num = 0

if test_num<=23:
    ########################### 0,1
    time1=[2010, 1, 1]
    time2=[2010, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2010, 12, 1]
    time2=[2011, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2011, 12, 1]
    time2=[2012, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,4
    time1=[2012, 12, 1]
    time2=[2012, 12, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    DJF.append(f)
    DJF_lats.append(lats)




    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='solid')

    plt.legend()
    plt.xlabel('latitude')
    plt.title('seasonal CAO zonal mean ditribution')
    plt.savefig('../figures/final/M_lats.png')
#################################################################################################################################

    ############SUMMER
    ########################### 0,1
    time1=[2010, 6, 1]
    time2=[2010, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2011, 6, 1]
    time2=[2011, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2012, 6, 1]
    time2=[2012, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='dashed')
    plt.scatter(lats,f,color=use_colors[test_num],label=modname[test_num],marker='x')

    plt.savefig('../figures/final/M_lats.png')

else:
    print(test_num)


# @Author: Geethma Werapitiya <wgeethma>
# @Date:   2022-07-17T14:34:35-06:00
# @Email:  wgeethma@uwyo.edu
# @Last modified by:   wgeethma
# @Last modified time: 2022-07-18T13:47:07-06:00

# import xarray as xr
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.cm import get_cmap
# import calendar
# from global_land_mask import globe
# import glob
# import math
# from scipy import stats
# import os
# import netCDF4 as nc
#
# from highres_read import read_var_hires
# from myReadGCMsDaily import read_var_mod
# from regrid_wght_3d import regrid_wght_wnans
#
# import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 100
# plt.clf()
# plt.rcParams['figure.figsize'] = (10, 10)
#
# from con_models import get_cons
# con, use_colors, varname, pvarname, modname, warming_modname, hiresmd, amip_md = get_cons()
#
# #latitude range
# latr1 = 30
# latr2 = 80

test_num = test_num+1

if test_num<=23:
    ########################### 0,1
    time1=[2010, 1, 1]
    time2=[2010, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2010, 12, 1]
    time2=[2011, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2011, 12, 1]
    time2=[2012, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,4
    time1=[2012, 12, 1]
    time2=[2012, 12, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='solid')

    plt.legend()
    plt.savefig('../figures/final/M_lats.png')


    ############SUMMER
    ########################### 0,1
    time1=[2010, 6, 1]
    time2=[2010, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2011, 6, 1]
    time2=[2011, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2012, 6, 1]
    time2=[2012, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='dashed')
    plt.scatter(lats,f,color=use_colors[test_num],label=modname[test_num],marker='x')

    plt.savefig('../figures/final/M_lats.png')

else:
    print(test_num)




# @Author: Geethma Werapitiya <wgeethma>
# @Date:   2022-07-17T14:34:35-06:00
# @Email:  wgeethma@uwyo.edu
# @Last modified by:   wgeethma
# @Last modified time: 2022-07-18T13:47:07-06:00

# import xarray as xr
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.cm import get_cmap
# import calendar
# from global_land_mask import globe
# import glob
# import math
# from scipy import stats
# import os
# import netCDF4 as nc
#
# from highres_read import read_var_hires
# from myReadGCMsDaily import read_var_mod
# from regrid_wght_3d import regrid_wght_wnans
#
# import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 100
# plt.clf()
# plt.rcParams['figure.figsize'] = (10, 10)
#
# from con_models import get_cons
# con, use_colors, varname, pvarname, modname, warming_modname, hiresmd, amip_md = get_cons()
#
# #latitude range
# latr1 = 30
# latr2 = 80

test_num = test_num+1

if test_num<=23:
    ########################### 0,1
    time1=[2010, 1, 1]
    time2=[2010, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2010, 12, 1]
    time2=[2011, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2011, 12, 1]
    time2=[2012, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,4
    time1=[2012, 12, 1]
    time2=[2012, 12, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='solid')

    plt.legend()
    plt.xlabel('latitude')
    plt.title('seasonal CAO zonal mean ditribution')
    plt.savefig('../figures/final/M_lats.png')


    ############SUMMER
    ########################### 0,1
    time1=[2010, 6, 1]
    time2=[2010, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2011, 6, 1]
    time2=[2011, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2012, 6, 1]
    time2=[2012, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='dashed')
    plt.scatter(lats,f,color=use_colors[test_num],label=modname[test_num],marker='x')

    plt.savefig('../figures/final/M_lats.png')

else:
    print(test_num)


# @Author: Geethma Werapitiya <wgeethma>
# @Date:   2022-07-17T14:34:35-06:00
# @Email:  wgeethma@uwyo.edu
# @Last modified by:   wgeethma
# @Last modified time: 2022-07-18T13:47:07-06:00

# import xarray as xr
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.cm import get_cmap
# import calendar
# from global_land_mask import globe
# import glob
# import math
# from scipy import stats
# import os
# import netCDF4 as nc
#
# from highres_read import read_var_hires
# from myReadGCMsDaily import read_var_mod
# from regrid_wght_3d import regrid_wght_wnans
#
# import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 100
# plt.clf()
# plt.rcParams['figure.figsize'] = (10, 10)
#
# from con_models import get_cons
# con, use_colors, varname, pvarname, modname, warming_modname, hiresmd, amip_md = get_cons()
#
# #latitude range
# latr1 = 30
# latr2 = 80

test_num = test_num+1

if test_num<=23:
    ########################### 0,1
    time1=[2010, 1, 1]
    time2=[2010, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2010, 12, 1]
    time2=[2011, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2011, 12, 1]
    time2=[2012, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,4
    time1=[2012, 12, 1]
    time2=[2012, 12, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='solid')

    plt.legend()
    plt.savefig('../figures/final/M_lats.png')


    ############SUMMER
    ########################### 0,1
    time1=[2010, 6, 1]
    time2=[2010, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2011, 6, 1]
    time2=[2011, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2012, 6, 1]
    time2=[2012, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='dashed')
    plt.scatter(lats,f,color=use_colors[test_num],label=modname[test_num],marker='x')

    plt.savefig('../figures/final/M_lats.png')

else:
    print(test_num)


# @Author: Geethma Werapitiya <wgeethma>
# @Date:   2022-07-17T14:34:35-06:00
# @Email:  wgeethma@uwyo.edu
# @Last modified by:   wgeethma
# @Last modified time: 2022-07-18T13:47:07-06:00

# import xarray as xr
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.cm import get_cmap
# import calendar
# from global_land_mask import globe
# import glob
# import math
# from scipy import stats
# import os
# import netCDF4 as nc
#
# from highres_read import read_var_hires
# from myReadGCMsDaily import read_var_mod
# from regrid_wght_3d import regrid_wght_wnans
#
# import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 100
# plt.clf()
# plt.rcParams['figure.figsize'] = (10, 10)
#
# from con_models import get_cons
# con, use_colors, varname, pvarname, modname, warming_modname, hiresmd, amip_md = get_cons()
#
# #latitude range
# latr1 = 30
# latr2 = 80

test_num = test_num+1

if test_num<=23:
    ########################### 0,1
    time1=[2010, 1, 1]
    time2=[2010, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2010, 12, 1]
    time2=[2011, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2011, 12, 1]
    time2=[2012, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,4
    time1=[2012, 12, 1]
    time2=[2012, 12, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='solid')

    plt.legend()
    plt.xlabel('latitude')
    plt.title('seasonal CAO zonal mean ditribution')
    plt.savefig('../figures/final/M_lats.png')


    ############SUMMER
    ########################### 0,1
    time1=[2010, 6, 1]
    time2=[2010, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2011, 6, 1]
    time2=[2011, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2012, 6, 1]
    time2=[2012, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='dashed')
    plt.scatter(lats,f,color=use_colors[test_num],label=modname[test_num],marker='x')

    plt.savefig('../figures/final/M_lats.png')

else:
    print(test_num)


# @Author: Geethma Werapitiya <wgeethma>
# @Date:   2022-07-17T14:34:35-06:00
# @Email:  wgeethma@uwyo.edu
# @Last modified by:   wgeethma
# @Last modified time: 2022-07-18T13:47:07-06:00

# import xarray as xr
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.cm import get_cmap
# import calendar
# from global_land_mask import globe
# import glob
# import math
# from scipy import stats
# import os
# import netCDF4 as nc
#
# from highres_read import read_var_hires
# from myReadGCMsDaily import read_var_mod
# from regrid_wght_3d import regrid_wght_wnans
#
# import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 100
# plt.clf()
# plt.rcParams['figure.figsize'] = (10, 10)
#
# from con_models import get_cons
# con, use_colors, varname, pvarname, modname, warming_modname, hiresmd, amip_md = get_cons()
#
# #latitude range
# latr1 = 30
# latr2 = 80

test_num = test_num+1

if test_num<=23:
    ########################### 0,1
    time1=[2010, 1, 1]
    time2=[2010, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2010, 12, 1]
    time2=[2011, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2011, 12, 1]
    time2=[2012, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,4
    time1=[2012, 12, 1]
    time2=[2012, 12, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='solid')

    plt.legend()
    plt.savefig('../figures/final/M_lats.png')


    ############SUMMER
    ########################### 0,1
    time1=[2010, 6, 1]
    time2=[2010, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2011, 6, 1]
    time2=[2011, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2012, 6, 1]
    time2=[2012, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='dashed')
    plt.scatter(lats,f,color=use_colors[test_num],label=modname[test_num],marker='x')

    plt.savefig('../figures/final/M_lats.png')

else:
    print(test_num)

# @Author: Geethma Werapitiya <wgeethma>
# @Date:   2022-07-17T14:34:35-06:00
# @Email:  wgeethma@uwyo.edu
# @Last modified by:   wgeethma
# @Last modified time: 2022-07-18T13:47:07-06:00

# import xarray as xr
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.cm import get_cmap
# import calendar
# from global_land_mask import globe
# import glob
# import math
# from scipy import stats
# import os
# import netCDF4 as nc
#
# from highres_read import read_var_hires
# from myReadGCMsDaily import read_var_mod
# from regrid_wght_3d import regrid_wght_wnans
#
# import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 100
# plt.clf()
# plt.rcParams['figure.figsize'] = (10, 10)
#
# from con_models import get_cons
# con, use_colors, varname, pvarname, modname, warming_modname, hiresmd, amip_md = get_cons()
#
# #latitude range
# latr1 = 30
# latr2 = 80

test_num = test_num+1

if test_num<=23:
    ########################### 0,1
    time1=[2010, 1, 1]
    time2=[2010, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2010, 12, 1]
    time2=[2011, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2011, 12, 1]
    time2=[2012, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,4
    time1=[2012, 12, 1]
    time2=[2012, 12, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='solid')

    plt.legend()
    plt.xlabel('latitude')
    plt.title('seasonal CAO zonal mean ditribution')
    plt.savefig('../figures/final/M_lats.png')


    ############SUMMER
    ########################### 0,1
    time1=[2010, 6, 1]
    time2=[2010, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2011, 6, 1]
    time2=[2011, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2012, 6, 1]
    time2=[2012, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='dashed')
    plt.scatter(lats,f,color=use_colors[test_num],label=modname[test_num],marker='x')

    plt.savefig('../figures/final/M_lats.png')

else:
    print(test_num)


# @Author: Geethma Werapitiya <wgeethma>
# @Date:   2022-07-17T14:34:35-06:00
# @Email:  wgeethma@uwyo.edu
# @Last modified by:   wgeethma
# @Last modified time: 2022-07-18T13:47:07-06:00

# import xarray as xr
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.cm import get_cmap
# import calendar
# from global_land_mask import globe
# import glob
# import math
# from scipy import stats
# import os
# import netCDF4 as nc
#
# from highres_read import read_var_hires
# from myReadGCMsDaily import read_var_mod
# from regrid_wght_3d import regrid_wght_wnans
#
# import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 100
# plt.clf()
# plt.rcParams['figure.figsize'] = (10, 10)
#
# from con_models import get_cons
# con, use_colors, varname, pvarname, modname, warming_modname, hiresmd, amip_md = get_cons()
#
# #latitude range
# latr1 = 30
# latr2 = 80

test_num = test_num+1

if test_num<=23:
    ########################### 0,1
    time1=[2010, 1, 1]
    time2=[2010, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2010, 12, 1]
    time2=[2011, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2011, 12, 1]
    time2=[2012, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,4
    time1=[2012, 12, 1]
    time2=[2012, 12, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='solid')

    plt.legend()
    plt.savefig('../figures/final/M_lats.png')


    ############SUMMER
    ########################### 0,1
    time1=[2010, 6, 1]
    time2=[2010, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2011, 6, 1]
    time2=[2011, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2012, 6, 1]
    time2=[2012, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='dashed')
    plt.scatter(lats,f,color=use_colors[test_num],label=modname[test_num],marker='x')

    plt.savefig('../figures/final/M_lats.png')

else:
    print(test_num)

# @Author: Geethma Werapitiya <wgeethma>
# @Date:   2022-07-17T14:34:35-06:00
# @Email:  wgeethma@uwyo.edu
# @Last modified by:   wgeethma
# @Last modified time: 2022-07-18T13:47:07-06:00

# import xarray as xr
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.cm import get_cmap
# import calendar
# from global_land_mask import globe
# import glob
# import math
# from scipy import stats
# import os
# import netCDF4 as nc
#
# from highres_read import read_var_hires
# from myReadGCMsDaily import read_var_mod
# from regrid_wght_3d import regrid_wght_wnans
#
# import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 100
# plt.clf()
# plt.rcParams['figure.figsize'] = (10, 10)
#
# from con_models import get_cons
# con, use_colors, varname, pvarname, modname, warming_modname, hiresmd, amip_md = get_cons()
#
# #latitude range
# latr1 = 30
# latr2 = 80

test_num = test_num+1

if test_num<=23:
    ########################### 0,1
    time1=[2010, 1, 1]
    time2=[2010, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2010, 12, 1]
    time2=[2011, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2011, 12, 1]
    time2=[2012, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,4
    time1=[2012, 12, 1]
    time2=[2012, 12, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='solid')

    plt.legend()
    plt.xlabel('latitude')
    plt.title('seasonal CAO zonal mean ditribution')
    plt.savefig('../figures/final/M_lats.png')


    ############SUMMER
    ########################### 0,1
    time1=[2010, 6, 1]
    time2=[2010, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2011, 6, 1]
    time2=[2011, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2012, 6, 1]
    time2=[2012, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='dashed')
    plt.scatter(lats,f,color=use_colors[test_num],label=modname[test_num],marker='x')

    plt.savefig('../figures/final/M_lats.png')

else:
    print(test_num)


# @Author: Geethma Werapitiya <wgeethma>
# @Date:   2022-07-17T14:34:35-06:00
# @Email:  wgeethma@uwyo.edu
# @Last modified by:   wgeethma
# @Last modified time: 2022-07-18T13:47:07-06:00

# import xarray as xr
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.cm import get_cmap
# import calendar
# from global_land_mask import globe
# import glob
# import math
# from scipy import stats
# import os
# import netCDF4 as nc
#
# from highres_read import read_var_hires
# from myReadGCMsDaily import read_var_mod
# from regrid_wght_3d import regrid_wght_wnans
#
# import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 100
# plt.clf()
# plt.rcParams['figure.figsize'] = (10, 10)
#
# from con_models import get_cons
# con, use_colors, varname, pvarname, modname, warming_modname, hiresmd, amip_md = get_cons()
#
# #latitude range
# latr1 = 30
# latr2 = 80

test_num = test_num+1

if test_num<=23:
    ########################### 0,1
    time1=[2010, 1, 1]
    time2=[2010, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2010, 12, 1]
    time2=[2011, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2011, 12, 1]
    time2=[2012, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,4
    time1=[2012, 12, 1]
    time2=[2012, 12, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='solid')

    plt.legend()
    plt.savefig('../figures/final/M_lats.png')


    ############SUMMER
    ########################### 0,1
    time1=[2010, 6, 1]
    time2=[2010, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2011, 6, 1]
    time2=[2011, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2012, 6, 1]
    time2=[2012, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='dashed')
    plt.scatter(lats,f,color=use_colors[test_num],label=modname[test_num],marker='x')

    plt.savefig('../figures/final/M_lats.png')

else:
    print(test_num)

# @Author: Geethma Werapitiya <wgeethma>
# @Date:   2022-07-17T14:34:35-06:00
# @Email:  wgeethma@uwyo.edu
# @Last modified by:   wgeethma
# @Last modified time: 2022-07-18T13:47:07-06:00

# import xarray as xr
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.cm import get_cmap
# import calendar
# from global_land_mask import globe
# import glob
# import math
# from scipy import stats
# import os
# import netCDF4 as nc
#
# from highres_read import read_var_hires
# from myReadGCMsDaily import read_var_mod
# from regrid_wght_3d import regrid_wght_wnans
#
# import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 100
# plt.clf()
# plt.rcParams['figure.figsize'] = (10, 10)
#
# from con_models import get_cons
# con, use_colors, varname, pvarname, modname, warming_modname, hiresmd, amip_md = get_cons()
#
# #latitude range
# latr1 = 30
# latr2 = 80

test_num = test_num+1

if test_num<=23:
    ########################### 0,1
    time1=[2010, 1, 1]
    time2=[2010, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2010, 12, 1]
    time2=[2011, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2011, 12, 1]
    time2=[2012, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,4
    time1=[2012, 12, 1]
    time2=[2012, 12, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='solid')

    plt.legend()
    plt.xlabel('latitude')
    plt.title('seasonal CAO zonal mean ditribution')
    plt.savefig('../figures/final/M_lats.png')


    ############SUMMER
    ########################### 0,1
    time1=[2010, 6, 1]
    time2=[2010, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2011, 6, 1]
    time2=[2011, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2012, 6, 1]
    time2=[2012, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='dashed')
    plt.scatter(lats,f,color=use_colors[test_num],label=modname[test_num],marker='x')

    plt.savefig('../figures/final/M_lats.png')

else:
    print(test_num)


# @Author: Geethma Werapitiya <wgeethma>
# @Date:   2022-07-17T14:34:35-06:00
# @Email:  wgeethma@uwyo.edu
# @Last modified by:   wgeethma
# @Last modified time: 2022-07-18T13:47:07-06:00

# import xarray as xr
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.cm import get_cmap
# import calendar
# from global_land_mask import globe
# import glob
# import math
# from scipy import stats
# import os
# import netCDF4 as nc
#
# from highres_read import read_var_hires
# from myReadGCMsDaily import read_var_mod
# from regrid_wght_3d import regrid_wght_wnans
#
# import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 100
# plt.clf()
# plt.rcParams['figure.figsize'] = (10, 10)
#
# from con_models import get_cons
# con, use_colors, varname, pvarname, modname, warming_modname, hiresmd, amip_md = get_cons()
#
# #latitude range
# latr1 = 30
# latr2 = 80

test_num = test_num+1

if test_num<=23:
    ########################### 0,1
    time1=[2010, 1, 1]
    time2=[2010, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2010, 12, 1]
    time2=[2011, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2011, 12, 1]
    time2=[2012, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,4
    time1=[2012, 12, 1]
    time2=[2012, 12, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='solid')

    plt.legend()
    plt.savefig('../figures/final/M_lats.png')


    ############SUMMER
    ########################### 0,1
    time1=[2010, 6, 1]
    time2=[2010, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2011, 6, 1]
    time2=[2011, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2012, 6, 1]
    time2=[2012, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='dashed')
    plt.scatter(lats,f,color=use_colors[test_num],label=modname[test_num],marker='x')

    plt.savefig('../figures/final/M_lats.png')

else:
    print(test_num)

# @Author: Geethma Werapitiya <wgeethma>
# @Date:   2022-07-17T14:34:35-06:00
# @Email:  wgeethma@uwyo.edu
# @Last modified by:   wgeethma
# @Last modified time: 2022-07-18T13:47:07-06:00

# import xarray as xr
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.cm import get_cmap
# import calendar
# from global_land_mask import globe
# import glob
# import math
# from scipy import stats
# import os
# import netCDF4 as nc
#
# from highres_read import read_var_hires
# from myReadGCMsDaily import read_var_mod
# from regrid_wght_3d import regrid_wght_wnans
#
# import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 100
# plt.clf()
# plt.rcParams['figure.figsize'] = (10, 10)
#
# from con_models import get_cons
# con, use_colors, varname, pvarname, modname, warming_modname, hiresmd, amip_md = get_cons()
#
# #latitude range
# latr1 = 30
# latr2 = 80

test_num = test_num+1

if test_num<=23:
    ########################### 0,1
    time1=[2010, 1, 1]
    time2=[2010, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2010, 12, 1]
    time2=[2011, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2011, 12, 1]
    time2=[2012, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,4
    time1=[2012, 12, 1]
    time2=[2012, 12, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='solid')

    plt.legend()
    plt.xlabel('latitude')
    plt.title('seasonal CAO zonal mean ditribution')
    plt.savefig('../figures/final/M_lats.png')


    ############SUMMER
    ########################### 0,1
    time1=[2010, 6, 1]
    time2=[2010, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2011, 6, 1]
    time2=[2011, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2012, 6, 1]
    time2=[2012, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='dashed')
    plt.scatter(lats,f,color=use_colors[test_num],label=modname[test_num],marker='x')

    plt.savefig('../figures/final/M_lats.png')

else:
    print(test_num)


# @Author: Geethma Werapitiya <wgeethma>
# @Date:   2022-07-17T14:34:35-06:00
# @Email:  wgeethma@uwyo.edu
# @Last modified by:   wgeethma
# @Last modified time: 2022-07-18T13:47:07-06:00

# import xarray as xr
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.cm import get_cmap
# import calendar
# from global_land_mask import globe
# import glob
# import math
# from scipy import stats
# import os
# import netCDF4 as nc
#
# from highres_read import read_var_hires
# from myReadGCMsDaily import read_var_mod
# from regrid_wght_3d import regrid_wght_wnans
#
# import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 100
# plt.clf()
# plt.rcParams['figure.figsize'] = (10, 10)
#
# from con_models import get_cons
# con, use_colors, varname, pvarname, modname, warming_modname, hiresmd, amip_md = get_cons()
#
# #latitude range
# latr1 = 30
# latr2 = 80

test_num = test_num+1

if test_num<=23:
    ########################### 0,1
    time1=[2010, 1, 1]
    time2=[2010, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2010, 12, 1]
    time2=[2011, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2011, 12, 1]
    time2=[2012, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,4
    time1=[2012, 12, 1]
    time2=[2012, 12, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='solid')

    plt.legend()
    plt.savefig('../figures/final/M_lats.png')


    ############SUMMER
    ########################### 0,1
    time1=[2010, 6, 1]
    time2=[2010, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2011, 6, 1]
    time2=[2011, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2012, 6, 1]
    time2=[2012, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='dashed')
    plt.scatter(lats,f,color=use_colors[test_num],label=modname[test_num],marker='x')

    plt.savefig('../figures/final/M_lats.png')

else:
    print(test_num)

# @Author: Geethma Werapitiya <wgeethma>
# @Date:   2022-07-17T14:34:35-06:00
# @Email:  wgeethma@uwyo.edu
# @Last modified by:   wgeethma
# @Last modified time: 2022-07-18T13:47:07-06:00

# import xarray as xr
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.cm import get_cmap
# import calendar
# from global_land_mask import globe
# import glob
# import math
# from scipy import stats
# import os
# import netCDF4 as nc
#
# from highres_read import read_var_hires
# from myReadGCMsDaily import read_var_mod
# from regrid_wght_3d import regrid_wght_wnans
#
# import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 100
# plt.clf()
# plt.rcParams['figure.figsize'] = (10, 10)
#
# from con_models import get_cons
# con, use_colors, varname, pvarname, modname, warming_modname, hiresmd, amip_md = get_cons()
#
# #latitude range
# latr1 = 30
# latr2 = 80

test_num = test_num+1

if test_num<=23:
    ########################### 0,1
    time1=[2010, 1, 1]
    time2=[2010, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2010, 12, 1]
    time2=[2011, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2011, 12, 1]
    time2=[2012, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,4
    time1=[2012, 12, 1]
    time2=[2012, 12, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='solid')

    plt.legend()
    plt.xlabel('latitude')
    plt.title('seasonal CAO zonal mean ditribution')
    plt.savefig('../figures/final/M_lats.png')


    ############SUMMER
    ########################### 0,1
    time1=[2010, 6, 1]
    time2=[2010, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2011, 6, 1]
    time2=[2011, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2012, 6, 1]
    time2=[2012, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='dashed')
    plt.scatter(lats,f,color=use_colors[test_num],label=modname[test_num],marker='x')

    plt.savefig('../figures/final/M_lats.png')

else:
    print(test_num)


# @Author: Geethma Werapitiya <wgeethma>
# @Date:   2022-07-17T14:34:35-06:00
# @Email:  wgeethma@uwyo.edu
# @Last modified by:   wgeethma
# @Last modified time: 2022-07-18T13:47:07-06:00

# import xarray as xr
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.cm import get_cmap
# import calendar
# from global_land_mask import globe
# import glob
# import math
# from scipy import stats
# import os
# import netCDF4 as nc
#
# from highres_read import read_var_hires
# from myReadGCMsDaily import read_var_mod
# from regrid_wght_3d import regrid_wght_wnans
#
# import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 100
# plt.clf()
# plt.rcParams['figure.figsize'] = (10, 10)
#
# from con_models import get_cons
# con, use_colors, varname, pvarname, modname, warming_modname, hiresmd, amip_md = get_cons()
#
# #latitude range
# latr1 = 30
# latr2 = 80

test_num = test_num+1

if test_num<=23:
    ########################### 0,1
    time1=[2010, 1, 1]
    time2=[2010, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2010, 12, 1]
    time2=[2011, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2011, 12, 1]
    time2=[2012, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,4
    time1=[2012, 12, 1]
    time2=[2012, 12, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='solid')

    plt.legend()
    plt.savefig('../figures/final/M_lats.png')


    ############SUMMER
    ########################### 0,1
    time1=[2010, 6, 1]
    time2=[2010, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2011, 6, 1]
    time2=[2011, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2012, 6, 1]
    time2=[2012, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='dashed')
    plt.scatter(lats,f,color=use_colors[test_num],label=modname[test_num],marker='x')

    plt.savefig('../figures/final/M_lats.png')

else:
    print(test_num)

# @Author: Geethma Werapitiya <wgeethma>
# @Date:   2022-07-17T14:34:35-06:00
# @Email:  wgeethma@uwyo.edu
# @Last modified by:   wgeethma
# @Last modified time: 2022-07-18T13:47:07-06:00

# import xarray as xr
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.cm import get_cmap
# import calendar
# from global_land_mask import globe
# import glob
# import math
# from scipy import stats
# import os
# import netCDF4 as nc
#
# from highres_read import read_var_hires
# from myReadGCMsDaily import read_var_mod
# from regrid_wght_3d import regrid_wght_wnans
#
# import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 100
# plt.clf()
# plt.rcParams['figure.figsize'] = (10, 10)
#
# from con_models import get_cons
# con, use_colors, varname, pvarname, modname, warming_modname, hiresmd, amip_md = get_cons()
#
# #latitude range
# latr1 = 30
# latr2 = 80

test_num = test_num+1

if test_num<=23:
    ########################### 0,1
    time1=[2010, 1, 1]
    time2=[2010, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2010, 12, 1]
    time2=[2011, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2011, 12, 1]
    time2=[2012, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,4
    time1=[2012, 12, 1]
    time2=[2012, 12, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='solid')

    plt.legend()
    plt.xlabel('latitude')
    plt.title('seasonal CAO zonal mean ditribution')
    plt.savefig('../figures/final/M_lats.png')


    ############SUMMER
    ########################### 0,1
    time1=[2010, 6, 1]
    time2=[2010, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2011, 6, 1]
    time2=[2011, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2012, 6, 1]
    time2=[2012, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='dashed')
    plt.scatter(lats,f,color=use_colors[test_num],label=modname[test_num],marker='x')

    plt.savefig('../figures/final/M_lats.png')

else:
    print(test_num)


# @Author: Geethma Werapitiya <wgeethma>
# @Date:   2022-07-17T14:34:35-06:00
# @Email:  wgeethma@uwyo.edu
# @Last modified by:   wgeethma
# @Last modified time: 2022-07-18T13:47:07-06:00

# import xarray as xr
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.cm import get_cmap
# import calendar
# from global_land_mask import globe
# import glob
# import math
# from scipy import stats
# import os
# import netCDF4 as nc
#
# from highres_read import read_var_hires
# from myReadGCMsDaily import read_var_mod
# from regrid_wght_3d import regrid_wght_wnans
#
# import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 100
# plt.clf()
# plt.rcParams['figure.figsize'] = (10, 10)
#
# from con_models import get_cons
# con, use_colors, varname, pvarname, modname, warming_modname, hiresmd, amip_md = get_cons()
#
# #latitude range
# latr1 = 30
# latr2 = 80

test_num = test_num+1

if test_num<=23:
    ########################### 0,1
    time1=[2010, 1, 1]
    time2=[2010, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2010, 12, 1]
    time2=[2011, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2011, 12, 1]
    time2=[2012, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,4
    time1=[2012, 12, 1]
    time2=[2012, 12, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='solid')

    plt.legend()
    plt.savefig('../figures/final/M_lats.png')


    ############SUMMER
    ########################### 0,1
    time1=[2010, 6, 1]
    time2=[2010, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2011, 6, 1]
    time2=[2011, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2012, 6, 1]
    time2=[2012, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='dashed')
    plt.scatter(lats,f,color=use_colors[test_num],label=modname[test_num],marker='x')

    plt.savefig('../figures/final/M_lats.png')

else:
    print(test_num)


# @Author: Geethma Werapitiya <wgeethma>
# @Date:   2022-07-17T14:34:35-06:00
# @Email:  wgeethma@uwyo.edu
# @Last modified by:   wgeethma
# @Last modified time: 2022-07-18T13:47:07-06:00

# import xarray as xr
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.cm import get_cmap
# import calendar
# from global_land_mask import globe
# import glob
# import math
# from scipy import stats
# import os
# import netCDF4 as nc
#
# from highres_read import read_var_hires
# from myReadGCMsDaily import read_var_mod
# from regrid_wght_3d import regrid_wght_wnans
#
# import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 100
# plt.clf()
# plt.rcParams['figure.figsize'] = (10, 10)
#
# from con_models import get_cons
# con, use_colors, varname, pvarname, modname, warming_modname, hiresmd, amip_md = get_cons()
#
# #latitude range
# latr1 = 30
# latr2 = 80

test_num = test_num+1

if test_num<=23:
    ########################### 0,1
    time1=[2010, 1, 1]
    time2=[2010, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2010, 12, 1]
    time2=[2011, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2011, 12, 1]
    time2=[2012, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,4
    time1=[2012, 12, 1]
    time2=[2012, 12, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='solid')

    plt.legend()
    plt.xlabel('latitude')
    plt.title('seasonal CAO zonal mean ditribution')
    plt.savefig('../figures/final/M_lats.png')


    ############SUMMER
    ########################### 0,1
    time1=[2010, 6, 1]
    time2=[2010, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2011, 6, 1]
    time2=[2011, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2012, 6, 1]
    time2=[2012, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='dashed')
    plt.scatter(lats,f,color=use_colors[test_num],label=modname[test_num],marker='x')

    plt.savefig('../figures/final/M_lats.png')

else:
    print(test_num)


# @Author: Geethma Werapitiya <wgeethma>
# @Date:   2022-07-17T14:34:35-06:00
# @Email:  wgeethma@uwyo.edu
# @Last modified by:   wgeethma
# @Last modified time: 2022-07-18T13:47:07-06:00

# import xarray as xr
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.cm import get_cmap
# import calendar
# from global_land_mask import globe
# import glob
# import math
# from scipy import stats
# import os
# import netCDF4 as nc
#
# from highres_read import read_var_hires
# from myReadGCMsDaily import read_var_mod
# from regrid_wght_3d import regrid_wght_wnans
#
# import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 100
# plt.clf()
# plt.rcParams['figure.figsize'] = (10, 10)
#
# from con_models import get_cons
# con, use_colors, varname, pvarname, modname, warming_modname, hiresmd, amip_md = get_cons()
#
# #latitude range
# latr1 = 30
# latr2 = 80

test_num = test_num+1

if test_num<=23:
    ########################### 0,1
    time1=[2010, 1, 1]
    time2=[2010, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2010, 12, 1]
    time2=[2011, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2011, 12, 1]
    time2=[2012, 2, 28]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,4
    time1=[2012, 12, 1]
    time2=[2012, 12, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='solid')

    plt.legend()
    plt.savefig('../figures/final/M_lats.png')


    ############SUMMER
    ########################### 0,1
    time1=[2010, 6, 1]
    time2=[2010, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    M_plot = []

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ############################################### 0,2
    time1=[2011, 6, 1]
    time2=[2011, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    ########################### 0,3
    time1=[2012, 6, 1]
    time2=[2012, 8, 30]

    for j in range(test_num,test_num+1):
        print(modname[j],' ', str(j))
        for i in varname:
            locals()[i+'__'+str(j+1)] = read_var_mod('surface', modname[j], 'historical', i, time1, time2)

        for k in pvarname:
            locals()[k+'__'+str(j+1)] = read_var_mod('p_level', modname[j], 'historical', k, time1, time2)

    print('done')

    for i in range(test_num,test_num+1): #l,mm+len(hiresmd)
        if i<len(modname):
            print(modname[i],str(i))
        else:
            print(hiresmd[i-mm],str(i))

        for j in varname:
            lat  = locals()[j+'__'+str(i+1)][0]
            lon  = locals()[j+'__'+str(i+1)][1]
            time = locals()[j+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            #print(j)
            locals()[j+str(i+1)] = locals()[j+'__'+str(i+1)][4]
            locals()[j+str(i+1)] = np.ma.filled(locals()[j+str(i+1)], fill_value=np.nan)
            locals()['plot_'+j+str(i+1)] = np.array(np.multiply(maskm,locals()[j+str(i+1)][:,lat_ind1[0]:lat_ind2[0],:]))
            locals()['grid_'+j+str(i+1)] = locals()['plot_'+j+str(i+1)]
            #regrid_wght_wnans(lats,lon,locals()['plot_'+j+str(i+1)],lats_edges,lons_edges)[0]

        for k in pvarname:
            #print(k)
            lat  = locals()[k+'__'+str(i+1)][0]
            lon  = locals()[k+'__'+str(i+1)][1]
            time = locals()[k+'__'+str(i+1)][2]

            x_lat = np.array(lat)
            lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
            lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
            lats = lat[lat_ind1[0]:lat_ind2[0]]

            x_lon = lon
            lon = np.array(lon)
            lon[lon > 180] = lon[lon > 180]-360

            maskm = np.ones((len(time),len(lats),len(lon)))

            for a in range(len(lats)):
                for b in range(len(lon)):
                    if globe.is_land(lats[a], lon[b])==True:
                        maskm[:,a,b] = math.nan
            locals()['plot_levels'+str(i+1)] = locals()['ta__'+str(i+1)][3]
            locals()['grid_'+k+str(i+1)] = []

            levels = locals()['plot_levels'+str(i+1)]

            for p in range(len(levels)):
                if levels[p] == 70000:
                    #print(levels[p])
                    locals()[k+str(i+1)] = locals()[k+'__'+str(i+1)][4]
                    locals()[k+str(i+1)] = np.ma.filled(locals()[k+str(i+1)], fill_value=np.nan)
                    temp_700   = np.array(np.multiply(maskm,locals()[k+str(i+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                    # regrid     = regrid_wght_wnans(lats,lon,temp_700,lats_edges,lons_edges)
                    # grid_t_700 = regrid[0]
                    # lat_n      = regrid[2][:,0]
                    break;

        theta_700 = temp_700*(100000/70000)**con
        # theta_700 = grid_t_700*(100000/70000)**con
        theta_t2m = locals()['grid_tas'+str(i+1)]*(100000/locals()['grid_psl'+str(i+1)])**con

        t = min(len(theta_t2m),len(theta_700))
        M_700  = theta_t2m[0:t,:,:] - theta_700[0:t,:,:]
        M_plot.extend(M_700)

    num_UM = []
    num_M  = []

    M_plot = np.array(M_plot)

    for j in range(np.shape(M_plot)[1]):
        count_UM = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]>=-9:
                    count_UM = count_UM+1
        num_UM.append(count_UM)

    for j in range(np.shape(M_plot)[1]):
        count_M = 0

        for i in range(np.shape(M_plot)[0]):
            for k in range(np.shape(M_plot)[2]):
                if M_plot[i,j,k]!=np.nan:
                    count_M = count_M+1
        num_M.append(count_M)

    num_M = np.array(num_M)
    num_UM = np.array(num_UM)

    f = np.divide(num_UM,num_M)

    plt.plot(lats,f,color=use_colors[test_num],label=modname[test_num],linestyle='dashed')
    plt.scatter(lats,f,color=use_colors[test_num],label=modname[test_num],marker='x')

    plt.savefig('../figures/final/M_lats.png')

else:
    print(test_num)
