# @Author: geethmawerapitiya
# @Date:   2022-07-15T00:29:10-06:00
# @Project: Research
# @Filename: macwind_function.py
# @Last modified by:   wgeethma
# @Last modified time: 2022-07-15T16:51:18-06:00

import glob
import numpy as np
import math
import os
import netCDF4 as nc

def obs(folder, sT, pressure, latr1, latr2):
    print('OBSERVATIONS')

    merlist = np.sort(glob.glob('../data_merra/all_lat_lon/level/MERRA2_*.nc'))
    sfclist = np.sort(glob.glob('../data_merra/all_lat_lon/'+folder+'/MERRA2_*.nc'))
    maclist = np.sort(glob.glob('../MACLWP_dailymean/take/wind1deg*.nc4'))

    new_list_s = []
    new_list_m = []
    new_list_c = []

    s = 0
    m = 0
    print((len(merlist), len(sfclist)))
    length = max(len(merlist), len(sfclist))

    #if 
    while (s < len(sfclist) and m < len(merlist)):
        #print(m,s)
        name_s = os.path.basename(sfclist[s])
        date_s = name_s.split(".")[2]

        name_m = os.path.basename(merlist[m])
        date_m = name_m.split(".")[2]

        if date_s==date_m:
            #print(date_s, date_m)
            new_list_s.append(sfclist[s])
            new_list_m.append(merlist[m])

            s = s+1
            m = m+1

        elif date_s<date_m:
            s = s+1

        elif date_s>date_m:
            m = m+1

    macwind = []
    new_s   = []
    new_m   = []
    macdate = []
    for i in range(len(new_list_s)):
        flag = 0
        name_s = os.path.basename(new_list_s[i])
        date_s = name_s.split(".")[2]

        for k in range(len(maclist)):

            name_mac = os.path.basename(maclist[k])
            date_mac = name_mac.split(".")[1]
            ddata    = nc.Dataset(maclist[k])
            mactime  = ddata.variables['time'][:]
            macw     = ddata.variables['sfcwind'][:]

            if date_mac==date_s[0:-2]:
                for r in range(len(mactime)):
                    if str(mactime[r]). zfill(2) == date_s[-2::]:
                        macwind.append(macw[r,:,:])
                        new_m.append(new_list_m[i])
                        new_s.append(new_list_s[i])
                        macdate.append(date_mac+str(mactime[r]). zfill(2))

                        flag = 1
                        break;

            if flag == 1:
                break;


    macwind = np.array(macwind)
    print('macwind done')
    ################################

    p_mer_T   = []
    p_mac_w   = []
    sfc_mer_T = []
    sfc_mer_P = []
    sfc_mer_U = []
    sfc_mer_V = []

    for i in range(len(new_s)): #len(merlist)
        d_path = new_m[i]
        data   = nc.Dataset(d_path)
        # print(d_path)

        if i==0:
            merlat = data.variables['lat'][:]
            merlon = data.variables['lon'][:]
            merlev = data.variables['lev'][:]
            for i in range(len(merlev)):
                if merlev[i] == pressure:
                    p_lev_obs = i
                    break;
            #shape latitude
            mer_lat = np.flip(merlat)
            mer_lat = np.array(mer_lat)
            mlat_ind1 = np.where(mer_lat == mer_lat.flat[np.abs(mer_lat - (latr1)).argmin()])[0]
            mlat_ind2 = np.where(mer_lat == mer_lat.flat[np.abs(mer_lat - (latr2)).argmin()])[0]
            p_mer_lat  = np.array(mer_lat[mlat_ind1[0]:mlat_ind2[0]])
            #shape longitude
            merlon[merlon > 180] = merlon[merlon > 180]-360
            # mer_lon = np.array(merlon)

        merT      = data.variables['T'][:] #(time, lev, lat, lon)
        mer_T     = np.array(merT[:,:,::-1,:])
        mer_T_700 = mer_T[:,p_lev_obs,mlat_ind1[0]:mlat_ind2[0],:]
        p_mer_T.extend(mer_T_700)

        s_path = new_s[i]
        sdata  = nc.Dataset(s_path)

        sfcT   = sdata.variables[sT][:]
        sfc_T = np.array(sfcT[:,::-1,:])
        sfc_mer_T.extend(sfc_T[:,mlat_ind1[0]:mlat_ind2[0],:])


        sfcP   = sdata.variables['SLP'][:]
        sfc_P  = np.array(sfcP[:,::-1,:])
        sfc_mer_P.extend(sfc_P[:,mlat_ind1[0]:mlat_ind2[0],:])

        sfcU   = sdata.variables['U10M'][:]
        sfc_U = np.array(sfcU[:,::-1,:])
        sfc_mer_U.extend(sfc_U[:,mlat_ind1[0]:mlat_ind2[0],:])

        sfcV   = sdata.variables['V10M'][:]
        sfc_V = np.array(sfcV[:,::-1,:])
        sfc_mer_V.extend(sfc_V[:,mlat_ind1[0]:mlat_ind2[0],:])

    p_mac_w = macwind[:,mlat_ind1[0]:mlat_ind2[0],:]

    macwind    = np.array(p_mac_w)
    temp    = np.array(p_mer_T)
    sfctemp = np.array(sfc_mer_T)
    sfcpres = np.array(sfc_mer_P)
    merwind = np.array(np.sqrt(np.array(sfc_mer_U)**2 + np.array(sfc_mer_V)**2))

    return(merwind, macwind, temp, sfctemp, sfcpres, p_lev_obs, p_mer_lat, merlon, merlev)
