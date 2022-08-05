# @Author: Geethma Werapitiya <wgeethma>
# @Date:   2022-07-18T02:37:52-06:00
# @Email:  wgeethma@uwyo.edu
# @Last modified by:   wgeethma
# @Last modified time: 2022-07-20T04:11:17-06:00

DJF = []
DJF_lats = []



for mod in range(0,2):
    M_plot = []

    time1=[2010, 1, 1]
    time2=[2010, 2, 28]

    for i in varname:
        locals()[i+'__'+str(mod+1)] = read_var_mod('surface', warming_modname[mod], 'historical', i, time1, time2)

    for k in pvarname:
        locals()[k+'__'+str(mod+1)] = read_var_mod('p_level', warming_modname[mod], 'historical', k, time1, time2)


    for j in varname:
        lat  = locals()[j+'__'+str(mod+1)][0]
        lon  = locals()[j+'__'+str(mod+1)][1]
        time = locals()[j+'__'+str(mod+1)][2]

        x_lat = np.array(lat)
        lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
        lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
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
        locals()[j+str(mod+1)] = locals()[j+'__'+str(mod+1)][4]
        locals()[j+str(mod+1)] = np.ma.filled(locals()[j+str(mod+1)], fill_value=np.nan)
        locals()['plot_'+j+str(mod+1)] = np.array(np.multiply(maskm,locals()[j+str(mod+1)][:,lat_ind1[0]:lat_ind2[0],:]))
        locals()['grid_'+j+str(mod+1)] = locals()['plot_'+j+str(mod+1)]

    for k in pvarname:
        #print(k)
        lat  = locals()[k+'__'+str(mod+1)][0]
        lon  = locals()[k+'__'+str(mod+1)][1]
        time = locals()[k+'__'+str(mod+1)][2]

        x_lat = np.array(lat)
        lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
        lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
        lats = lat[lat_ind1[0]:lat_ind2[0]]

        x_lon = lon
        lon = np.array(lon)
        lon[lon > 180] = lon[lon > 180]-360

        maskm = np.ones((len(time),len(lats),len(lon)))

        for a in range(len(lats)):
            for b in range(len(lon)):
                if globe.is_land(lats[a], lon[b])==True:
                    maskm[:,a,b] = math.nan
        locals()['plot_levels'+str(mod+1)] = locals()['ta__'+str(mod+1)][3]
        locals()['grid_'+k+str(mod+1)] = []

        levels = locals()['plot_levels'+str(mod+1)]

        for p in range(len(levels)):
            if levels[p] == 70000:
                locals()[k+str(mod+1)] = locals()[k+'__'+str(mod+1)][4]
                locals()[k+str(mod+1)] = np.ma.filled(locals()[k+str(mod+1)], fill_value=np.nan)
                temp_700   = np.array(np.multiply(maskm,locals()[k+str(mod+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                break;

    theta_700 = temp_700*(100000/70000)**con
    theta_t2m = locals()['grid_tas'+str(mod+1)]*(100000/locals()['grid_psl'+str(mod+1)])**con

    # t = min(len(theta_t2m),len(theta_700))
    M_700  = theta_t2m - theta_700
    M_plot.extend(M_700)

############################################### 0,2
    time1=[2010, 12, 1]
    time2=[2011, 2, 28]

    for i in varname:
        locals()[i+'__'+str(mod+1)] = read_var_mod('surface', warming_modname[mod], 'historical', i, time1, time2)

    for k in pvarname:
        locals()[k+'__'+str(mod+1)] = read_var_mod('p_level', warming_modname[mod], 'historical', k, time1, time2)


    for j in varname:
        lat  = locals()[j+'__'+str(mod+1)][0]
        lon  = locals()[j+'__'+str(mod+1)][1]
        time = locals()[j+'__'+str(mod+1)][2]

        x_lat = np.array(lat)
        lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
        lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
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
        locals()[j+str(mod+1)] = locals()[j+'__'+str(mod+1)][4]
        locals()[j+str(mod+1)] = np.ma.filled(locals()[j+str(mod+1)], fill_value=np.nan)
        locals()['plot_'+j+str(mod+1)] = np.array(np.multiply(maskm,locals()[j+str(mod+1)][:,lat_ind1[0]:lat_ind2[0],:]))
        locals()['grid_'+j+str(mod+1)] = locals()['plot_'+j+str(mod+1)]

    for k in pvarname:
        #print(k)
        lat  = locals()[k+'__'+str(mod+1)][0]
        lon  = locals()[k+'__'+str(mod+1)][1]
        time = locals()[k+'__'+str(mod+1)][2]

        x_lat = np.array(lat)
        lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
        lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
        lats = lat[lat_ind1[0]:lat_ind2[0]]

        x_lon = lon
        lon = np.array(lon)
        lon[lon > 180] = lon[lon > 180]-360

        maskm = np.ones((len(time),len(lats),len(lon)))

        for a in range(len(lats)):
            for b in range(len(lon)):
                if globe.is_land(lats[a], lon[b])==True:
                    maskm[:,a,b] = math.nan
        locals()['plot_levels'+str(mod+1)] = locals()['ta__'+str(mod+1)][3]
        locals()['grid_'+k+str(mod+1)] = []

        levels = locals()['plot_levels'+str(mod+1)]

        for p in range(len(levels)):
            if levels[p] == 70000:
                locals()[k+str(mod+1)] = locals()[k+'__'+str(mod+1)][4]
                locals()[k+str(mod+1)] = np.ma.filled(locals()[k+str(mod+1)], fill_value=np.nan)
                temp_700   = np.array(np.multiply(maskm,locals()[k+str(mod+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                break;

    theta_700 = temp_700*(100000/70000)**con
    theta_t2m = locals()['grid_tas'+str(mod+1)]*(100000/locals()['grid_psl'+str(mod+1)])**con

    # t = min(len(theta_t2m),len(theta_700))
    M_700  = theta_t2m - theta_700
    M_plot.extend(M_700)

########################### 0,3
    time1=[2011, 12, 1]
    time2=[2012, 2, 28]

    for i in varname:
        locals()[i+'__'+str(mod+1)] = read_var_mod('surface', warming_modname[mod], 'historical', i, time1, time2)

    for k in pvarname:
        locals()[k+'__'+str(mod+1)] = read_var_mod('p_level', warming_modname[mod], 'historical', k, time1, time2)


    for j in varname:
        lat  = locals()[j+'__'+str(mod+1)][0]
        lon  = locals()[j+'__'+str(mod+1)][1]
        time = locals()[j+'__'+str(mod+1)][2]

        x_lat = np.array(lat)
        lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
        lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
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
        locals()[j+str(mod+1)] = locals()[j+'__'+str(mod+1)][4]
        locals()[j+str(mod+1)] = np.ma.filled(locals()[j+str(mod+1)], fill_value=np.nan)
        locals()['plot_'+j+str(mod+1)] = np.array(np.multiply(maskm,locals()[j+str(mod+1)][:,lat_ind1[0]:lat_ind2[0],:]))
        locals()['grid_'+j+str(mod+1)] = locals()['plot_'+j+str(mod+1)]

    for k in pvarname:
        #print(k)
        lat  = locals()[k+'__'+str(mod+1)][0]
        lon  = locals()[k+'__'+str(mod+1)][1]
        time = locals()[k+'__'+str(mod+1)][2]

        x_lat = np.array(lat)
        lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
        lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
        lats = lat[lat_ind1[0]:lat_ind2[0]]

        x_lon = lon
        lon = np.array(lon)
        lon[lon > 180] = lon[lon > 180]-360

        maskm = np.ones((len(time),len(lats),len(lon)))

        for a in range(len(lats)):
            for b in range(len(lon)):
                if globe.is_land(lats[a], lon[b])==True:
                    maskm[:,a,b] = math.nan
        locals()['plot_levels'+str(mod+1)] = locals()['ta__'+str(mod+1)][3]
        locals()['grid_'+k+str(mod+1)] = []

        levels = locals()['plot_levels'+str(mod+1)]

        for p in range(len(levels)):
            if levels[p] == 70000:
                locals()[k+str(mod+1)] = locals()[k+'__'+str(mod+1)][4]
                locals()[k+str(mod+1)] = np.ma.filled(locals()[k+str(mod+1)], fill_value=np.nan)
                temp_700   = np.array(np.multiply(maskm,locals()[k+str(mod+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                break;

    theta_700 = temp_700*(100000/70000)**con
    theta_t2m = locals()['grid_tas'+str(mod+1)]*(100000/locals()['grid_psl'+str(mod+1)])**con

    # t = min(len(theta_t2m),len(theta_700))
    M_700  = theta_t2m - theta_700
    M_plot.extend(M_700)

########################### 0,4
    time1=[2012, 12, 1]
    time2=[2012, 12, 30]

    for i in varname:
        locals()[i+'__'+str(mod+1)] = read_var_mod('surface', warming_modname[mod], 'historical', i, time1, time2)

    for k in pvarname:
        locals()[k+'__'+str(mod+1)] = read_var_mod('p_level', warming_modname[mod], 'historical', k, time1, time2)


    for j in varname:
        lat  = locals()[j+'__'+str(mod+1)][0]
        lon  = locals()[j+'__'+str(mod+1)][1]
        time = locals()[j+'__'+str(mod+1)][2]

        x_lat = np.array(lat)
        lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
        lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
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
        locals()[j+str(mod+1)] = locals()[j+'__'+str(mod+1)][4]
        locals()[j+str(mod+1)] = np.ma.filled(locals()[j+str(mod+1)], fill_value=np.nan)
        locals()['plot_'+j+str(mod+1)] = np.array(np.multiply(maskm,locals()[j+str(mod+1)][:,lat_ind1[0]:lat_ind2[0],:]))
        locals()['grid_'+j+str(mod+1)] = locals()['plot_'+j+str(mod+1)]

    for k in pvarname:
        #print(k)
        lat  = locals()[k+'__'+str(mod+1)][0]
        lon  = locals()[k+'__'+str(mod+1)][1]
        time = locals()[k+'__'+str(mod+1)][2]

        x_lat = np.array(lat)
        lat_ind1 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr1)).argmin()])[0]
        lat_ind2 = np.where(x_lat == x_lat.flat[np.abs(x_lat - (latr2)).argmin()])[0]
        lats = lat[lat_ind1[0]:lat_ind2[0]]

        x_lon = lon
        lon = np.array(lon)
        lon[lon > 180] = lon[lon > 180]-360

        maskm = np.ones((len(time),len(lats),len(lon)))

        for a in range(len(lats)):
            for b in range(len(lon)):
                if globe.is_land(lats[a], lon[b])==True:
                    maskm[:,a,b] = math.nan
        locals()['plot_levels'+str(mod+1)] = locals()['ta__'+str(mod+1)][3]
        locals()['grid_'+k+str(mod+1)] = []

        levels = locals()['plot_levels'+str(mod+1)]

        for p in range(len(levels)):
            if levels[p] == 70000:
                locals()[k+str(mod+1)] = locals()[k+'__'+str(mod+1)][4]
                locals()[k+str(mod+1)] = np.ma.filled(locals()[k+str(mod+1)], fill_value=np.nan)
                temp_700   = np.array(np.multiply(maskm,locals()[k+str(mod+1)][:,p,lat_ind1[0]:lat_ind2[0],:]))
                break;

    theta_700 = temp_700*(100000/70000)**con
    theta_t2m = locals()['grid_tas'+str(mod+1)]*(100000/locals()['grid_psl'+str(mod+1)])**con

    # t = min(len(theta_t2m),len(theta_700))
    M_700  = theta_t2m - theta_700
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

    plt.plot(lats,f,color=use_colors[mod],label=warming_modname[mod],linestyle='solid')

    # plt.legend()
    # plt.xlabel('latitude')
    # plt.title('seasonal CAO zonal mean ditribution\nhistorical and warming')
    # plt.savefig('../figures/final/warming_M_lats.png')
