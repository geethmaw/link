# @Author: geethmawerapitiya
# @Date:   2022-07-15T00:29:10-06:00
# @Project: Research
# @Filename: macwind_function.py
# @Last modified by:   geethmawerapitiya
# @Last modified time: 2022-07-15T00:32:51-06:00

def macwind():
    ## OBSERVATIONS
    import glob
    merlist = np.sort(glob.glob('../data_merra/all_lat_lon/level/MERRA2_*.nc'))
    sfclist = np.sort(glob.glob('../data_merra/all_lat_lon/surface/MERRA2_*.nc'))
    maclist = np.sort(glob.glob('../MACLWP_dailymean/take/wind1deg*.nc4'))

    new_list_s = []
    new_list_m = []
    new_list_c = []

    s = 0
    m = 0
    length = max(len(merlist), len(sfclist))

    while m != length:
        print(s,m)
        name_s = os.path.basename(sfclist[s])
        date_s = name_s.split(".")[2]

        name_m = os.path.basename(merlist[m])
        date_m = name_m.split(".")[2]
        print(sfclist[s],date_s)
        print(merlist[m],date_m)

        if date_s==date_m:
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
                        # print(maclist[k],str(mactime[r]). zfill(2))
                        # print(new_list_s[i],date_s)
                        new_m.append(new_list_m[i])
                        new_s.append(new_list_s[i])
                        macdate.append(date_mac+str(mactime[r]). zfill(2))

                        flag = 1
                        break;

            if flag == 1:
                break;


    macwind = np.array(macwind)

    return(macwind)
