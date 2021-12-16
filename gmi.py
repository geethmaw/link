# import gzip
# with gzip.open('/netdata/R1/data/wgeethma/gmi/2018/f35_20180831v8.2.gz', 'rb') as f:
#     file_content = f.read()
#     name = '/Grid/SST'
#     data = f[name][:]


# import gzip
# import shutil
# with gzip.open('/netdata/R1/data/wgeethma/gmi/2018/f35_20180831v8.2.gz', 'rb') as f_in:
#     with open('/netdata/R1/data/wgeethma/gmi/2018/file.txt', 'wb') as f_out:
#         shutil.copyfileobj(f_in, f_out)

import netCDF4
import gzip

with gzip.open('/netdata/R1/data/wgeethma/gmi/2018/f35_20180831v8.2.gz') as gz:
    with netCDF4.Dataset('dummy', mode='r', memory=gz.read()) as nc:
        print(nc.variables)
