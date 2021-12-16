import matplotlib.pyplot as plt
import numpy as np
import glob
#####cw=cyg, aw=am, b=bias, ar=r, cT=cT, aT=aT, lon=longt, lat=latit
fig = plt.figure(figsize=(12,6))
fn = glob.glob('/glade/work/geethma/research/npzfilesn/2019/january/u10*npz')
fn = np.array(fn)
arf=[]
bf = []
######
# print(fn)
for i in range(0,len(fn)):
    d = np.load(fn[i])
    ar = np.array(d['ar'])
    b = np.array(d['b'])
    for i in range(0,len(ar)):
        if ar[i]>0:
            arf.append(np.log(1+ar[i]))
            bf.append(b[i])
#        if ar[i]>0:
#            plt.scatter(ar,b, c='r')
plt.scatter(arf,bf, c='r')
plt.title('Wind bias Vs amsr rain rate on 2019 January')
plt.xlabel('Rain rate (mm h-1)')
plt.ylabel('Bias of cygnss and asmsr2 winds (ms-1)')
# plt.show()
plt.savefig('/glade/work/geethma/research/npzfilesn/2019/january/plots/n_log_plot_biasVSrr.png')
