import matplotlib.pyplot as plt
import numpy as np
import glob
#####data in npz npzfiles
#####cw=cyg, aw=am, b=bias, ar=r, cT=cT, aT=aT, lon=longt, lat=latit
fig = plt.figure(figsize=(12,6))
fn = glob.glob('/glade/work/geethma/research/npzfilesn/2019/january/u10_01062019.npz')  #u10*npz
fn = np.array(fn)
awf=[]
cwf = []
# print(fn)
for i in range(0,len(fn)): #len(fn)
    print (i)
    d = np.load(fn[i])
    aw = np.array(d['aw'])
    cw = np.array(d['cw'])
    ##################
    for i in range(0,len(aw)):
        awf.append(aw[i])
        cwf.append(cw[i])
# #     awf.append(aw)
# #     cwf.append(cw)
# # # plt.scatter(cwf,awf, c='b')
# #     awf = np.array(awf)
# #     cwf = np.array(cwf)
#     plt.scatter(cw,aw, c='b')
#     # print(awf.shape())
#     # print(awf.shape(),'\n')
# plt.scatter(cwf,awf, c='b')
coefficients = np.polyfit(cwf, awf, 1)
m = round(coefficients[0],2)
c = round(coefficients[1],2)
poly = np.poly1d(coefficients)
new_x = np.linspace(min(cwf), max(cwf))
new_y = poly(new_x)
plt.plot(cwf, awf, "o", label='scatter plot')
plt.plot(new_x, new_y, label='polyfit')
x = [0,17.5]
y = [0,17.5]
plt.plot(x,y,c='r',label='ideal mapping')
plt.legend()
plt.text(12, 4, 'gradient = '+str(m))
plt.text(12, 3, 'intercept = '+str(c)+' ms-1')
# # print(m, c)
plt.title('AMSR vs CYGNSS Wind Speeds on 06th Jan 2019')
plt.xlabel('CYGNSS (m/s)')
plt.ylabel('AMSR (m/s)')
plt.savefig('/glade/work/geethma/research/npzfilesn/2019/january/plots/jan6_plot_amVScyg.png') #/plots
#"/glade/work/geethma/research/npzfilesn/2019/jann/u10_01072019.npz"
