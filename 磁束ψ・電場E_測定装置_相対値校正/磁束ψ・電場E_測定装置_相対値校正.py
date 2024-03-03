
#=======================================================

import os
import csv
import numpy as np
import scipy.fftpack as sf
from scipy.io import netcdf
import matplotlib.pyplot as plt

savedir = os.getcwd() + "/" + "相対値較正"
if not(os.access(savedir,os.F_OK)):
    os.mkdir(savedir)

Fontsize = 20
plt.rcParams.update({'mathtext.default':'default', 'mathtext.fontset':'stix', 'font.family':'Arial', 'font.size':Fontsize})

#=======================================================

Shot = 'Data_gpa/PF3(2kV)'

Time = 6000    # PF3(2kV)抽出時間
LP_freq = 200  # ローパス [kHz]

FigsizeM = (10,5)
FigsizeL = (12,4)

(xmin, xmax) = (2000, 10000)     # 時間発展横軸
(ymin, ymax) = (-0.0475, 0.02)   # 時間発展縦軸

(rmin, rmax) = (90, 500)         # 径方向分布横軸
(umin, umax) = (-0.042, -0.018)  # 径方向分布縦軸
(cmin, cmax) = (0.95, 1.05)      # 較正値縦軸

#=======================================================

sample = 20000

BzCH = 16
flag_bzll = range(44,60)
Posi_bzll = [110,150,200,250,270,290,320,340,380,400,420,440,450,460,470,480]
Coil_dir  = [-1,-1,1,1,1,1,1,1,-1,1,1,1,-1,1,1,-1]

name_bzll = []
for i in range(BzCH):
    Oscillo_2 = 'dl716_2_%s' % (i+1)
    name_bzll.append(Oscillo_2)

#=======================================================

f1 = netcdf.netcdf_file('%s.gpa' % (Shot))

sample_period = getattr(f1,'sample_period')
full_range1   = getattr(f1,'full_range')
precision1    = getattr(f1,'precision')

init_time = 0 #[sec]
fin_time = init_time + sample_period[flag_bzll[0]] * sample #[sec]
dt = sample_period[flag_bzll[0]]
time = np.arange(init_time, fin_time, dt) #[sec]
time = time * 1e+6 #[usec]

#=======================================================
#  PF3(2kV)の時間発展
#=======================================================

Val = []
for i in range(BzCH):
    val = f1.variables[name_bzll[i]][:] * full_range1[flag_bzll[i]] / precision1[flag_bzll[i]] * np.array(Coil_dir[i])
    val = sf.fft(val)
    freq = sf.fftfreq(int(sample), dt)
    val[abs(freq) > LP_freq*1e3] = 0
    val = sf.ifft(val)
    Val.append(np.real(val))

fig,ax = plt.subplots(figsize=FigsizeM)
ax.plot(time, Val[0],  label='R = %s  [mm]' % (Posi_bzll[0]))
ax.plot(time, Val[1],  label='R = %s  [mm]' % (Posi_bzll[1]))
ax.plot(time, Val[2],  label='R = %s  [mm]' % (Posi_bzll[2]))
ax.plot(time, Val[3],  label='R = %s  [mm]' % (Posi_bzll[3]))
ax.plot(time, Val[4],  label='R = %s  [mm]' % (Posi_bzll[4]))
ax.plot(time, Val[5],  label='R = %s  [mm]' % (Posi_bzll[5]))
ax.plot(time, Val[6],  label='R = %s  [mm]' % (Posi_bzll[6]))
ax.plot(time, Val[7],  label='R = %s  [mm]' % (Posi_bzll[7]))
ax.plot(time, Val[8],  label='R = %s  [mm]' % (Posi_bzll[8]))
ax.plot(time, Val[9],  label='R = %s  [mm]' % (Posi_bzll[9]))
ax.plot(time, Val[10], label='R = %s  [mm]' % (Posi_bzll[10]))
ax.plot(time, Val[11], label='R = %s  [mm]' % (Posi_bzll[11]))
ax.plot(time, Val[12], label='R = %s  [mm]' % (Posi_bzll[12]))
ax.plot(time, Val[13], label='R = %s  [mm]' % (Posi_bzll[13]))
ax.plot(time, Val[14], label='R = %s  [mm]' % (Posi_bzll[14]))
ax.plot(time, Val[15], label='R = %s  [mm]' % (Posi_bzll[15]))
plt.gca().xaxis.set_tick_params(which='both', direction='in', bottom=True)
plt.gca().yaxis.set_tick_params(which='both', direction='in', left=True)
plt.hlines([0], xmin, xmax, "black", linestyle=":", lw=2)
plt.vlines([Time], ymin, ymax, "black", linestyle=":", lw=2)
ax.set_xlabel("Time  [μs]", labelpad=15, fontsize=Fontsize)
ax.set_ylabel("dBz/dt  [V]", labelpad=12, fontsize=Fontsize)
plt.title("PF3 (2kV)    #200818006", fontsize=Fontsize)
plt.tick_params(pad=8)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.xticks(np.arange(2000, 12000, 2000))
plt.yticks(np.arange(-0.04, 0.04, 0.02))
plt.legend(bbox_to_anchor=(1, 1.02), loc='upper left', labelspacing=0.8, fontsize=9.5, ncol=1)
plt.savefig("%s/dBz(PF3)_T.png" %(savedir), bbox_inches='tight')
plt.close()

#=======================================================
#  最小二乗法フィッティング
#=======================================================

y_calib = []
for i in range(BzCH):
    calib = Val[i][Time]
    y_calib.append(calib)
y_calib = np.array(y_calib)

y_calib_fit = []
for i in range(BzCH):
    fit = np.poly1d(np.polyfit(Posi_bzll, y_calib, 3))(Posi_bzll)[i]
    y_calib_fit.append(fit)
y_calib_fit = np.array(y_calib_fit)

fig,ax = plt.subplots(figsize=FigsizeL)
ax.plot(Posi_bzll, y_calib, 'o', markersize=8, label='Measured value')
ax.plot(Posi_bzll, y_calib_fit, '-o', markersize=8, lw=2, label='Least squares (cubic)')
plt.gca().xaxis.set_tick_params(which='both', direction='in', bottom=True)
plt.gca().yaxis.set_tick_params(which='both', direction='in', left=True)
ax.set_xlabel("R  [mm]", labelpad=15, fontsize=Fontsize)
ax.set_ylabel("dBz/dt  [V]", labelpad=12, fontsize=Fontsize)
plt.title("PF3 (2kV)    T = %s [μs]     #200818006" % (Time), fontsize=Fontsize)
plt.tick_params(pad=8)
plt.xlim(rmin, rmax)
plt.ylim(umin, umax)
plt.xticks(np.arange(100, 550, 50))
plt.yticks(np.arange(-0.04, -0.01, 0.01))
plt.legend(fontsize=20, loc='upper left')
plt.savefig("%s/dBz(PF3)_R.png" %(savedir), bbox_inches='tight')
plt.close()

#=======================================================
#  低周波プローブ較正値 (相対値較正)
#=======================================================

y_relative = y_calib_fit / y_calib

fig,ax = plt.subplots(figsize=FigsizeL)
ax.plot(Posi_bzll, y_relative, 'o', markersize=10)
plt.gca().xaxis.set_tick_params(which='both', direction='in', bottom=True)
plt.gca().yaxis.set_tick_params(which='both', direction='in', left=True)
plt.hlines([1], rmin, rmax, "black", linestyle=":", lw=3)
ax.set_xlabel("R  [mm]", labelpad=15, fontsize=Fontsize)
ax.set_ylabel("Calibration value  $C_{R}$", labelpad=12, fontsize=Fontsize)
plt.title("PF3 (2kV)    #200818006", fontsize=Fontsize)
plt.tick_params(pad=8)
plt.xlim(rmin, rmax)
plt.ylim(cmin, cmax)
plt.xticks(np.arange(100, 550, 50))
plt.yticks(np.arange(0.95, 1.05, 0.05))
plt.savefig("%s/dBz(PF3)_C.png" %(savedir), bbox_inches='tight')
plt.close()

print(*y_relative, sep=', ')

csvname = '%s/Relative_Value.csv' % (savedir)
csvfile = open(csvname, 'w', newline="")
writer = csv.writer(csvfile)
file = ([y_relative])
file_r = list(map(list,zip(*file)))
writer.writerows(file_r) 
csvfile.close()

#=======================================================