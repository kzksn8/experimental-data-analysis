
#=======================================================

import os
import numpy as np
import matplotlib.pyplot as plt

savedir = os.getcwd() + "/" + "絶対値較正"
if not(os.access(savedir,os.F_OK)):
    os.mkdir(savedir)

Fontsize = 20
plt.rcParams.update({'mathtext.default':'default', 'mathtext.fontset':'stix','font.family':'Arial', 'font.size':Fontsize})

#=======================================================
#  差動プローブ較正値
#=======================================================

x_calib11 = []
for i in range(11):
    if i<9:
        x_name = 'No.0%s' % (i+1)
    else:
        x_name = 'No.%s' % (i+1)
    x_calib11.append(x_name)

rfile = 'Data_csv/Differential_Probe_1.csv'
data = np.loadtxt(rfile, comments='#' ,delimiter=',')
y_calib_1 = data[1]

x_calib22 = []
for i in range(11):
    x_name = 'No.%s' % (i+12)
    x_calib22.append(x_name)

rfile = 'Data_csv/Differential_Probe_2.csv'
data = np.loadtxt(rfile, comments='#' ,delimiter=',')
y_calib_2 = data[1]

#=======================================================

def Graph_Format1(ymin, ymax):

    plt.grid(which = "major", axis = "x", color = "black", linestyle = "--", linewidth = 0.2)
    plt.grid(which = "major", axis = "y", color = "black", linestyle = "--", linewidth = 0.2)
    plt.gca().xaxis.set_tick_params(which='both', direction='in', bottom=True)
    plt.gca().yaxis.set_tick_params(which='both', direction='in', left=True)
    plt.axhline(y=1, color='black', linestyle=":", lw=3)
    plt.tick_params(pad=8)
    plt.ylim(ymin, ymax)

fig = plt.figure(figsize=(15,6))
    
ax1 = fig.add_subplot(211)
ax1.plot(x_calib11, y_calib_1, marker="o", markersize=12, linestyle='None')
Graph_Format1(0.99, 1.01)

ax2 = fig.add_subplot(212)
ax2.plot(x_calib22, y_calib_2, marker="o", markersize=12, linestyle='None')
Graph_Format1(0.99, 1.015)

plt.subplots_adjust(hspace=0.3)
fig.text(0.0625, 0.5, 'Calibration value  $C_{D}$', fontsize=20, ha='center', va='center', rotation='vertical')
plt.savefig("%s/Differential_Probe.png" %(savedir), bbox_inches='tight')
plt.close()

#=======================================================
#  低周波プローブ較正値 (絶対値較正)
#=======================================================

rfile = 'Data_csv/Magnetic_Probe_Abs.csv'
data = np.loadtxt(rfile, comments='#' ,delimiter=',')

x_freq = data[0]
y_abs1 = data[1]
y_abs2 = data[2]

#=======================================================

def Graph_Format2(xmin, xmax, ymin, ymax, Yticks):

    plt.tick_params(pad=8)
    plt.gca().xaxis.set_tick_params(which='both', direction='in', bottom=True)
    plt.gca().yaxis.set_tick_params(which='both', direction='in', left=True)
    plt.hlines([1], xmin, xmax, "black", linestyle=":", lw=3)
    ax.set_xscale('log')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.yticks(Yticks)

fig,ax=plt.subplots(figsize=(10,4))
ax.plot(x_freq, y_abs1, marker="o", markersize=10, lw=3)
ax.set_xlabel("Freqency  [kHz]", labelpad=12, fontsize=Fontsize)
ax.set_ylabel("$B_{coil} \ / \ B_{helm}$", labelpad=20, fontsize=Fontsize)
Graph_Format2(0.8e0, 1.2e3, -0.2, 4.2, np.arange(0, 5, 1))
plt.savefig("%s/Magnetic_Probe_Abs1.png" %(savedir), bbox_inches='tight')
plt.close()

fig,ax=plt.subplots(figsize=(10,4))
ax.plot(x_freq, y_abs2, marker="o", markersize=10, lw=3)
ax.set_xlabel("Freqency  [kHz]", labelpad=12, fontsize=Fontsize)
ax.set_ylabel("$C_{A} \ * \ (B_{coil} \ / \ B_{helm})$", labelpad=24, fontsize=Fontsize)
Graph_Format2(0.8e0, 1.2e3, -0.2, 4.2, np.arange(0, 5, 1))
plt.savefig("%s/Magnetic_Probe_Abs2.png" %(savedir), bbox_inches='tight')
plt.close()

#=======================================================
#  低周波プローブ較正値 (位相特性)
#=======================================================

rfile = 'Data_csv/Magnetic_Probe_Phase.csv'
data = np.loadtxt(rfile, comments='#' ,delimiter=',')

x_freq = data[0]
y_phase = data[1]

#=======================================================

fig,ax=plt.subplots(figsize=(10,4))
ax.plot(x_freq, y_phase, marker="o", markersize=10, lw=3)
ax.set_xlabel("Freqency  [kHz]", labelpad=12, fontsize=Fontsize)
ax.set_ylabel("Phase  [deg]", labelpad=12, fontsize=Fontsize)
plt.tick_params(pad=8)
plt.gca().xaxis.set_tick_params(which='both', direction='in', bottom=True)
plt.gca().yaxis.set_tick_params(which='both', direction='in', left=True)
plt.hlines([0], 0.8, 1200, "black", linestyle=":", lw=3)
plt.yticks(np.arange(-90, 180, 90))
ax.set_xscale('log')
plt.xlim(0.8, 1.2e3)
plt.ylim(-110, 110)
plt.savefig("%s/Magnetic_Probe_Phase.png" %(savedir), bbox_inches='tight')
plt.close()

#=======================================================