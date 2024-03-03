#=======================================================
#  Import gpa file
#=======================================================

(Operation_Class, Operation_Shot) = (4, 2)

Operation1 = 'Open'
Shotnumber1 = ['#201028007', '#201028013']
(Shot1_1, Shot1_2) = ('Data_gpa/201028007', 'Data_gpa/201028013')

Operation2 = 'Short [1\']'
Shotnumber2 = ['#201103007', '#201103014']
(Shot2_1, Shot2_2) = ('Data_gpa/201103007', 'Data_gpa/201103014')

Operation3 = 'Short [4\']'
Shotnumber3 = ['#201030009', '#201030006']
(Shot3_1, Shot3_2) = ('Data_gpa/201030009', 'Data_gpa/201030006')

Operation4 = 'Short [1\'][2\'][3\'][4\']'
Shotnumber4 = ['#201029003', '#201029011']
(Shot4_1, Shot4_2) = ('Data_gpa/201029003', 'Data_gpa/201029011')

Operation = [Operation1, Operation2, Operation3, Operation4]

Shot = [[Shot1_1, Shot1_2], [Shot2_1, Shot2_2], [Shot3_1, Shot3_2], [Shot4_1, Shot4_2]]
Shotnumber = [Shotnumber1, Shotnumber2, Shotnumber3, Shotnumber4]

#=======================================================
#  Import module
#=======================================================

import os
import math
import numpy as np
import scipy.fftpack as sf
from scipy.io import netcdf
import matplotlib.pyplot as plt
import matplotlib.ticker as ptick

#=======================================================
#  Parameter of graph
#=======================================================

Fontsize = 18
LP_freq = 200

Figsize = (12,4)

(TeTxmin, TeTxmax) = (9450, 9600)
(TeTymin, TeTymax) = (3, 34)

(IcTxmin, IcTxmax) = (9450, 9600)
(IcTymin, IcTymax) = (-0.05, 0.6)

(neTxmin, neTxmax) = (9450, 9600)
(neTymin, neTymax) = (-0.3e19, 3.5e19)

xticks_time = np.arange(9450,9625,25)

#=======================================================
#  Formatting of graph
#=======================================================

savedir = os.getcwd() + "/" + "Langmuir_Probe"
if not(os.access(savedir,os.F_OK)):
    os.mkdir(savedir)

plt.rcParams.update({'mathtext.default':'default', 'mathtext.fontset':'stix', 'font.family':'Arial', 'font.size':Fontsize})

#=======================================================
#  Parameter of oscillo scope
#=======================================================

sample = 10000
(sampleA, sampleB) = (9400, 9800)
sampleC = sampleB - sampleA

LangCH = 1
flag = range(74,76)
ch_name = ["dl708_2_7","dl708_2_8"]

Z_Posi = 'Z = 0 [mm]'
Posi_Lang = [200]

#=======================================================
#  Parameter of Langmuir probe
#=======================================================

Amp = 20          # Differencial probe ratio
e   = 1.602e-19   # Elementary charge [C]
me  = 9.11e-31    # Electron mass [kg]
mi  = 4*1.67e-27  #　Helium mass [kg]
kb  = 1.381e-23   # Boltzman's constant
ep0 = 8.85e-12    # Dielectric constant of vacuum
B   = 0.25        # Magneric field in UTST [T]
SC  = 6.786e-6    # Electrode surface area [m^2]

#=======================================================
#  Definition of gpa file data
#=======================================================

def File_Data(shotname):

    f1 = netcdf.netcdf_file('%s.gpa' % (shotname))

    precision1 = getattr(f1,'precision')
    full_range1 = getattr(f1,'full_range')
    sample_period = getattr(f1,'sample_period')

    init_time = 0 #[sec]
    fin_time = init_time + sample_period[flag[0]] * sample #[sec]
    dt = sample_period[flag[0]]
    time = np.arange(init_time, fin_time, dt) #[sec]
    time = time * 1e+6 #[usec]
    time = time[sampleA:sampleB]

    return(f1, sample_period, full_range1, precision1, init_time, fin_time, dt, time)

#=======================================================
#  Definition of Langmuir probe
#=======================================================

def Langmuir_Probe():
    
    global f1, ch_name, flag, precision1
    
    te = f1.variables[ch_name[1]][:] * full_range1[flag[1]] / precision1[flag[1]] * (Amp / math.log(2))
    te = te[sampleA:sampleB]

    tefft = sf.fft(te)
    freq = sf.fftfreq(int(sampleC), dt)  
    te[abs(freq) > LP_freq * 1e3] = 0
    Te = sf.ifft(tefft)
    Te = np.real(Te)

    ic = -f1.variables[ch_name[0]][:] * full_range1[flag[0]] / precision1[flag[0]] * 24.2
    ic = ic[sampleA:sampleB]

    icfft = sf.fft(ic)
    freq = sf.fftfreq(int(sampleC), dt)  
    ic[abs(freq) > LP_freq * 1e3] = 0
    ic[abs(freq) < 0] = 0
    Ic = sf.ifft(icfft)
    Ic = np.real(Ic)

    ne = ((Ic*np.exp(1/2))/(e*SC))*np.sqrt(mi/(kb*Te*11600))

    return (Te, Ic, ne)

#=======================================================
#  Definition of Langmuir probe multiplot
#=======================================================

def Langmuir_Multi(signal1, signal2, signal3, signal4, type_graph, str_title):

    global time

    dict = {'Te':'Te  [eV]', 'Ic':'Ic  [A]', 'ne':'ne  [$m^{-3}$]', 'Position':'R = 200  [mm]'}

    savedir_Langmuir_multi = savedir + "/" + "Langmuir_Multiplot"
    if not(os.access(savedir_Langmuir_multi,os.F_OK)):
        os.mkdir(savedir_Langmuir_multi)

    def Langmuir_Multi_Sub(Bool, linewidth, xmin, xmax, ymin, ymax, location, col, name):

        fig,ax = plt.subplots(figsize=Figsize)
        ax.plot(time, signal1, lw=2, label = Operation[0])
        ax.plot(time, signal2, lw=2, label = Operation[1])
        ax.plot(time, signal3, lw=2, label = Operation[2])
        ax.plot(time, signal4, lw=2, label = Operation[3])
        ax.set_xlabel("Time  [$\mu$s]", labelpad=8, fontsize=Fontsize)
        ax.set_ylabel("%s" %(dict[type_graph]), labelpad=12, fontsize=Fontsize)
        if (Bool==True):
            ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
            ax.yaxis.offsetText.set_fontsize(Fontsize)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.gca().xaxis.set_tick_params(which='both', direction='in', bottom=True)
        plt.gca().yaxis.set_tick_params(which='both', direction='in', left=True)
        plt.title("%s      %s" % (name, dict[str_title]), fontsize=Fontsize)
        plt.hlines([0], xmin, xmax, "black", linestyle=":", lw=linewidth)
        plt.legend(fontsize=16, loc=location, ncol=col)
        plt.tick_params(pad=8)
        plt.xlim(xmin, xmax)
        plt.xticks(xticks_time)
        plt.ylim(ymin, ymax)
        plt.savefig("%s/%s.png" %(savedir_Langmuir_multi, name), bbox_inches='tight')
        plt.close()

    if(type_graph == 'Te'):
        Langmuir_Multi_Sub(False, 0, TeTxmin, TeTxmax, TeTymin, TeTymax, 'lower right', 2, "Te")

    elif(type_graph == 'Ic'):
        Langmuir_Multi_Sub(False, 2, IcTxmin, IcTxmax, IcTymin, IcTymax, 'upper left', 1, "Ic")

    elif(type_graph == 'ne'):
        Langmuir_Multi_Sub(True, 2, neTxmin, neTxmax, neTymin, neTymax, 'upper left', 1, "ne")

    else:()

#=======================================================
#  Definition of Langmuir probe errorplot
#=======================================================

def Langmuir_Error(signal1, signal2, signal3, signal4, signal5, signal6, signal7, signal8, type_graph, str_title):

    global time

    dict = {'Te':'Te  [eV]', 'Ic':'Ic  [A]', 'ne':'ne  [$m^{-3}$]', 'Position':'R = 200  [mm]'}

    savedir_Langmuir_error = savedir + "/" + "Langmuir_Errorplot"
    if not(os.access(savedir_Langmuir_error,os.F_OK)):
        os.mkdir(savedir_Langmuir_error)

    def Langmuir_Error_Sub(Bool, linewidth, xmin, xmax, ymin, ymax, location, col, name):

        fig,ax = plt.subplots(figsize=Figsize)
        ax.errorbar(time, signal1, yerr=signal2, elinewidth=0.5, lw=2, label = Operation[0])
        ax.errorbar(time, signal3, yerr=signal4, elinewidth=0.5, lw=2, label = Operation[1])
        ax.errorbar(time, signal5, yerr=signal6, elinewidth=0.5, lw=2, label = Operation[2])
        ax.errorbar(time, signal7, yerr=signal8, elinewidth=0.5, lw=2, label = Operation[3])
        ax.set_xlabel("Time  [$\mu$s]", labelpad=8, fontsize=Fontsize)
        ax.set_ylabel("%s" %(dict[type_graph]), labelpad=12, fontsize=Fontsize)
        if (Bool==True):
            ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
            ax.yaxis.offsetText.set_fontsize(Fontsize)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.gca().xaxis.set_tick_params(which='both', direction='in', bottom=True)
        plt.gca().yaxis.set_tick_params(which='both', direction='in', left=True)
        plt.title(name + "      %s" % (dict[str_title]), fontsize=Fontsize)
        plt.hlines([0], xmin, xmax, "black", linestyle=":", lw=linewidth)
        plt.legend(fontsize=16, loc=location, ncol=col)
        plt.tick_params(pad=8)
        plt.xlim(xmin, xmax)
        plt.xticks(xticks_time)
        plt.ylim(ymin, ymax)
        plt.savefig("%s/%s.png" %(savedir_Langmuir_error, name), bbox_inches='tight')
        plt.close()        

    if(type_graph == 'Te'):
        Langmuir_Error_Sub(False, 0, TeTxmin, TeTxmax, TeTymin, TeTymax, 'lower right', 2, "Te")

    elif(type_graph == 'Ic'):
        Langmuir_Error_Sub(False, 2, IcTxmin, IcTxmax, IcTymin, IcTymax, 'upper left', 1, "Ic")

    elif(type_graph == 'ne'):
        Langmuir_Error_Sub(True, 2, neTxmin, neTxmax, neTymin, neTymax, 'upper left', 1, "ne")

    else:()

#=======================================================
#  Return of Langmuir probe
#=======================================================

Te, Ic, ne = np.zeros((Operation_Class, Operation_Shot, sampleC)), np.zeros((Operation_Class, Operation_Shot, sampleC)), np.zeros((Operation_Class, Operation_Shot, sampleC))
for i in range(Operation_Class):
    for j in range(Operation_Shot):
        for shotname in [Shot[i][j]]:
            (f1, sample_period, full_range1, precision1, init_time, fin_time, dt, time) = File_Data(shotname)
            Te[i][j], Ic[i][j], ne[i][j] = Langmuir_Probe()
            Te, Ic, ne = Te.reshape([Operation_Class, Operation_Shot, sampleC]), Ic.reshape([Operation_Class, Operation_Shot, sampleC]), ne.reshape([Operation_Class, Operation_Shot, sampleC])

    Te_ave, Te_err = [], []
    Ic_ave, Ic_err = [], []
    ne_ave, ne_err = [], []
    for i in range(Operation_Class): 
        Te_ave_sub, Te_err_sub = [], []
        Ic_ave_sub, Ic_err_sub = [], []
        ne_ave_sub, ne_err_sub = [], []            
        for j in range(sampleC):
            Te_max, Te_min = max([Te[i][0][j], Te[i][1][j]]), min([Te[i][0][j], Te[i][1][j]])
            Ic_max, Ic_min = max([Ic[i][0][j], Ic[i][1][j]]), min([Ic[i][0][j], Ic[i][1][j]])
            ne_max, ne_min = max([ne[i][0][j], ne[i][1][j]]), min([ne[i][0][j], ne[i][1][j]])
            Te_average, Te_error = Te_max/2 + Te_min/2, Te_max - Te_min
            Ic_average, Ic_error = Ic_max/2 + Ic_min/2, Ic_max - Ic_min
            ne_average, ne_error = ne_max/2 + ne_min/2, ne_max - ne_min
            Te_ave_sub.append(Te_average), Te_err_sub.append(Te_error)
            Ic_ave_sub.append(Ic_average), Ic_err_sub.append(Ic_error)
            ne_ave_sub.append(ne_average), ne_err_sub.append(ne_error)
        Te_ave.append(Te_ave_sub), Te_err.append(Te_err_sub)
        Ic_ave.append(Ic_ave_sub), Ic_err.append(Ic_err_sub)
        ne_ave.append(ne_ave_sub), ne_err.append(ne_err_sub)

#=======================================================
#  Return of Langmuir probe multiplot
#=======================================================

    Langmuir_Multi(Te[0][0], Te[1][0], Te[2][0], Te[3][0], "Te", "Position")
    Langmuir_Multi(Ic[0][0], Ic[1][0], Ic[2][0], Ic[3][0], "Ic", "Position")
    Langmuir_Multi(ne[0][0], ne[1][0], ne[2][0], ne[3][0], "ne", "Position")

#=======================================================
#  Return of Langmuir probe errorplot
#=======================================================

    Langmuir_Error(Te_ave[0], Te_err[0], Te_ave[1], Te_err[1], Te_ave[2], Te_err[2], Te_ave[3], Te_err[3], "Te", "Position")
    Langmuir_Error(Ic_ave[0], Ic_err[0], Ic_ave[1], Ic_err[1], Ic_ave[2], Ic_err[2], Ic_ave[3], Ic_err[3], "Ic", "Position")
    Langmuir_Error(ne_ave[0], ne_err[0], ne_ave[1], ne_err[1], ne_ave[2], ne_err[2], ne_ave[3], ne_err[3], "ne", "Position")

#=======================================================