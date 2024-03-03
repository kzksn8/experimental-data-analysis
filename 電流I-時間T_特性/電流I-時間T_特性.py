#=======================================================
#  Import gpa file
#=======================================================

(Operation_Class, Operation_Shot) = (7, 2)

Operation1 = 'Open'
Shotnumber1 = ['#201028014', '#201028007']
(Shot1_in, Shot1_out) = ('Data_gpa/201028014', 'Data_gpa/201028007')

Operation2 = 'Short [1\']'
Shotnumber2 = ['#201103007', '#201103014']
(Shot2_in, Shot2_out) = ('Data_gpa/201103007', 'Data_gpa/201103014')

Operation3 = 'Short [2\']'
Shotnumber3 = ['#201105011', '#201105004']
(Shot3_in, Shot3_out) = ('Data_gpa/201105011', 'Data_gpa/201105004')

Operation4 = 'Short [3\']'
Shotnumber4 = ['#201216015', '#201216011']
(Shot4_in, Shot4_out) = ('Data_gpa/201216015', 'Data_gpa/201216011')

Operation5 = 'Short [4\']'
Shotnumber5 = ['#201030009', '#201030006']
(Shot5_in, Shot5_out) = ('Data_gpa/201030009', 'Data_gpa/201030006')

Operation6 = 'Short [1\'][2\'][3\'][4\']'
Shotnumber6 = ['#201029003', '#201029011']
(Shot6_in, Shot6_out) = ('Data_gpa/201029003', 'Data_gpa/201029011')

Operation7 = 'Short [1\'2\'3\'4\']'
Shotnumber7 = ['#201215002', '#201215010']
(Shot7_in, Shot7_out) = ('Data_gpa/201215002', 'Data_gpa/201215010')

Operation = [Operation1, Operation2, Operation3, Operation4, Operation5, Operation6, Operation7]

Shot = [[Shot1_in, Shot1_out], [Shot2_in, Shot2_out], [Shot3_in, Shot3_out], [Shot4_in, Shot4_out], [Shot5_in, Shot5_out], [Shot6_in, Shot6_out], [Shot7_in, Shot7_out]]
Shotnumber = [Shotnumber1, Shotnumber2, Shotnumber3, Shotnumber4, Shotnumber5, Shotnumber6, Shotnumber7]

#=======================================================
#  Import module
#=======================================================

import os
import numpy as np
import scipy.fftpack as sf
from scipy.io import netcdf
import matplotlib.pyplot as plt

#=======================================================
#  Parameters of graph
#=======================================================

fontsize = 18
LP_freq = 200

Figsize = (10, 3.5)

(Ixmin, Ixmax) = (9450, 9600)
(Iymin, Iymax) = (-0.2, 1.1)

xticks_time = np.arange(9450,9625,25)

#=======================================================
#  Formatting of graph
#=======================================================

savedir = os.getcwd() + "/" + "Electrodes"
if not(os.access(savedir,os.F_OK)):
    os.mkdir(savedir)

plt.rcParams.update({'mathtext.default':'default', 'mathtext.fontset':'stix', 'font.family':'Arial', 'font.size':fontsize})

#=======================================================
#  Parameters of oscillo scope
#=======================================================

sample = 10000
(sampleA, sampleB) = (9200, 9800)
sampleC = sampleB - sampleA

ICH = 4
flag = range(39,43)

ch_name = []
for i in range(ICH):
    Oscillo = 'dl716_1_%s' % (i+12)
    ch_name.append(Oscillo)

Posi = ['R = 112.5 - 127.5  [mm]', 'R = 132.5 - 147.5  [mm]', 
        'R = 152.5 - 167.5  [mm]', 'R = 172.5 - 187.5  [mm]']

#=======================================================
#  Definition of gpa file data
#=======================================================

def File_Data(shotname):

    f1 = netcdf.netcdf_file('%s.gpa' % (shotname))

    precision1    = getattr(f1,'precision')
    full_range1   = getattr(f1,'full_range')
    sample_period = getattr(f1,'sample_period')

    init_time = 0 #[sec]
    fin_time = init_time + sample_period[flag[0]] * sample #[sec]
    dt = sample_period[flag[0]]
    time = np.arange(init_time, fin_time, dt) #[sec]
    time = time * 1e+6 #[usec]
    time = time[sampleA:sampleB]

    return(f1, sample_period, full_range1, precision1, init_time, fin_time, dt, time)

#=======================================================
#  Difinition of current
#=======================================================

def Current():

    global f1, ch_name, flag, full_range1, precision1

    current = []

    for i in range(ICH):

        val = f1.variables[ch_name[i]][:] * full_range1[flag[i]] / precision1[flag[i]] * 24.2 / 1000
        val = val[sampleA:sampleB]
        val = sf.fft(val)
        freq = sf.fftfreq(int(sampleC), dt)
        val[abs(freq) > LP_freq*1e3] = 0
        val = sf.ifft(val)
        current.append(np.real(val))

    return current
    
#=======================================================
#  Definition of I multiplot (1)
#=======================================================

def I_Multi1(signal1, signal2, signal3, signal4, type_graph, str_title):

    global time

    dict = {'I1':'Current  [kA]', 'I2':'Current  [kA]', 'I3':'Current  [kA]',
            'Name1':'1組のみ短絡した場合', 'Name2':'4組を同時短絡した場合', 'Name3':'全て短絡させた場合'}

    savedir_I_multi = savedir + "/" + "Current_Multiplot"
    if not(os.access(savedir_I_multi,os.F_OK)):
        os.mkdir(savedir_I_multi)

    def I_Multi_Sub1():

        fig,ax = plt.subplots(figsize=Figsize)
        ax.plot(time, signal1, lw=2, label=Operation2)
        ax.plot(time, signal2, lw=2, label=Operation3)
        ax.plot(time, signal3, lw=2, label=Operation4)
        ax.plot(time, signal4, lw=2, label=Operation5)
        ax.set_xlabel("Time [$\mu$s]", labelpad=8, fontsize=fontsize)
        ax.set_ylabel("%s" %(dict[type_graph]), labelpad=15, fontsize=fontsize)
        plt.hlines([0], Ixmin, Ixmax, "black", linestyle=":", lw=2)
        plt.gca().xaxis.set_tick_params(which='both', direction='in', bottom=True)
        plt.gca().yaxis.set_tick_params(which='both', direction='in', left=True)
        plt.legend(fontsize=16, loc='upper right', ncol=1)
        plt.tick_params(pad=8)
        plt.xlim(Ixmin, Ixmax)
        plt.xticks(xticks_time)

    if(type_graph == 'I1'):

        I_Multi_Sub1()
        plt.title("Current     Short[1\'], [2\'], [3\'], [4\']", fontsize=fontsize)
        plt.savefig("%s/%s.png" %(savedir_I_multi, dict[str_title]), bbox_inches='tight')
        plt.close()

    elif(type_graph == 'I2'):

        I_Multi_Sub1()
        plt.title("Current     Short[1\'][2\'][3\'][4\']", fontsize=fontsize)
        plt.savefig("%s/%s.png" %(savedir_I_multi, dict[str_title]), bbox_inches='tight')
        plt.close()

    elif(type_graph == 'I3'):

        I_Multi_Sub1()
        plt.title("Current     Short[1\'2\'3\'4\']", fontsize=fontsize)
        plt.savefig("%s/%s.png" %(savedir_I_multi, dict[str_title]), bbox_inches='tight')
        plt.close()

    else:()

#=======================================================
#  Definition of I multiplot (2)
#=======================================================

def I_Multi2(signal1, type_graph, str_title):

    global time

    dict = {'I':'Current  [kA]', 'Name':'全て短絡させた場合の総電流'}

    savedir_I_multi = savedir + "/" + "Current_Multiplot"
    if not(os.access(savedir_I_multi,os.F_OK)):
        os.mkdir(savedir_I_multi)

    if(type_graph == 'I'):

        fig,ax = plt.subplots(figsize=Figsize)
        ax.plot(time, signal1, lw=2, label=Operation6)
        ax.set_xlabel("Time [$\mu$s]", labelpad=8, fontsize=fontsize)
        ax.set_ylabel("%s" %(dict[type_graph]), labelpad=15, fontsize=fontsize)
        plt.title("Current     Short[1\'2\'3\'4\']", fontsize=fontsize)
        plt.hlines([0], Ixmin, Ixmax, "black", linestyle=":", lw=2)
        plt.gca().xaxis.set_tick_params(which='both', direction='in', bottom=True)
        plt.gca().yaxis.set_tick_params(which='both', direction='in', left=True)
        plt.legend(fontsize=16, loc='upper right', ncol=1)
        plt.tick_params(pad=8)
        plt.xlim(Ixmin, Ixmax)
        plt.xticks(xticks_time)
        plt.savefig("%s/%s.png" %(savedir_I_multi, dict[str_title]), bbox_inches='tight')
        plt.close()

    else:()

#=======================================================
#  Definition of I errorplot (1)
#=======================================================

def I_Error1(signal1, signal2, signal3, signal4, signal5, signal6, signal7, signal8, type_graph, str_title):

    global time

    dict = {'I1':'Current  [kA]', 'I2':'Current  [kA]', 'I3':'Current  [kA]',
            'Name1':'1組のみ短絡した場合', 'Name2':'4組を同時短絡した場合', 'Name3':'全て短絡させた場合'}

    savedir_I_error = savedir + "/" + "Current_Errorplot"
    if not(os.access(savedir_I_error,os.F_OK)):
        os.mkdir(savedir_I_error)

    def I_Error_Sub1():

        fig,ax = plt.subplots(figsize=Figsize)
        ax.errorbar(time, signal1, yerr=signal2, elinewidth=0.5, lw=2, label=Operation2)
        ax.errorbar(time, signal3, yerr=signal4, elinewidth=0.5, lw=2, label=Operation3)
        ax.errorbar(time, signal5, yerr=signal6, elinewidth=0.5, lw=2, label=Operation4)
        ax.errorbar(time, signal7, yerr=signal8, elinewidth=0.5, lw=2, label=Operation5)
        ax.set_xlabel("Time [$\mu$s]", labelpad=8, fontsize=fontsize)
        ax.set_ylabel("%s" %(dict[type_graph]), labelpad=15, fontsize=fontsize)
        plt.hlines([0], Ixmin, Ixmax, "black", linestyle=":", lw=2)
        plt.gca().xaxis.set_tick_params(which='both', direction='in', bottom=True)
        plt.gca().yaxis.set_tick_params(which='both', direction='in', left=True)
        plt.legend(fontsize=16, loc='upper right', ncol=1)
        plt.tick_params(pad=8)
        plt.xlim(Ixmin, Ixmax)
        plt.xticks(xticks_time)

    if(type_graph == 'I1'):

        I_Error_Sub1()
        plt.title("Current     Short[1\'], [2\'], [3\'], [4\']", fontsize=fontsize)
        plt.savefig("%s/%s.png" %(savedir_I_error, dict[str_title]), bbox_inches='tight')
        plt.close()

    elif(type_graph == 'I2'):

        I_Error_Sub1()
        plt.title("Current     Short[1\'][2\'][3\'][4\']", fontsize=fontsize)
        plt.savefig("%s/%s.png" %(savedir_I_error, dict[str_title]), bbox_inches='tight')
        plt.close()

    elif(type_graph == 'I3'):

        I_Error_Sub1()
        plt.title("Current     Short[1\'2\'3\'4\']", fontsize=fontsize)
        plt.savefig("%s/%s.png" %(savedir_I_error, dict[str_title]), bbox_inches='tight')
        plt.close()

    else:()

#=======================================================
#  Definition of I errorplot (2)
#=======================================================

def I_Error2(signal1, signal2, type_graph, str_title):

    global time

    dict = {'I':'Current  [kA]', 'Name':'電極を全て短絡させた場合の総電流'}

    savedir_I_error = savedir + "/" + "Current_Errorplot"
    if not(os.access(savedir_I_error,os.F_OK)):
        os.mkdir(savedir_I_error)

    if(type_graph == 'I'):

        fig,ax = plt.subplots(figsize=Figsize)
        ax.errorbar(time, signal1, yerr=signal2, elinewidth=0.5, lw=2, label=Operation6)
        ax.set_xlabel("Time [$\mu$s]", labelpad=8, fontsize=fontsize)
        ax.set_ylabel("%s" %(dict[type_graph]), labelpad=15, fontsize=fontsize)
        plt.title("Current     Short[1\'2\'3\'4\']", fontsize=fontsize)
        plt.hlines([0], Ixmin, Ixmax, "black", linestyle=":", lw=2)
        plt.gca().xaxis.set_tick_params(which='both', direction='in', bottom=True)
        plt.gca().yaxis.set_tick_params(which='both', direction='in', left=True)
        plt.legend(fontsize=16, loc='upper right', ncol=1)
        plt.tick_params(pad=8)
        plt.xlim(Ixmin, Ixmax)
        plt.xticks(xticks_time)
        plt.savefig("%s/%s.png" %(savedir_I_error, dict[str_title]), bbox_inches='tight')
        plt.close()

    else:()

#=======================================================
#  Return of current
#=======================================================

I = np.zeros((Operation_Class, Operation_Shot, ICH, sampleC))
for i in range(Operation_Class):
    for j in range(Operation_Shot):
        for shotname in [Shot[i][j]]:
            (f1, sample_period, full_range1, precision1, init_time, fin_time, dt, time) = File_Data(shotname)
            I[i][j] = Current()
            I = I.reshape([Operation_Class, Operation_Shot, ICH, sampleC])

#=======================================================
#  Return of I multiplot (1)
#=======================================================

    I_Multi1(np.array(I[1][0][0]), np.array(I[2][0][1]), -np.array(I[3][0][2]), np.array(I[4][0][3]), "I1", 'Name1')
    I_Multi1(np.array(I[5][0][0]), np.array(I[5][0][1]), -np.array(I[5][0][2]), np.array(I[5][0][3]), "I2", 'Name2')
    I_Multi1(np.array(I[6][0][0]), np.array(I[6][0][1]), -np.array(I[6][0][2]), np.array(I[6][0][3]), "I3", 'Name3')

#=======================================================
#  Return of I multiplot (2)
#=======================================================

    I_Multi2(np.array(I[6][0][0]) + np.array(I[6][0][1]) - np.array(I[6][0][2]) + np.array(I[6][0][3]), "I", 'Name')

#=======================================================
#  Return of I errorplot (1)
#=======================================================

    I_ave, I_err = [], []
    for i in range(Operation_Class): 
        I_ave_sub, I_err_sub = [], []
        for j in range(ICH):
            I_ave_sub_sub, I_err_sub_sub = [], []            
            for k in range(sampleC):
                I_max, I_min = max([I[i][0][j][k], I[i][1][j][k]]), min([I[i][0][j][k], I[i][1][j][k]])
                I_average, I_error = I_max/2 + I_min/2, I_max - I_min
                I_ave_sub_sub.append(I_average), I_err_sub_sub.append(I_error)
            I_ave_sub.append(I_ave_sub_sub), I_err_sub.append(I_err_sub_sub)
        I_ave.append(I_ave_sub), I_err.append(I_err_sub)

    I_Error1(np.array(I_ave[1][0]), I_err[1][0], np.array(I_ave[2][1]), I_err[2][1], -np.array(I_ave[3][2]), I_err[3][2], np.array(I_ave[4][3]), I_err[4][3], "I1", 'Name1')
    I_Error1(np.array(I_ave[5][0]), I_err[5][0], np.array(I_ave[5][1]), I_err[5][1], -np.array(I_ave[5][2]), I_err[5][2], np.array(I_ave[5][3]), I_err[5][3], "I2", 'Name2')
    I_Error1(np.array(I_ave[6][0]), I_err[6][0], np.array(I_ave[6][1]), I_err[6][1], -np.array(I_ave[6][2]), I_err[6][2], np.array(I_ave[6][3]), I_err[6][3], "I3", 'Name3')

#=======================================================
#  Return of I errorplot (2)
#=======================================================

    I_ave, I_err = [], []
    for i in range(sampleC):
        I_max = max([np.array(I[6][0][0][i]) + np.array(I[6][0][1][i]) - np.array(I[6][0][2][i]) + np.array(I[6][0][3][i]), np.array(I[6][1][0][i]) + np.array(I[6][1][1][i]) - np.array(I[6][1][2][i]) + np.array(I[6][1][3][i])])
        I_min = min([np.array(I[6][0][0][i]) + np.array(I[6][0][1][i]) - np.array(I[6][0][2][i]) + np.array(I[6][0][3][i]), np.array(I[6][1][0][i]) + np.array(I[6][1][1][i]) - np.array(I[6][1][2][i]) + np.array(I[6][1][3][i])])
        I_average, I_error = I_max/2 + I_min/2, I_max - I_min
        I_ave.append(I_average), I_err.append(I_error)

    I_Error2(np.array(I_ave), I_err, "I", 'Name')

#=======================================================