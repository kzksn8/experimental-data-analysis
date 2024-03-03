#=======================================================
#  Import gpa file
#=======================================================

(Operation_Class, Operation_Shot) = (7, 4)

Operation1 = 'Open'
Shotnumber1 = ['#201028014', '#201028007']
(Shot1_1_in, Shot1_1_out) = ('Data_gpa/201028014', 'Data_gpa/201028007')
(Shot1_2_in, Shot1_2_out) = ('Data_gpa/201028011', 'Data_gpa/201028005')

Operation2 = 'Short [1\']'
Shotnumber2 = ['#201103007', '#201103014']
(Shot2_1_in, Shot2_1_out) = ('Data_gpa/201103007', 'Data_gpa/201103014')
(Shot2_2_in, Shot2_2_out) = ('Data_gpa/201103005', 'Data_gpa/201103019')

Operation3 = 'Short [2\']'
Shotnumber3 = ['#201105011', '#201105004']
(Shot3_1_in, Shot3_1_out) = ('Data_gpa/201105011', 'Data_gpa/201105004')
(Shot3_2_in, Shot3_2_out) = ('Data_gpa/201105010', 'Data_gpa/201105002')

Operation4 = 'Short [3\']'
Shotnumber4 = ['#201216015', '#201216011']
(Shot4_1_in, Shot4_1_out) = ('Data_gpa/201216015', 'Data_gpa/201216011')
(Shot4_2_in, Shot4_2_out) = ('Data_gpa/201216017', 'Data_gpa/201216009')

Operation5 = 'Short [4\']'
Shotnumber5 = ['#201030009', '#201030006']
(Shot5_1_in, Shot5_1_out) = ('Data_gpa/201030009', 'Data_gpa/201030006')
(Shot5_2_in, Shot5_2_out) = ('Data_gpa/201030011', 'Data_gpa/201030005')

Operation6 = 'Short [1\'][2\'][3\'][4\']'
Shotnumber6 = ['#201029003', '#201029011']
(Shot6_1_in, Shot6_1_out) = ('Data_gpa/201029003', 'Data_gpa/201029011')
(Shot6_2_in, Shot6_2_out) = ('Data_gpa/201029005', 'Data_gpa/201029013')

Operation7 = 'Short [1\'2\'3\'4\']'
Shotnumber7 = ['#201215002', '#201215010']
(Shot7_1_in, Shot7_1_out) = ('Data_gpa/201215002', 'Data_gpa/201215010')
(Shot7_2_in, Shot7_2_out) = ('Data_gpa/201215009', 'Data_gpa/201215016')

Operation  = [Operation1, Operation2, Operation3, Operation4, Operation5, Operation6, Operation7]

Shot = [Shot1_1_out, Shot2_1_out, Shot3_1_in, Shot4_1_in, Shot5_1_out, Shot6_1_out, Shot7_1_in]
Shot_in = [Shot1_1_in, Shot2_1_in, Shot3_1_in, Shot4_1_in, Shot5_1_in, Shot6_1_in, Shot7_1_in]
Shot_out = [Shot1_1_out, Shot2_1_out, Shot3_1_out, Shot4_1_out, Shot5_1_out, Shot6_1_out, Shot7_1_out]

Shotnumber = [Shotnumber1, Shotnumber2, Shotnumber3, Shotnumber4, Shotnumber5, Shotnumber6, Shotnumber7]

#=======================================================
#  Import module
#=======================================================

import os
import numpy as np
import scipy.fftpack as sf
from scipy.io import netcdf
from scipy import integrate
import matplotlib.pyplot as plt
from scipy.special import ellipk
from scipy.special import ellipe
from scipy.interpolate import interp1d

#=======================================================
#  Parameters of Graph
#=======================================================

Fontsize = 18
LP_freq = 200

FigsizeM = (9, 3.5)
FigsizeL = (12, 5.5)

(xmin, xmax) = (9425, 9575)

(Eparaymin,   Eparaymax)   = (-180, 140)
(EparaEzymin, EparaEzymax) = (-200, 140)
(EparaEtymin, EparaEtymax) = (-140, 180)

xticks_time = np.arange(9425,9600,25)

Xp_range = 0.0003

color_type = 200
cmap_type_Epara    = 'jet'
cmap_type_EparaEz  = 'jet_r'
cmap_type_EparaEt  = 'jet'

#=======================================================
#  Formatting of graph
#=======================================================

savedir = os.getcwd() + "/" + "Epara_Time"
if not(os.access(savedir,os.F_OK)):
    os.mkdir(savedir)

plt.rcParams.update({'mathtext.default':'default', 'mathtext.fontset':'stix', 'font.family':'Arial', 'font.size':Fontsize})

#=======================================================
#  Parameter of oscillo scope
#=======================================================

sample = 10000
(sampleA, sampleB) = (8800, 9800)
sampleC = sampleB - sampleA

EparaCH = 17
Posi_Epara = [160,180,200,220,240,260,280,300,320,340,360,380,400,420,440,460,480]

EzCH = 16
flag = range(28,44)
Posi_Ez = [140,160,180,200,220,240,260,280,300,320,340,360,380,400,420,440,460,480,500]

EzCH_Dupe = 3
Posi_Ezdupe = [300,320,340]

(EzCH_in, EzCH_out) = (8,11)

ch_name = []
for i in range(EzCH):
    Oscillo = 'dl716_1_%s' % (i+1)
    ch_name.append(Oscillo)

BzCH,EtCH = 16,15
flag_bzll = range(44,60)
inv_bzll  = [1.,1.,-1.,-1.,-1.,-1.,-1.,-1.,1.,-1.,-1.,-1.,1.,-1.,-1.,1.]
Posi_bzll = [110,150,200,250,270,290,320,340,380,400,420,440,450,460,470,480]
Posi_Etll = [150,200,250,270,290,320,340,380,400,420,440,450,460,470,480]

name_bzll = []
for i in range(BzCH):
    Oscillo_2 = 'dl716_2_%s' % (i+1)
    name_bzll.append(Oscillo_2)

y_relative = [0.985100312075366, 1.0264444410082347, 1.014178101514291, 0.9774372245292574, 
              0.9778123791745478, 1.0013997471407676, 1.007161826170335, 1.013467275771488, 
              0.983971382382668, 1.0117771470083035, 1.027142963617728, 0.979958325462205, 
              1.0197870575051997, 0.9944148682778086, 0.9639819445113512, 1.021418357620166]

m0 = 4 * np.pi * 1e-7
NS = 300 * 0.0025**2 * np.pi
cor_helm = 1.333
A = 1 / NS

(valnmin, valnmax) = (9500, 9550)
dval = int(valnmax-valnmin)

xTime = np.arange(sampleA, sampleB, 1)
xSpBz = (max(Posi_bzll)-min(Posi_bzll))*10
xSpEt = (max(Posi_Etll)-min(Posi_Etll))*10

#=======================================================
#  Parameter of EF coil
#=======================================================

(V, turn, EFnum) = (120, 200, 6)
(Z_EF, R_EF, z_probe) = (1.05, 855 * 1e-3, 0)
I = - (0.849 * (1.19 * V - 5.32) - 5.56)

EF = []
for i in range(BzCH):
    alpha_u = np.sqrt((R_EF + Posi_bzll[i]*1e-3)**2 + (z_probe - Z_EF)**2)
    alpha_l = np.sqrt((R_EF + Posi_bzll[i]*1e-3)**2 + (z_probe + Z_EF)**2)
    beta_u = (R_EF - Posi_bzll[i]*1e-3)**2 + (z_probe - Z_EF)**2
    beta_l = (R_EF - Posi_bzll[i]*1e-3)**2 + (z_probe + Z_EF)**2
    k2_u = (4 * R_EF * Posi_bzll[i]*1e-3) / alpha_u**2
    k2_l = (4 * R_EF * Posi_bzll[i]*1e-3) / alpha_l**2
    K_u, K_l = ellipk(k2_u), ellipk(k2_l)
    E_u, E_l = ellipe(k2_u), ellipe(k2_l)
    Bz_upper = ((m0 * turn * EFnum * I/EFnum) / (2 * np.pi)) * (1 / alpha_u) * (K_u + ((R_EF**2 - (Posi_bzll[i]*1e-3)**2) - (z_probe - Z_EF)**2) / beta_u * E_u)
    Bz_lower = ((m0 * turn * EFnum * I/EFnum) / (2 * np.pi)) * (1 / alpha_l) * (K_l + ((R_EF**2 - (Posi_bzll[i]*1e-3)**2) - (z_probe + Z_EF)**2) / beta_l * E_l)
    nEF = Bz_upper + Bz_lower
    EF.append(nEF)

#=======================================================
#  Definition of gpa file data
#=======================================================

def File_Data(shotname):

    f1 = netcdf.netcdf_file('%s.gpa' % (shotname))
    f2 = netcdf.netcdf_file('Data_gpa/TF(2.7kV).gpa')
    f3 = netcdf.netcdf_file('Data_gpa/PF3(2kV).gpa')

    sample_period = getattr(f1,'sample_period')
    precision1, precision2, precision3  = getattr(f1,'precision'), getattr(f2,'precision'), getattr(f3,'precision')
    full_range1, full_range2, full_range3 = getattr(f1,'full_range'), getattr(f2,'full_range'), getattr(f3,'full_range')

    init_time = 0 #[sec]
    fin_time = init_time + sample_period[flag_bzll[0]] * sample #[sec]
    dt = sample_period[flag_bzll[0]]
    time = np.arange(init_time, fin_time, dt) #[sec]
    time = time * 1e+6 #[usec]
    time_PF3 = time[0:sampleA]
    time = time[sampleA:sampleB]

    return(f1, f2, f3, sample_period, full_range1, full_range2, full_range3, precision1, precision2, precision3, init_time, fin_time, dt, time, time_PF3)

#=======================================================
#  Definition of spline interpolation
#=======================================================

def Spline(in_x, in_y):
    
    out_x = np.linspace(np.min(in_x), np.max(in_x), (np.max(in_x)-np.min(in_x))*10)
    func_spline = interp1d(in_x, in_y, kind='cubic')
    out_y = func_spline(out_x)

    return out_x, out_y

#=======================================================
#  Definition of geting nearest value
#=======================================================

def Get_Nearest(list, num):
    
    idx = np.abs(np.asarray(list) - num).argmin()
    
    return idx

#=======================================================
#  Definition of single probe
#=======================================================

def Single_Probe():

    global f1, ch_name, flag, precision1

    Ez = []
    for i in range(EzCH):
        ez = f1.variables[ch_name[i]][:] * full_range1[flag[i]] / precision1[flag[i]] * 200 / 50
        ez = ez * np.sqrt(2)
        ez = ez[sampleA:sampleB]
        ez = sf.fft(ez)
        freq = sf.fftfreq(int(sampleC), dt)
        ez[abs(freq) > LP_freq*1e3] = 0
        ez = sf.ifft(ez)
        Ez.append(np.real(ez))

    Ez = -np.array(Ez)

    return Ez

#=======================================================
#  Definition of magnetic probe
#=======================================================

def Magnetic_Probe():

    global f1, f2, f3, inv_bzll, name_bzll, flag_bzll, precision1, precision2, precision3, full_range1, full_range2, full_range3, time, time_PF3, dt

    Val_bzll, valfreq, Bzll = [], [], []
    
    for i in range(BzCH):

        val_bzll = y_relative[i] * inv_bzll[i] * f1.variables[name_bzll[i]][:] * full_range1[flag_bzll[i]] / precision1[flag_bzll[i]]
        TF_bzll  = y_relative[i] * inv_bzll[i] * f2.variables[name_bzll[i]][:] * full_range2[flag_bzll[i]] / precision2[flag_bzll[i]]
        PF3_bzll = y_relative[i] * inv_bzll[i] * f3.variables[name_bzll[i]][:] * full_range3[flag_bzll[i]] / precision3[flag_bzll[i]]
        val_bzll = val_bzll - TF_bzll
        ofs_bzll = np.average(val_bzll[0:3800])
        val_bzll = val_bzll - ofs_bzll
        val_bzll = val_bzll[sampleA:sampleB]
        Val_bzll.append(val_bzll)
        
        valfourier = val_bzll[valnmin-sampleA:valnmax-sampleA]
        valfourier = valfourier * np.hanning(dval)
        valfourierfft = sf.fft(valfourier)
        valfourierabs = abs(valfourierfft)
        valfreq.append(valfourierabs)

        val_bzll_fft = sf.fft(val_bzll)
        freq_bzll = sf.fftfreq(int(len(val_bzll)), dt)
        val_bzll_fft[abs(freq_bzll) > LP_freq*1e3] = 0
        val_bzll_ifft = sf.ifft(val_bzll_fft)

        int_bzll= integrate.cumtrapz(val_bzll_ifft, time*1e-6, initial=0)
        int_bzll = -int_bzll * A * cor_helm

        PF3_bzll = PF3_bzll[0:sampleA]
        int_PF3_bzll = integrate.cumtrapz(PF3_bzll, time_PF3*1e-6, initial=0)
        int_PF3_bzll = -int_PF3_bzll * A * cor_helm

        int_bzll = int_bzll + EF[i] + int_PF3_bzll[sampleA-1]*0.202
        Bzll.append(np.real(int_bzll))
    
    TBz = np.array(Bzll).T

    SpPosiBz, SpTBz = [], []
    for i in range(sampleC):
        spPosiBz, spTBz = Spline(Posi_bzll, TBz[int(i)])
        SpPosiBz.append(spPosiBz), SpTBz.append(spTBz)

    psil = []
    for i in range(BzCH):
        if i==0:
            npsi = np.array([0]*sampleC)
        else:
            npsi = 2*np.pi*((Bzll[i]+Bzll[i-1])/2)*Posi_bzll[i]*1e-3*(Posi_bzll[i]-Posi_bzll[i-1])*1e-3
        psil.append(npsi)

    psi = np.array(psil)

    addpsi = np.zeros((BzCH, sampleC))
    for i in range(BzCH):
        if i==0:
            addpsi[0] = psi[0]
        else:
            addpsi[i] = addpsi[i-1] + psi[i]
        addpsi = addpsi.reshape([BzCH, sampleC])

    dpsi = np.zeros((BzCH, sampleC))
    for i in range(BzCH):
        for j in range(sampleC):
            if j==0:
                dpsi[i][0] = np.array([0])
            else:
                dpsi[i][j] = addpsi[i][j] - addpsi[i][j-1]
            dpsi = dpsi.reshape([BzCH, sampleC])

    Etll = np.zeros((EtCH, sampleC))
    for i in range(EtCH):
        Etll[i] = ((-1./(2*np.pi*Posi_Etll[i]*1e-3))*(dpsi[i+1]/dt)).real
        Etll = Etll.reshape([EtCH, sampleC])

    TEt = Etll.T

    SpPosiEt, SpTEt = [], []
    for i in range(sampleC):
        spPosiEt, spTEt = Spline(Posi_Etll, TEt[int(i)])
        SpPosiEt.append(spPosiEt), SpTEt.append(spTEt)

    return (SpTBz, SpTEt)

#=======================================================
#  Definition of Bt data
#=======================================================

def Bt_dat(Shotname):

    Bt_Operation = []
    for i in range(EparaCH):
        read = open('Data_dat/Bt_' + Shotname + '/Bt_' + Shotname + '_R=%s0.dat' % (16+i*2), 'r')
        for j in range(4):
            data = read.readline()
        bt_Operation = []
        for j in range(sampleC):
            data = read.readline()
            bt_Operation.append(float(data.split()[1]))
        Bt_Operation.append(bt_Operation)
    
    return Bt_Operation

#=======================================================
#  Definition of Epara multiplot (1)
#=======================================================

def Epara_Multi1(signal1, signal2, signal3, signal4, signal5, signal6, signal7, signal8, signal9, signal10, signal11, signal12, signal13, signal14, signal15, signal16, signal17, type_graph, str_title, shotnumber1, shotnumber2):    

    def Epara_Multi_Sub1(name1, name2, savedict):

        global time

        dict = {'Epara': '$E_{//}$  [V/m]', 'EparaEz': '$E_{//,z}$  [V/m]', 'EparaEt': '$E_{//,t}$  [V/m]', 'Operation1': Operation[0], 'Operation2': Operation[1], 'Operation3': Operation[2], 'Operation4': Operation[3], 'Operation5': Operation[4], 'Operation6': Operation[5], 'Operation7': Operation[6],
                'Shotnumber1-1': Shotnumber[0][0], 'Shotnumber1-2': Shotnumber[1][0], 'Shotnumber1-3': Shotnumber[2][0], 'Shotnumber1-4': Shotnumber[3][0], 'Shotnumber1-5': Shotnumber[4][0], 'Shotnumber1-6': Shotnumber[5][0], 'Shotnumber1-7': Shotnumber[6][0],
                'Shotnumber2-1': Shotnumber[0][1], 'Shotnumber2-2': Shotnumber[1][1], 'Shotnumber2-3': Shotnumber[2][1], 'Shotnumber2-4': Shotnumber[3][1], 'Shotnumber2-5': Shotnumber[4][1], 'Shotnumber2-6': Shotnumber[5][1], 'Shotnumber2-7': Shotnumber[6][1]}

        fig,ax = plt.subplots(figsize=FigsizeL)
        ax.plot(time, signal1,  label = 'R = %s  [mm]' % (Posi_Epara[0]))
        ax.plot(time, signal2,  label = 'R = %s  [mm]' % (Posi_Epara[1]))
        ax.plot(time, signal3,  label = 'R = %s  [mm]' % (Posi_Epara[2]))
        ax.plot(time, signal4,  label = 'R = %s  [mm]' % (Posi_Epara[3]))
        ax.plot(time, signal5,  label = 'R = %s  [mm]' % (Posi_Epara[4]))
        ax.plot(time, signal6,  label = 'R = %s  [mm]' % (Posi_Epara[5]))
        ax.plot(time, signal7,  label = 'R = %s  [mm]' % (Posi_Epara[6]))
        ax.plot(time, signal8,  label = 'R = %s  [mm]' % (Posi_Epara[7]))
        ax.plot(time, signal9,  label = 'R = %s  [mm]' % (Posi_Epara[8]))
        ax.plot(time, signal10, label = 'R = %s  [mm]' % (Posi_Epara[9]))
        ax.plot(time, signal11, label = 'R = %s  [mm]' % (Posi_Epara[10]))
        ax.plot(time, signal12, label = 'R = %s  [mm]' % (Posi_Epara[11]))
        ax.plot(time, signal13, label = 'R = %s  [mm]' % (Posi_Epara[12]))
        ax.plot(time, signal14, label = 'R = %s  [mm]' % (Posi_Epara[13]))
        ax.plot(time, signal15, label = 'R = %s  [mm]' % (Posi_Epara[14]))
        ax.plot(time, signal16, label = 'R = %s  [mm]' % (Posi_Epara[15]))
        ax.plot(time, signal17, label = 'R = %s  [mm]' % (Posi_Epara[16]))
        ax.set_xlabel("Time [$\mu$s]", labelpad=10, fontsize=Fontsize)
        ax.set_ylabel("%s" %(dict[type_graph]), labelpad=12, fontsize=Fontsize)
        plt.title("%s   %s    %s  %s" % (name1, dict[str_title], dict[shotnumber1], dict[shotnumber2]), fontsize=Fontsize)
        plt.legend(bbox_to_anchor=(1, 1.03), loc='upper left', labelspacing=0.4, fontsize=13)
        plt.hlines([0], xmin, xmax, "black", linestyle=":", lw=2)
        plt.gca().xaxis.set_tick_params(which='both', direction='in', bottom=True)
        plt.gca().yaxis.set_tick_params(which='both', direction='in', left=True)
        plt.tick_params(pad=8)
        plt.xlim(xmin, xmax)
        plt.xticks(xticks_time)
        plt.savefig(savedict + "/%s_%s.png" %(name2, dict[str_title]), bbox_inches='tight')
        plt.close()

    if(type_graph == 'Epara'):

        savedir_Epara_multi1 = savedir + "/" + "Epara_Multiplot1"
        if not(os.access(savedir_Epara_multi1,os.F_OK)):
            os.mkdir(savedir_Epara_multi1)

        Epara_Multi_Sub1("$E_{//}$", "Epara", savedir_Epara_multi1)

    elif(type_graph == 'EparaEz'):

        savedir_EparaEz_multi1 = savedir + "/" + "EparaEz_Multiplot1"
        if not(os.access(savedir_EparaEz_multi1,os.F_OK)):
            os.mkdir(savedir_EparaEz_multi1)

        Epara_Multi_Sub1("$E_{//,z}$", "EparaEz", savedir_EparaEz_multi1)

    elif(type_graph == 'EparaEt'):

        savedir_EparaEt_multi1 = savedir + "/" + "EparaEt_Multiplot1"
        if not(os.access(savedir_EparaEt_multi1,os.F_OK)):
            os.mkdir(savedir_EparaEt_multi1)     

        Epara_Multi_Sub1("$E_{//,t}$", "EparaEt", savedir_EparaEt_multi1)

    else:()

#=======================================================
#  Definition of Epara multiplot (2)
#=======================================================

def Epara_Multi2(signal1, signal2, signal3, signal4, signal5, signal6, signal7, type_graph, str_title):

    def Epara_Multi_Sub2(name1, name2, savedict):

        global time
    
        dict = {'Epara': '$E_{//}$  [V/m]', 'EparaEz': '$E_{//,z}$  [V/m]', 'EparaEt': '$E_{//,t}$  [V/m]',
                '1':'R = 140  [mm]', '2':'R = 160  [mm]', '3':'R = 180  [mm]', '4':'R = 200  [mm]', '5':'R = 220  [mm]', '6':'R = 240  [mm]', '7':'R = 260  [mm]', '8':'R = 280  [mm]', '9':'R = 300  [mm]',
                '10':'R = 320  [mm]', '11':'R = 340  [mm]', '12':'R = 360  [mm]', '13':'R = 380  [mm]', '14':'R = 400  [mm]', '15':'R = 420  [mm]', '16':'R = 440  [mm]', '17':'R = 460  [mm]', '18':'R = 480  [mm]', '19':'R = 500  [mm]'}

        fig,ax = plt.subplots(figsize=FigsizeM)
        ax.plot(time, signal1, label = Operation[0])
        ax.plot(time, signal2, label = Operation[1])
        ax.plot(time, signal3, label = Operation[2])
        ax.plot(time, signal4, label = Operation[3])
        ax.plot(time, signal5, label = Operation[4])
        ax.plot(time, signal6, label = Operation[5])
        ax.plot(time, signal7, label = Operation[6])
        ax.set_xlabel("Time [$\mu$s]", labelpad=10, fontsize=Fontsize)
        ax.set_ylabel("%s" %(dict[type_graph]), labelpad=12, fontsize=Fontsize)
        plt.title("%s    %s" % (name1, dict[str_title]), fontsize=Fontsize)
        plt.hlines([0], xmin, xmax, "black", linestyle=":", lw=2)
        plt.legend(bbox_to_anchor=(1, 1.00), loc='upper left', labelspacing=0.5, fontsize=14)
        plt.gca().xaxis.set_tick_params(which='both', direction='in', bottom=True)
        plt.gca().yaxis.set_tick_params(which='both', direction='in', left=True)
        plt.tick_params(pad=8)
        plt.xlim(xmin, xmax)
        plt.xticks(xticks_time)
        plt.savefig(savedict + "/%s_%s.png" %(name2, dict[str_title]), bbox_inches='tight')
        plt.close()

    if(type_graph == 'Epara'):

        savedir_Epara_multi2 = savedir + "/" + "Epara_Multiplot2"
        if not(os.access(savedir_Epara_multi2,os.F_OK)):
            os.mkdir(savedir_Epara_multi2)

        Epara_Multi_Sub2("$E_{//}$", "Epara", savedir_Epara_multi2)

    elif(type_graph == 'EparaEz'):

        savedir_EparaEz_multi2 = savedir + "/" + "EparaEz_Multiplot2"
        if not(os.access(savedir_EparaEz_multi2,os.F_OK)):
            os.mkdir(savedir_EparaEz_multi2)

        Epara_Multi_Sub2("$E_{//,z}$", "EparaEz", savedir_EparaEz_multi2)

    elif(type_graph == 'EparaEt'):

        savedir_EparaEt_multi2 = savedir + "/" + "EparaEt_Multiplot2"
        if not(os.access(savedir_EparaEt_multi2,os.F_OK)):
            os.mkdir(savedir_EparaEt_multi2)     

        Epara_Multi_Sub2("$E_{//,t}$", "EparaEt", savedir_EparaEt_multi2)

    else:()

#=======================================================
#  Return of single probe
#=======================================================

Ez_in, Ez_out = np.zeros((Operation_Class, EzCH, sampleC)), np.zeros((Operation_Class, EzCH, sampleC))
for i in range(Operation_Class):
    for shotname in [Shot_in[i]]:
        (f1, f2, f3, sample_period, full_range1, full_range2, full_range3, precision1, precision2, precision3, init_time, fin_time, dt, time, time_PF3) = File_Data(shotname)
        Ez_in[i] = Single_Probe()
        Ez_in = Ez_in.reshape([Operation_Class, EzCH, sampleC])
    for shotname in [Shot_out[i]]:
        (f1, f2, f3, sample_period, full_range1, full_range2, full_range3, precision1, precision2, precision3, init_time, fin_time, dt, time, time_PF3) = File_Data(shotname)
        Ez_out[i] = Single_Probe()
        Ez_out = Ez_out.reshape([Operation_Class, EzCH, sampleC])

    Ez_list = []
    for i in range(Operation_Class):
        ez = [Ez_in[i][1], Ez_in[i][2], Ez_in[i][3], Ez_in[i][4], Ez_in[i][5], Ez_in[i][6], Ez_in[i][7], Ez_out[i][0], Ez_out[i][1], Ez_out[i][2], Ez_out[i][3], Ez_out[i][4], Ez_out[i][5], Ez_out[i][6], Ez_out[i][7], Ez_out[i][8], Ez_out[i][9]]
        Ez_list.append(ez)

    Ez = np.array(Ez_list)

#=======================================================
#  Return of magnetic probe
#=======================================================

(SpTBz, SpTEt) = (np.zeros((Operation_Class, sampleC, xSpBz)), np.zeros((Operation_Class, sampleC, xSpEt)))
for i in range(Operation_Class):
    for shotname in [Shot[i]]:
        (f1, f2, f3, sample_period, full_range1, full_range2, full_range3, precision1, precision2, precision3, init_time, fin_time, dt, time, time_PF3) = File_Data(shotname)
        (SpTBz[i], SpTEt[i]) = Magnetic_Probe()
        (SpTBz, SpTEt) = (SpTBz.reshape([Operation_Class, sampleC, xSpBz]), SpTEt.reshape([Operation_Class, sampleC, xSpEt]))

    Bz_Spline, Et_Spline = [], []
    for i in range(Operation_Class):
        Bz_spline, Et_spline = SpTBz[i].T, SpTEt[i].T
        Bz_Spline.append(Bz_spline), Et_Spline.append(Et_spline)

    Bz_list, Et_list = [], []
    for i in range(Operation_Class):
        Bz_list_sub, Et_list_sub = [], []
        for j in range(EparaCH):
            bz, et = Bz_Spline[i][499+200*j], Et_Spline[i][99+200*j]
            Bz_list_sub.append(bz), Et_list_sub.append(et)
        Bz_list.append(Bz_list_sub), Et_list.append(Et_list_sub)

    Bz, Et = np.array(Bz_list), np.array(Et_list)

#=======================================================
#  Return of Bt data
#=======================================================
    
    Bt_Open = Bt_dat(Shotnumber[0][1])
    Bt_Short1 = Bt_dat(Shotnumber[1][1])
    Bt_Short2 = Bt_dat(Shotnumber[2][1])
    Bt_Short3 = Bt_dat(Shotnumber[3][0])
    Bt_Short4 = Bt_dat(Shotnumber[4][1])
    Bt_Short1234_1 = Bt_dat(Shotnumber[5][1])
    Bt_Short1234_2 = Bt_dat(Shotnumber[6][0])
    
    Bt_list = [Bt_Open, Bt_Short1, Bt_Short2, Bt_Short3, Bt_Short4, Bt_Short1234_1, Bt_Short1234_2]

    Bt = np.array(Bt_list)

#=======================================================
#  Calculation of Epara
#=======================================================

    Epara, EparaEz, EparaEt = [], [], []
    for i in range(Operation_Class):
        Epara_sub, EparaEz_sub, EparaEt_sub = [], [], []
        for j in range(EparaCH):
            epara   = ((np.sqrt((Bz[i][j])**2 + (Bt[i][j])**2))**(-1)) * (Ez[i][j] * 1000 * Bz[i][j] + Et[i][j] * Bt[i][j])
            eparaEz = ((np.sqrt((Bz[i][j])**2 + (Bt[i][j])**2))**(-1)) * (Ez[i][j] * 1000 * Bz[i][j])
            eparaEt = ((np.sqrt((Bz[i][j])**2 + (Bt[i][j])**2))**(-1)) * (Et[i][j] * Bt[i][j])
            Epara_sub.append(epara), EparaEz_sub.append(eparaEz), EparaEt_sub.append(eparaEt)
        Epara.append(Epara_sub), EparaEz.append(EparaEz_sub), EparaEt.append(EparaEt_sub)

#=======================================================
#  Return of Epara multiplot (1)
#=======================================================

    for i in range(Operation_Class):
        Epara_Multi1(Epara[i][0], Epara[i][1], Epara[i][2], Epara[i][3], Epara[i][4], Epara[i][5], Epara[i][6], Epara[i][7], Epara[i][8], Epara[i][9],
                     Epara[i][10], Epara[i][11], Epara[i][12], Epara[i][13], Epara[i][14], Epara[i][15], Epara[i][16], "Epara", "Operation%s" % (i+1), "Shotnumber1-%s" % (i+1), "Shotnumber2-%s" % (i+1))
        Epara_Multi1(EparaEz[i][0], EparaEz[i][1], EparaEz[i][2], EparaEz[i][3], EparaEz[i][4], EparaEz[i][5], EparaEz[i][6], EparaEz[i][7], EparaEz[i][8], EparaEz[i][9],
                     EparaEz[i][10], EparaEz[i][11], EparaEz[i][12], EparaEz[i][13], EparaEz[i][14], EparaEz[i][15], EparaEz[i][16], "EparaEz", "Operation%s" % (i+1), "Shotnumber1-%s" % (i+1), "Shotnumber2-%s" % (i+1))
        Epara_Multi1(EparaEt[i][0], EparaEt[i][1], EparaEt[i][2], EparaEt[i][3], EparaEt[i][4], EparaEt[i][5], EparaEt[i][6], EparaEt[i][7], EparaEt[i][8], EparaEt[i][9],
                     EparaEt[i][10], EparaEt[i][11], EparaEt[i][12], EparaEt[i][13], EparaEt[i][14], EparaEt[i][15], EparaEt[i][16], "EparaEt", "Operation%s" % (i+1), "Shotnumber1-%s" % (i+1), "Shotnumber2-%s" % (i+1))

#=======================================================
#  Return of Epara multiplot (2)
#=======================================================

    for i in range(EparaCH):
        Epara_Multi2(Epara[0][i], Epara[1][i], Epara[2][i], Epara[3][i], Epara[4][i], Epara[5][i], Epara[6][i], "Epara", "%s" % (i+1))
        Epara_Multi2(EparaEz[0][i], EparaEz[1][i], EparaEz[2][i], EparaEz[3][i], EparaEz[4][i], EparaEz[5][i], EparaEz[6][i], "EparaEz", "%s" % (i+1))
        Epara_Multi2(EparaEt[0][i], EparaEt[1][i], EparaEt[2][i], EparaEt[3][i], EparaEt[4][i], EparaEt[5][i], EparaEt[6][i], "EparaEt", "%s" % (i+1))

#=======================================================