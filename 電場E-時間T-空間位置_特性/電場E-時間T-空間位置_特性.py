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

FigsizeC = (12, 3.5)

(xmin, xmax) = (9470, 9540)
(ymin, ymax) = (160, 480)

(EparaCmin,   EparaCmax)   = (-180, 140)
(EparaEzCmin, EparaEzCmax) = (-200, 140)
(EparaEtCmin, EparaEtCmax) = (-140, 180)

yticks_radius = np.arange(200,500,100)

Xp_range = 0.0003

color_type = 200
cmap_type_Epara    = 'jet'
cmap_type_EparaEz  = 'jet_r'
cmap_type_EparaEt  = 'jet'

#=======================================================
#  Formatting of graph
#=======================================================

savedir = os.getcwd() + "/" + "Epara_Color"
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
#  Definition of magnetic probe (1)
#=======================================================

def Magnetic_Probe1():

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
#  Definition of magnetic probe (2)
#=======================================================

def Magnetic_Probe2():

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

    SpPosi, Xpoint, SpTBz = [], [], []
    Xpoint_max, Xpoint_min, SpTBz_max, SpTBz_min = [], [], [], []
    Xpoint_average, Xpoint_range = [], []

    for i in range(sampleC):

        spPosi_bz, spTBz = Spline(Posi_bzll[2:16], TBz[int(i)][2:16])
        spPosi_bz, spTBz_max = Spline(Posi_bzll[2:16], TBz[int(i)][2:16])
        spPosi_bz, spTBz_min = Spline(Posi_bzll[2:16], TBz[int(i)][2:16])

        SpPosi.append(spPosi_bz), SpTBz.append(spTBz)
        SpTBz_max.append(spTBz_max), SpTBz_min.append(spTBz_min) 

        nXpoint = Get_Nearest(spTBz_max, 0)
        nXpoint_max = Get_Nearest(spTBz_max, -Xp_range)
        nXpoint_min = Get_Nearest(spTBz_min, Xp_range)

        Bzrange = 50
        spTBz[spPosi_bz > spPosi_bz[nXpoint] + Bzrange] = 0
        spTBz[spPosi_bz < spPosi_bz[nXpoint] - Bzrange] = 0
        spTBz_max[spPosi_bz > spPosi_bz[nXpoint_max] + Bzrange] = 0
        spTBz_max[spPosi_bz < spPosi_bz[nXpoint_max] - Bzrange] = 0
        spTBz_min[spPosi_bz > spPosi_bz[nXpoint_min] + Bzrange] = 0
        spTBz_min[spPosi_bz < spPosi_bz[nXpoint_min] - Bzrange] = 0

        Xpoint.append(spPosi_bz[nXpoint])
        Xpoint_max.append(spPosi_bz[nXpoint_max])
        Xpoint_min.append(spPosi_bz[nXpoint_min])

        xpoint_average = Xpoint_max[i]/2 + Xpoint_min[i]/2
        xpoint_range = abs(Xpoint_max[i] - Xpoint_min[i])

        Xpoint_average.append(xpoint_average)
        Xpoint_range.append(xpoint_range)

    return (Xpoint, Xpoint_average, Xpoint_range)

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
#  Definition of Epara X-point colorplot (1)
#=======================================================

def Epara_Color1(signal1, signal2, type_graph, str_title):

    def Epara_Graph1(cmap_type, zmin, zmax, name1, name2, savedict):

        global time

        dict = {'Epara':'$E_{//}$  [V/m]', 'EparaEz':'$E_{//,z}$  [V/m]', 'EparaEt':'$E_{//,t}$  [V/m]',
                'Operation1': Operation[0], 'Operation2': Operation[1], 'Operation3': Operation[2], 'Operation4': Operation[3], 'Operation5': Operation[4], 'Operation6': Operation[5], 'Operation7': Operation[6]}

        plt.figure(figsize=FigsizeC)
        plt.plot(time, signal1, 'black', marker='o', markersize='5', label='X-point')
        X,Y = np.meshgrid(time, Posi_Epara)
        CF = plt.contourf(X, Y, signal2, color_type, cmap=cmap_type)
        CF.set_clim(zmin, zmax)
        CB = plt.colorbar(CF, ticks=[-240,-200,-160,-120,-80,-40,0,40,80,120,160,200])
        CB.set_label("%s" %(dict[type_graph]), labelpad=8, fontsize=Fontsize)
        plt.title(name1 + "   %s     X-point : Bz = 0" % (dict[str_title]), fontsize=Fontsize)
        plt.gca().xaxis.set_tick_params(which='both', direction='in', bottom=True)
        plt.gca().yaxis.set_tick_params(which='both', direction='in', left=True)
        plt.xlabel("Time  [μs]", labelpad=8, fontsize=Fontsize)
        plt.ylabel("R [mm]", labelpad=15, fontsize=Fontsize)
        plt.legend(fontsize=20, loc='lower right')
        plt.tick_params(pad=8)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.yticks(yticks_radius)
        plt.savefig(savedict + "/%s_%s.png" %(name2, dict[str_title]), bbox_inches='tight')
        plt.close()

    if(type_graph == 'Epara'):

        savedir_Epara_color1 = savedir + "/" + "Epara_Colorplot1"
        if not(os.access(savedir_Epara_color1,os.F_OK)):
            os.mkdir(savedir_Epara_color1)

        Epara_Graph1(cmap_type_Epara, EparaCmin, EparaCmax, "$E_{//}$", "Epara", savedir_Epara_color1)

    elif(type_graph == 'EparaEz'):

        savedir_EparaEz_color1 = savedir + "/" + "EparaEz_Colorplot1"
        if not(os.access(savedir_EparaEz_color1,os.F_OK)):
            os.mkdir(savedir_EparaEz_color1)

        Epara_Graph1(cmap_type_EparaEz, EparaEzCmin, EparaEzCmax, "$E_{//,z}$", "EparaEz", savedir_EparaEz_color1)

    elif(type_graph == 'EparaEt'):

        savedir_EparaEt_color1 = savedir + "/" + "EparaEt_Colorplot1"
        if not(os.access(savedir_EparaEt_color1,os.F_OK)):
            os.mkdir(savedir_EparaEt_color1)

        Epara_Graph1(cmap_type_EparaEt, EparaEtCmin, EparaEtCmax, "$E_{//,z}$", "EparaEt", savedir_EparaEt_color1)

    else:()

#=======================================================
#  Definition of Epara X-point colorplot (2)
#=======================================================

def Epara_Color2(signal1, signal2, signal3, type_graph, str_title):

    def Epara_Graph2(cmap_type, zmin, zmax, name1, name2, savedict):

        global time

        dict = {'Epara':'$E_{//}$  [V/m]', 'EparaEz':'$E_{//,z}$  [V/m]', 'EparaEt':'$E_{//,t}$  [V/m]',
                'Operation1': Operation[0], 'Operation2': Operation[1], 'Operation3': Operation[2], 'Operation4': Operation[3], 'Operation5': Operation[4], 'Operation6': Operation[5], 'Operation7': Operation[6]}

        plt.figure(figsize=FigsizeC)
        plt.errorbar(time, signal1, yerr=signal2, lw=2, elinewidth=2, color='black', ecolor='black', label='X-point')
        X,Y = np.meshgrid(time, Posi_Epara)
        CF = plt.contourf(X, Y, signal3, color_type, cmap=cmap_type)
        CF.set_clim(zmin, zmax)
        CB = plt.colorbar(CF, ticks=[-240,-200,-160,-120,-80,-40,0,40,80,120,160,200])
        CB.set_label("%s" %(dict[type_graph]), labelpad=8, fontsize=Fontsize)
        plt.title(name1 + "   %s     X-point : |Bz| < %s [mT]" % (dict[str_title], Xp_range*1000), fontsize=Fontsize)
        plt.gca().xaxis.set_tick_params(which='both', direction='in', bottom=True)
        plt.gca().yaxis.set_tick_params(which='both', direction='in', left=True)
        plt.xlabel("Time  [μs]", labelpad=8, fontsize=Fontsize)
        plt.ylabel("R [mm]", labelpad=15, fontsize=Fontsize)
        plt.legend(fontsize=20, loc='lower right')
        plt.tick_params(pad=8)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.yticks(yticks_radius)
        plt.savefig(savedict + "/%s_%s.png" %(name2, dict[str_title]), bbox_inches='tight')
        plt.close()

    if(type_graph == 'Epara'):

        savedir_Epara_color2 = savedir + "/" + "Epara_Colorplot2"
        if not(os.access(savedir_Epara_color2,os.F_OK)):
            os.mkdir(savedir_Epara_color2)

        Epara_Graph2(cmap_type_Epara, EparaCmin, EparaCmax, "$E_{//}$", "Epara", savedir_Epara_color2)

    elif(type_graph == 'EparaEz'):

        savedir_EparaEz_color2 = savedir + "/" + "EparaEz_Colorplot2"
        if not(os.access(savedir_EparaEz_color2,os.F_OK)):
            os.mkdir(savedir_EparaEz_color2)

        Epara_Graph2(cmap_type_EparaEz, EparaEzCmin, EparaEzCmax, "$E_{//,z}$", "EparaEz", savedir_EparaEz_color2)

    elif(type_graph == 'EparaEt'):

        savedir_EparaEt_color2 = savedir + "/" + "EparaEt_Colorplot2"
        if not(os.access(savedir_EparaEt_color2,os.F_OK)):
            os.mkdir(savedir_EparaEt_color2)

        Epara_Graph2(cmap_type_EparaEt, EparaEtCmin, EparaEtCmax, "$E_{//,z}$", "EparaEt", savedir_EparaEt_color2)

    else:()

#=======================================================
#  Definition of Epara X-point colorplot (3)
#=======================================================

def Epara_Color3(signal1, signal2, signal3, signal4, signal5, signal6, signal7, signal8, signal9, signal10, signal11, signal12, signal13, signal14, type_graph, str_title):

    def Epara_Graph3(cmap_type, zmin, zmax, name1, name2, savedict):

        def Epara_Graph_Sub3():

            plt.gca().xaxis.set_tick_params(which='both', direction='in', bottom=True)
            plt.gca().yaxis.set_tick_params(which='both', direction='in', left=True)
            plt.ylabel("R [mm]", labelpad=15, fontsize=Fontsize)
            plt.legend(fontsize=20, loc='lower right')
            plt.tick_params(pad=8)
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            plt.yticks(yticks_radius)

        global time

        fig = plt.figure(figsize=(12,25))
        X,Y = np.meshgrid(time, Posi_Epara)

        ax1 = fig.add_subplot(711)
        ax1.set_xticklabels([])
        ax1.plot(time, signal1, 'black', marker='o', markersize='5', label='X-point')
        CF = plt.contourf(X, Y, signal2, color_type, cmap=cmap_type)
        CF.set_clim(zmin, zmax)
        plt.title(name1 + "   %s    %s  %s" % (Operation[0], Shotnumber[0][0], Shotnumber[0][1]), fontsize=Fontsize)
        Epara_Graph_Sub3()

        ax2 = fig.add_subplot(712)
        ax2.set_xticklabels([])
        ax2.plot(time, signal3, 'black', marker='o', markersize='5', label='X-point')
        CF = plt.contourf(X, Y, signal4, color_type, cmap=cmap_type)
        CF.set_clim(zmin, zmax)
        plt.title(name1 + "   %s    %s  %s" % (Operation[1], Shotnumber[1][0], Shotnumber[1][1]), fontsize=Fontsize)
        Epara_Graph_Sub3()

        ax3 = fig.add_subplot(713)
        ax3.set_xticklabels([])
        ax3.plot(time, signal5, 'black', marker='o', markersize='5', label='X-point')
        CF = plt.contourf(X, Y, signal6, color_type, cmap=cmap_type)
        CF.set_clim(zmin, zmax)
        plt.title(name1 + "   %s    %s  %s" % (Operation[2], Shotnumber[2][0], Shotnumber[2][1]), fontsize=Fontsize)
        Epara_Graph_Sub3()

        ax4 = fig.add_subplot(714)
        ax4.set_xticklabels([])
        ax4.plot(time, signal7, 'black', marker='o', markersize='5', label='X-point')
        CF = plt.contourf(X, Y, signal8, color_type, cmap=cmap_type)
        CF.set_clim(zmin, zmax)
        plt.title(name1 + "   %s    %s  %s" % (Operation[3], Shotnumber[3][0], Shotnumber[3][1]), fontsize=Fontsize)
        Epara_Graph_Sub3()

        ax5 = fig.add_subplot(715)
        ax5.set_xticklabels([])
        ax5.plot(time, signal9, 'black', marker='o', markersize='5', label='X-point')
        CF = plt.contourf(X, Y, signal10, color_type, cmap=cmap_type)
        CF.set_clim(zmin, zmax)
        plt.title("$E_{//}$   %s    %s  %s" % (Operation[4], Shotnumber[4][0], Shotnumber[4][1]), fontsize=Fontsize)
        Epara_Graph_Sub3()

        ax6 = fig.add_subplot(716)
        ax6.set_xticklabels([])
        ax6.plot(time, signal11, 'black', marker='o', markersize='5', label='X-point')
        CF = plt.contourf(X, Y, signal12, color_type, cmap=cmap_type)
        CF.set_clim(zmin, zmax)
        plt.title(name1 + "   %s    %s  %s" % (Operation[5], Shotnumber[5][0], Shotnumber[5][1]), fontsize=Fontsize)
        Epara_Graph_Sub3()

        ax7 = fig.add_subplot(717)
        ax7.plot(time, signal13, 'black', marker='o', markersize='5', label='X-point')
        CF = plt.contourf(X, Y, signal14, color_type, cmap=cmap_type)
        CF.set_clim(zmin, zmax)
        plt.title(name1 + "   %s    %s  %s" % (Operation[6], Shotnumber[6][0], Shotnumber[6][1]), fontsize=Fontsize)
        Epara_Graph_Sub3()

        plt.subplots_adjust(hspace=0.25)
        fig.text(0.51, 0.1, 'Time  [μs]', fontsize=Fontsize, ha='center', va='center')
        plt.savefig("%s" % (savedict) + "/%s.png" % (name2), bbox_inches='tight')
        plt.close()

    if(type_graph == 'Epara'):

        savedir_Epara_color3 = savedir + "/" + "Epara_Colorplot3"
        if not(os.access(savedir_Epara_color3,os.F_OK)):
            os.mkdir(savedir_Epara_color3)

        Epara_Graph3(cmap_type_Epara, EparaCmin, EparaCmax, "$E_{//}$", "Epara", savedir_Epara_color3)

    elif(type_graph == 'EparaEz'):

        savedir_EparaEz_color3 = savedir + "/" + "EparaEz_Colorplot3"
        if not(os.access(savedir_EparaEz_color3,os.F_OK)):
            os.mkdir(savedir_EparaEz_color3)

        Epara_Graph3(cmap_type_EparaEz, EparaEzCmin, EparaEzCmax, "$E_{//,z}$", "EparaEz", savedir_EparaEz_color3)

    elif(type_graph == 'EparaEt'):

        savedir_EparaEt_color3 = savedir + "/" + "EparaEt_Colorplot3"
        if not(os.access(savedir_EparaEt_color3,os.F_OK)):
            os.mkdir(savedir_EparaEt_color3)

        Epara_Graph3(cmap_type_EparaEt, EparaEtCmin, EparaEtCmax, "$E_{//,z}$", "EparaEt", savedir_EparaEt_color3)

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

(SpTBz, SpTEt), (Xp, Xpave, Xpran) = (np.zeros((Operation_Class, sampleC, xSpBz)), np.zeros((Operation_Class, sampleC, xSpEt))), (np.zeros((Operation_Class, sampleC)), np.zeros((Operation_Class, sampleC)), np.zeros((Operation_Class, sampleC)))
for i in range(Operation_Class):
    for shotname in [Shot[i]]:
        (f1, f2, f3, sample_period, full_range1, full_range2, full_range3, precision1, precision2, precision3, init_time, fin_time, dt, time, time_PF3) = File_Data(shotname)
        (SpTBz[i], SpTEt[i]), (Xp[i], Xpave[i], Xpran[i]) = Magnetic_Probe1(), Magnetic_Probe2()
        (SpTBz, SpTEt), (Xp, Xpave, Xpran) = (SpTBz.reshape([Operation_Class, sampleC, xSpBz]), SpTEt.reshape([Operation_Class, sampleC, xSpEt])), (Xp.reshape([Operation_Class, sampleC]), Xpave.reshape([Operation_Class, sampleC]), Xpran.reshape([Operation_Class, sampleC]))

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
#  Return of Epara X-point colorplot (1)
#=======================================================

    for i in range(Operation_Class):
        Epara_Color1(Xp[i], Epara[i], "Epara", "Operation%s" % (i+1))
        Epara_Color1(Xp[i], EparaEz[i], "EparaEz", "Operation%s" % (i+1))
        Epara_Color1(Xp[i], EparaEt[i], "EparaEt", "Operation%s" % (i+1))

#=======================================================
#  Return of Epara X-point colorplot (2)
#=======================================================

    for i in range(Operation_Class):
        Epara_Color2(Xpave[i], Xpran[i], Epara[i], "Epara", "Operation%s" % (i+1))
        Epara_Color2(Xpave[i], Xpran[i], EparaEz[i], "EparaEz", "Operation%s" % (i+1))
        Epara_Color2(Xpave[i], Xpran[i], EparaEt[i], "EparaEt", "Operation%s" % (i+1))

#=======================================================
#  Return of Epara X-point colorplot (3)
#=======================================================

    Epara_Color3(Xp[0], Epara[0], Xp[1], Epara[1], Xp[2], Epara[2], Xp[3], Epara[3], Xp[4], Epara[4], Xp[5], Epara[5], Xp[6], Epara[6], "Epara", " ")
    Epara_Color3(Xp[0], EparaEz[0], Xp[1], EparaEz[1], Xp[2], EparaEz[2], Xp[3], EparaEz[3], Xp[4], EparaEz[4], Xp[5], EparaEz[5], Xp[6], EparaEz[6], "EparaEz", " ")
    Epara_Color3(Xp[0], EparaEt[0], Xp[1], EparaEt[1], Xp[2], EparaEt[2], Xp[3], EparaEt[3], Xp[4], EparaEt[4], Xp[5], EparaEt[5], Xp[6], EparaEt[6], "EparaEt", " ")

#=======================================================