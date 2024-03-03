#=======================================================
#  Import gpa file
#=======================================================

(Operation_Class, Operation_Shot) = (6, 2)

Operation1 = 'Open'
Shotnumber1 = ['#200819005', '#200819008']
(Shot1_in, Shot1_out) = ('200819005', '200819008')

Operation2 = 'Short [1]'
Shotnumber2 = ['#200909011', '#200909004']
(Shot2_in, Shot2_out) = ('200909011', '200909004')

Operation3 = 'Short [2]'
Shotnumber3 = ['#200828003', '#200828008']
(Shot3_in, Shot3_out) = ('200828003', '200828008')

Operation4 = 'Short [3]'
Shotnumber4 = ['#200826008', '#200826003']
(Shot4_in, Shot4_out) = ('200826008', '200826003')

Operation5 = 'Short [4]'
Shotnumber5 = ['#200820008', '#200820005']
(Shot5_in, Shot5_out) = ('200820008', '200820005')

Operation6 = 'Short [1][2][3][4]'
Shotnumber6 = ['#200908007', '#200908009']
(Shot6_in, Shot6_out) = ('200908007', '200908009')

Operation  = [Operation1, Operation2, Operation3, Operation4, Operation5, Operation6]

Shot = [Shot1_out, Shot2_in, Shot3_in, Shot4_in, Shot5_in, Shot6_out]
Shotnumber = [Shotnumber1, Shotnumber2, Shotnumber3, Shotnumber4, Shotnumber5, Shotnumber6]

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
#  Parameter of graph
#=======================================================

Fontsize = 18
LP_freq = 200

Spline_Amp = 100

Figsize = (10,3.5)

(t0, t1) = (9470, 9540)
Time = np.arange(t0, t1+1, 1)
Timelen = t1-t0+1

(Psixmin, Psixmax) = (9470, 9540)
(Psiymin, Psiymax) = (115, 480)

yticks_radius = np.arange(200,500,100)

Z = 0  # Psi[t][50], Psiraw[t][495]~[504]

Electrode_Z = [185, 285]  # Psi[t][99], Psiraw[t][990]~[999], Z = 230

Electrode_R1 = [112.5, 127.5]  # Sp_Psi_z230[t][20],  Sp_Psi_z230[t][140]
Electrode_R2 = [132.5, 147.5]  # Sp_Psi_z230[t][180], Sp_Psi_z230[t][300]
Electrode_R3 = [152.5, 167.5]  # Sp_Psi_z230[t][340], Sp_Psi_z230[t][460]
Electrode_R4 = [172.5, 187.5]  # Sp_Psi_z230[t][500], Sp_Psi_z230[t][620]

Plotnumber_in = [320, 460, 620, 780]       # 追加電極内側の径方向位置の格納番号
Plotnumber_out = [400, 580, 740, 900]      # 追加電極外側の径方向位置の格納番号
Plotnumber_Limiter = [280]  # リミター外側の径方向位置の格納番号

#=======================================================
#  Formatting of Graph
#=======================================================

savedir = os.getcwd() + "/" + "Magnetic_Probe_Psi"
if not(os.access(savedir,os.F_OK)):
    os.mkdir(savedir)

plt.rcParams.update({'mathtext.default':'default', 'mathtext.fontset':'stix', 'font.family':'Arial', 'font.size':Fontsize})

#=======================================================
#  Parameter of oscillo scope
#=======================================================

sample = 10000
(sampleA, sampleB) = (8800, 9800)
sampleC = sampleB - sampleA

Shotlist = []
for i in range(Operation_Class):
    shotlist = 'Data_gpa/' + Shot[i]
    Shotlist.append(shotlist)

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

def Spline_Bz(in_x, in_y):
    
    out_x = np.linspace(np.min(in_x), np.max(in_x), (np.max(in_x)-np.min(in_x))*100)
    func_spline = interp1d(in_x, in_y, kind='cubic')
    out_y = func_spline(out_x)

    return out_x, out_y

def Spline_Psi(in_x, in_y):
    
    out_x = np.linspace(np.min(in_x), np.max(in_x), (np.max(in_x)*100-np.min(in_x)*100)*Spline_Amp)
    func_spline = interp1d(in_x, in_y, kind='cubic')
    out_y = func_spline(out_x)

    return out_x, out_y

#=======================================================
#  Definition of getting nearest value
#=======================================================

def Get_Nearest(list, num):
    
    idx = np.abs(np.asarray(list) - num).argmin()
    
    return idx

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

    TBz = np.array(Bzll).T  # （R,Bz） → (Bz,R)

    SpPosi, Xpoint, SpTBz = [], [], []

    for i in range(670,741,1):

        spPosi_bz, spTBz = Spline_Bz(Posi_bzll[2:16], TBz[int(i)][2:16])
        SpPosi.append(spPosi_bz), SpTBz.append(spTBz)
        nXpoint = Get_Nearest(spTBz, 0)
        
        Bzrange = 50
        spTBz[spPosi_bz > spPosi_bz[nXpoint] + Bzrange] = 0
        spTBz[spPosi_bz < spPosi_bz[nXpoint] - Bzrange] = 0
        Xpoint.append(spPosi_bz[nXpoint])

    return Xpoint

#=======================================================
#  Import dat file (R-axis)
#=======================================================

psiraw_r = 'Data_psi/Psi_r.dat'
Psiraw_r = np.loadtxt(psiraw_r, comments='#')

Psi_r = []
for i in range(len(Psiraw_r)):
    psi_r = Psiraw_r[i]
    Psi_r.extend(psi_r)

#=======================================================
#  Import dat file (Psi)
#=======================================================

Psiraw = []  # Psiの2次元分布
for i in range(Operation_Class):
    Psiraw_sub = []
    for j in range(Timelen):
        if(j<10):
            psiread = 'Data_psi/' + Shot[i] + '/2d0%s.dat' % (j)
        else:
            psiread = 'Data_psi/' + Shot[i] + '/2d%s.dat' % (j)
        psiraw_sub = np.loadtxt(psiread, comments='#')
        Psiraw_sub.append(psiraw_sub)
    Psiraw.append(Psiraw_sub)

Psi_z0 = []  # Psi (Z=0)
for i in range(Operation_Class):
    Psi_z0_sub = []
    for j in range(Timelen):
        Psi_z0_sub_sub = []
        for k in range(len(Psiraw_r)):
            psi_z0_sub_sub = Psiraw[i][j][490+k]
            Psi_z0_sub_sub.extend(psi_z0_sub_sub)
        Psi_z0_sub.append(Psi_z0_sub_sub)
    Psi_z0.append(Psi_z0_sub)

Psi_z230 = []  # Psi (Z=230)
for i in range(Operation_Class):
    Psi_z230_sub = []
    for j in range(Timelen):
        Psi_z230_sub_sub = []
        for k in range(len(Psiraw_r)):
            psi_z230_sub_sub = Psiraw[i][j][990+k]
            Psi_z230_sub_sub.extend(psi_z230_sub_sub)
        Psi_z230_sub.append(Psi_z230_sub_sub)
    Psi_z230.append(Psi_z230_sub)

#=======================================================
#  Psi spline interpolation
#=======================================================

Sp_Psi_r, Sp_Psi_z0, Sp_Psi_z230 = [], [], []
for i in range(Operation_Class):
    Sp_Psi_r_sub, Sp_Psi_z0_sub, Sp_Psi_z230_sub = [], [], []
    for j in range(Timelen):
        sp_Psi_r_sub, sp_Psi_z0_sub = Spline_Psi(Psi_r, Psi_z0[i][j])
        sp_Psi_r_sub, sp_Psi_z230_sub = Spline_Psi(Psi_r, Psi_z230[i][j])
        Sp_Psi_r_sub.append(sp_Psi_r_sub)
        Sp_Psi_z0_sub.append(sp_Psi_z0_sub)
        Sp_Psi_z230_sub.append(sp_Psi_z230_sub)
    Sp_Psi_r.append(Sp_Psi_r_sub)
    Sp_Psi_z0.append(Sp_Psi_z0_sub)
    Sp_Psi_z230.append(Sp_Psi_z230_sub)

#=======================================================
#  Radial Position (Limiter outside)
#=======================================================

Electrode_Psi_Limiter = []  # Psi (リミター外側)
for i in range(Operation_Class):
    Electrode_Psi_Limiter_sub = []
    for j in range(t1-t0+1):
        Electrode_Psi_Limiter_sub_sub = []
        for k in range(len(Plotnumber_Limiter)):
            electrode_psi_Limiter_sub_sub = Sp_Psi_z230[i][j][Plotnumber_Limiter[k]]
            Electrode_Psi_Limiter_sub_sub.append(electrode_psi_Limiter_sub_sub)     
        Electrode_Psi_Limiter_sub.append(Electrode_Psi_Limiter_sub_sub)
    Electrode_Psi_Limiter.append(Electrode_Psi_Limiter_sub)

Dupe_Limiter = []  # リミター外側とZ=0のPsiが同じになる径方向位置
for i in range(Operation_Class):
    Dupe_Limiter_sub = []
    for j in range(t1-t0+1):
        Dupe_Limiter_sub_sub = []
        for k in range(len(Plotnumber_Limiter)):
            dupe_Limiter_sub_sub = Get_Nearest(Sp_Psi_z0[i][j][0:1200], Electrode_Psi_Limiter[i][j][k])
            dupe_Limiter_sub_sub = 110 + 0.112 * dupe_Limiter_sub_sub
            Dupe_Limiter_sub_sub.append(dupe_Limiter_sub_sub)
        Dupe_Limiter_sub.append(Dupe_Limiter_sub_sub)
    Dupe_Limiter.append(Dupe_Limiter_sub)

Dupe_Limiterl = []
for i in range(Operation_Class):
    dupe_Limiterl = np.array(Dupe_Limiter[i]).T
    Dupe_Limiterl.append(dupe_Limiterl)

Dupe_Limiter = np.array(Dupe_Limiterl)
Dupe_Limiter_110 = np.full((Operation_Class, len(Plotnumber_Limiter), t1-t0+1), 110)

#=======================================================
#  Radial Position (Additional electrodes inside)
#=======================================================

Electrode_Psi_in = []  # Psi (追加電極内側)
for i in range(Operation_Class):
    Electrode_Psi_in_sub = []
    for j in range(Timelen):
        Electrode_Psi_in_sub_sub = []
        for k in range(len(Plotnumber_in)):
            electrode_psi_in_sub_sub = Sp_Psi_z230[i][j][Plotnumber_in[k]]
            Electrode_Psi_in_sub_sub.append(electrode_psi_in_sub_sub)     
        Electrode_Psi_in_sub.append(Electrode_Psi_in_sub_sub)
    Electrode_Psi_in.append(Electrode_Psi_in_sub)

Dupe_in = []  # 追加電極内側とZ=0のPsiが同じになる径方向位置
for i in range(Operation_Class):
    Dupe_in_sub = []
    for j in range(Timelen):
        Dupe_in_sub_sub = []
        for k in range(len(Plotnumber_in)):
            dupe_in_sub_sub = Get_Nearest(Sp_Psi_z0[i][j][0:1500], Electrode_Psi_in[i][j][k])
            dupe_in_review = abs(Sp_Psi_z0[i][j][dupe_in_sub_sub] - Electrode_Psi_in[i][j][k])
            if(dupe_in_review < 0.08):
                dupe_in_sub_sub_review = 110 + 0.112 * dupe_in_sub_sub
            else:
                dupe_in_sub_sub_review = 0
            Dupe_in_sub_sub.append(dupe_in_sub_sub_review)
        Dupe_in_sub.append(Dupe_in_sub_sub)
    Dupe_in.append(Dupe_in_sub)

Dupe_inl = []
for i in range(Operation_Class):
    dupe_inl = np.array(Dupe_in[i]).T
    Dupe_inl.append(dupe_inl)

Dupe_in = np.array(Dupe_inl)

#=======================================================
#  Radial Position (Additional electrodes outside)
#=======================================================

Electrode_Psi_out = []  # Psi (追加電極外側)
for i in range(Operation_Class):
    Electrode_Psi_out_sub = []
    for j in range(Timelen):
        Electrode_Psi_out_sub_sub = []
        for k in range(len(Plotnumber_out)):
            electrode_psi_out_sub_sub = Sp_Psi_z230[i][j][Plotnumber_out[k]]
            Electrode_Psi_out_sub_sub.append(electrode_psi_out_sub_sub)     
        Electrode_Psi_out_sub.append(Electrode_Psi_out_sub_sub)
    Electrode_Psi_out.append(Electrode_Psi_out_sub)

Dupe_out = []  # 追加電極外側とZ=0のPsiが同じになる径方向位置
for i in range(Operation_Class):
    Dupe_out_sub = []
    for j in range(Timelen):
        Dupe_out_sub_sub = []
        for k in range(len(Plotnumber_out)):
            dupe_out_sub_sub = Get_Nearest(Sp_Psi_z0[i][j][0:1500], Electrode_Psi_out[i][j][k])
            dupe_out_review = abs(Sp_Psi_z0[i][j][dupe_out_sub_sub] - Electrode_Psi_out[i][j][k])
            if(dupe_out_review < 0.08):
                dupe_out_sub_sub_review = 110 + 0.112 * dupe_out_sub_sub
            else:
                dupe_out_sub_sub_review = 0
            Dupe_out_sub_sub.append(dupe_out_sub_sub_review)
        Dupe_out_sub.append(Dupe_out_sub_sub)
    Dupe_out.append(Dupe_out_sub)

Dupe_outl = []
for i in range(Operation_Class):
    dupe_outl = np.array(Dupe_out[i]).T
    Dupe_outl.append(dupe_outl)

Dupe_out = np.array(Dupe_outl)

#=======================================================
#  Radial Position (inside + outside)
#=======================================================

Dupe_ave, Dupe_err = [], []
for i in range(Operation_Class):
    Dupe_ave_sub, Dupe_err_sub = [], []
    for j in range(len(Plotnumber_in)):
        Dupe_ave_sub_sub, Dupe_err_sub_sub = [], []
        for k in range(Timelen):
            Dupe_max, Dupe_min = max([Dupe_in[i][j][k], Dupe_out[i][j][k]]), min([Dupe_in[i][j][k], Dupe_out[i][j][k]])
            Dupe_average, dupe_error = Dupe_max/2 + Dupe_min/2, Dupe_max - Dupe_min
            if(dupe_error < 150):
                Dupe_error = Dupe_max - Dupe_min
            else:
                Dupe_error = 0
            Dupe_ave_sub_sub.append(Dupe_average), Dupe_err_sub_sub.append(Dupe_error)
        Dupe_ave_sub.append(Dupe_ave_sub_sub), Dupe_err_sub.append(Dupe_err_sub_sub)
    Dupe_ave.append(Dupe_ave_sub), Dupe_err.append(Dupe_err_sub)

Dupe_Limiter_ave, Dupe_Limiter_err = [], []
for i in range(Operation_Class):
    Dupe_Limiter_ave_sub, Dupe_Limiter_err_sub = [], []
    for j in range(len(Plotnumber_Limiter)):
        Dupe_Limiter_ave_sub_sub, Dupe_Limiter_err_sub_sub = [], []
        for k in range(t1-t0+1):
            Dupe_Limiter_max, Dupe_Limiter_min = max([Dupe_Limiter[i][j][k], Dupe_Limiter_110[i][j][k]]), min([Dupe_Limiter[i][j][k], Dupe_Limiter_110[i][j][k]])
            Dupe_Limiter_average, Dupe_Limiter_error = Dupe_Limiter_max/2 + Dupe_Limiter_min/2, Dupe_Limiter_max - Dupe_Limiter_min
            if(dupe_error < 300):
                Dupe_Limiter_error = Dupe_Limiter_max - Dupe_Limiter_min
            else:
                Dupe_Limiter_error = 300
            Dupe_Limiter_ave_sub_sub.append(Dupe_Limiter_average), Dupe_Limiter_err_sub_sub.append(Dupe_Limiter_error)
        Dupe_Limiter_ave_sub.append(Dupe_Limiter_ave_sub_sub), Dupe_Limiter_err_sub.append(Dupe_Limiter_err_sub_sub)
    Dupe_Limiter_ave.append(Dupe_Limiter_ave_sub), Dupe_Limiter_err.append(Dupe_Limiter_err_sub)

#=======================================================
#  Definition of Psi muitiplot (1)
#=======================================================

def Psi1(signal1, Limiter1, Limiter2, type_graph, str_title, shotnumber):

    global time

    dict = {'Operation': Operation[0], 'Shotnumber': Shotnumber[0][1]}

    savedir_Psi1 = savedir + "/" + "Psi_Plot1"
    if not(os.access(savedir_Psi1,os.F_OK)):
        os.mkdir(savedir_Psi1)

    if(type_graph == 'Psi'):

        fig,ax = plt.subplots(figsize=Figsize)
        ax.plot(Time, signal1, 'red', marker='o', markersize='4', label='X-point')
        ax.errorbar(Time, Limiter1, yerr=Limiter2, elinewidth=3, lw=0, ecolor='lightblue', label='Psi : Limiter')
        plt.gca().xaxis.set_tick_params(which='both', direction='in', bottom=True)
        plt.gca().yaxis.set_tick_params(which='both', direction='in', left=True)
        plt.xlabel("Time  [μs]", labelpad=8, fontsize=Fontsize)
        plt.ylabel("R [mm]", labelpad=15, fontsize=Fontsize)
        plt.title("Psi   %s    %s" % (dict[str_title], dict[shotnumber]), fontsize=Fontsize)
        plt.legend(bbox_to_anchor=(1, 1.00), loc='upper left', labelspacing=1, fontsize=16)
        plt.tick_params(pad=8)
        plt.xlim(Psixmin, Psixmax)
        plt.ylim(Psiymin, Psiymax)
        plt.yticks(yticks_radius)
        plt.savefig("%s/Psi_%s.png" %(savedir_Psi1, dict[str_title]), bbox_inches='tight')
        plt.close()

    else:()

#=======================================================
#  Definition of Psi muitiplot (2)
#=======================================================

def Psi2(signal1, signal2, signal3, Limiter1, Limiter2, type_graph, str_title, shotnumber, electrode):

    global time

    dict = {'Operation1': Operation[1], 'Operation2': Operation[2], 'Operation3': Operation[3], 'Operation4': Operation[4],
            'Shotnumber1': Shotnumber[0][0], 'Shotnumber2': Shotnumber[1][0], 'Shotnumber3': Shotnumber[2][0], 'Shotnumber4': Shotnumber[3][0], 'Shotnumber5': Shotnumber[4][0],
            'Electrode1': 'Electrode[1]', 'Electrode2': 'Electrode[2]', 'Electrode3': 'Electrode[3]', 'Electrode4': 'Electrode[4]'}

    savedir_Psi1 = savedir + "/" + "Psi_Plot1"
    if not(os.access(savedir_Psi1,os.F_OK)):
        os.mkdir(savedir_Psi1)

    if(type_graph == 'Psi'):

        fig,ax = plt.subplots(figsize=Figsize)
        ax.plot(Time, signal1, 'red', marker='o', markersize='4', label='X-point')
        ax.errorbar(Time, Limiter1, yerr=Limiter2, elinewidth=3, lw=0, ecolor='lightblue', label='Psi : Limiter')
        ax.errorbar(Time, signal2, yerr=signal3, elinewidth=3, lw=0, ecolor='black', label='Psi : %s' % (dict[electrode]))
        plt.gca().xaxis.set_tick_params(which='both', direction='in', bottom=True)
        plt.gca().yaxis.set_tick_params(which='both', direction='in', left=True)
        plt.xlabel("Time  [μs]", labelpad=8, fontsize=Fontsize)
        plt.ylabel("R [mm]", labelpad=15, fontsize=Fontsize)
        plt.title("Psi   %s    %s" % (dict[str_title], dict[shotnumber]), fontsize=Fontsize)
        plt.legend(bbox_to_anchor=(1, 1.00), loc='upper left', labelspacing=1, fontsize=16)
        plt.tick_params(pad=8)
        plt.xlim(Psixmin, Psixmax)
        plt.ylim(Psiymin, Psiymax)
        plt.yticks(yticks_radius)
        plt.savefig("%s/Psi_%s.png" %(savedir_Psi1, dict[str_title]), bbox_inches='tight')
        plt.close()

    else:()

#=======================================================
#  Definition of Psi muitiplot (3)
#=======================================================

def Psi3(signal1, signal2, signal3, signal4, signal5, signal6, signal7, signal8, signal9, Limiter1, Limiter2, type_graph, str_title, shotnumber):

    global time

    dict = {'Operation': Operation[5], 'Shotnumber': Shotnumber[5][1]}

    savedir_Psi1 = savedir + "/" + "Psi_Plot1"
    if not(os.access(savedir_Psi1,os.F_OK)):
        os.mkdir(savedir_Psi1)

    if(type_graph == 'Psi'):

        fig,ax = plt.subplots(figsize=Figsize)
        ax.plot(Time, signal1, 'red', marker='o', markersize='4', label='X-point')
        ax.errorbar(Time, Limiter1, yerr=Limiter2, elinewidth=3, lw=0, ecolor='lightblue', label='Psi : Limiter')
        ax.errorbar(Time, signal2, yerr=signal3, elinewidth=3, lw=0, ecolor='black', label='Psi : Electrode[1]')
        ax.errorbar(Time, signal4, yerr=signal5, elinewidth=3, lw=0, label='Psi : Electrode[2]')
        ax.errorbar(Time, signal6, yerr=signal7, elinewidth=3, lw=0, label='Psi : Electrode[3]')
        ax.errorbar(Time, signal8, yerr=signal9, elinewidth=3, lw=0, label='Psi : Electrode[4]')
        plt.gca().xaxis.set_tick_params(which='both', direction='in', bottom=True)
        plt.gca().yaxis.set_tick_params(which='both', direction='in', left=True)
        plt.xlabel("Time  [μs]", labelpad=8, fontsize=Fontsize)
        plt.ylabel("R [mm]", labelpad=15, fontsize=Fontsize)
        plt.title("Psi   %s    %s" % (dict[str_title], dict[shotnumber]), fontsize=Fontsize)
        plt.legend(bbox_to_anchor=(1, 1.04), loc='upper left', labelspacing=1, fontsize=16)
        plt.tick_params(pad=8)
        plt.xlim(Psixmin, Psixmax)
        plt.ylim(Psiymin, Psiymax)
        plt.yticks(yticks_radius)
        plt.savefig("%s/Psi_%s.png" %(savedir_Psi1, dict[str_title]), bbox_inches='tight')
        plt.close()

    else:()

#=======================================================
#  Definition of Psi muitiplot (4)
#=======================================================

def Psi4(Xp1, Xp2, Xp3, Xp4, Xp5, Xp6, signal1, signal2, signal3, signal4, signal5, signal6, signal7,
         signal8, signal9, signal10, signal11, signal12, signal13, signal14, signal15, signal16,
         Limiter1, Limiter2, Limiter3, Limiter4, Limiter5, Limiter6, Limiter7, Limiter8, Limiter9, Limiter10, Limiter11, Limiter12, type_graph, str_title):

    global time

    savedir_Psi2 = savedir + "/" + "Psi_Plot2"
    if not(os.access(savedir_Psi2,os.F_OK)):
        os.mkdir(savedir_Psi2)

    def Psi_Sub3():

        plt.gca().xaxis.set_tick_params(which='both', direction='in', bottom=True)
        plt.gca().yaxis.set_tick_params(which='both', direction='in', left=True)
        plt.ylabel("R [mm]", labelpad=15, fontsize=Fontsize)
        plt.tick_params(pad=8)
        plt.xlim(Psixmin, Psixmax)
        plt.ylim(Psiymin, Psiymax)
        plt.yticks(yticks_radius)

    if(type_graph == 'Psi'):

        fig = plt.figure(figsize=(12,22))
        
        ax1 = fig.add_subplot(611)
        ax1.set_xticklabels([])
        ax1.plot(Time, Xp1, 'red', marker='o', markersize='4', label='X-point')
        ax1.errorbar(Time, Limiter1, yerr=Limiter2, elinewidth=3, lw=0, ecolor='lightblue', label='Psi : Limiter')
        plt.title("Psi   %s    #%s" % (Operation[0], Shot[0]), fontsize=Fontsize)
        plt.legend(bbox_to_anchor=(1, 1.00), loc='upper left', labelspacing=1, fontsize=16)
        Psi_Sub3()

        ax2 = fig.add_subplot(612)
        ax2.set_xticklabels([])
        ax2.plot(Time, Xp2, 'red', marker='o', markersize='4', label='X-point')
        ax2.errorbar(Time, Limiter3, yerr=Limiter4, elinewidth=3, lw=0, ecolor='lightblue', label='Psi : Limiter')
        ax2.errorbar(Time, signal1, yerr=signal2, elinewidth=3, lw=0, color='black', label='Psi : Electrode[1]')
        plt.title("Psi   %s    #%s" % (Operation[1], Shot[1]), fontsize=Fontsize)
        plt.legend(bbox_to_anchor=(1, 1.00), loc='upper left', labelspacing=1, fontsize=16)
        Psi_Sub3()

        ax3 = fig.add_subplot(613)
        ax3.set_xticklabels([])
        ax3.plot(Time, Xp3, 'red', marker='o', markersize='4', label='X-point')
        ax3.errorbar(Time, Limiter5, yerr=Limiter6, elinewidth=3, lw=0, ecolor='lightblue', label='Psi : Limiter')
        ax3.errorbar(Time, signal3, yerr=signal4, elinewidth=3, lw=0, color='black', label='Psi : Electrode[2]')
        plt.title("Psi   %s    #%s" % (Operation[2], Shot[2]), fontsize=Fontsize)
        plt.legend(bbox_to_anchor=(1, 1.00), loc='upper left', labelspacing=1, fontsize=16)
        Psi_Sub3()

        ax4 = fig.add_subplot(614)
        ax4.set_xticklabels([])
        ax4.plot(Time, Xp4, 'red', marker='o', markersize='4', label='X-point')
        ax4.errorbar(Time, Limiter7, yerr=Limiter9, elinewidth=3, lw=0, ecolor='lightblue', label='Psi : Limiter')
        ax4.errorbar(Time, signal5, yerr=signal6, elinewidth=3, lw=0, color='black', label='Psi : Electrode[3]')
        plt.title("Psi   %s    #%s" % (Operation[3], Shot[3]), fontsize=Fontsize)
        plt.legend(bbox_to_anchor=(1, 1.00), loc='upper left', labelspacing=1, fontsize=16)
        Psi_Sub3()

        ax5 = fig.add_subplot(615)
        ax5.set_xticklabels([])
        ax5.plot(Time, Xp5, 'red', marker='o', markersize='4', label='X-point')
        ax5.errorbar(Time, Limiter9, yerr=Limiter10, elinewidth=3, lw=0, ecolor='lightblue', label='Psi : Limiter')
        ax5.errorbar(Time, signal7, yerr=signal8, elinewidth=3, lw=0, color='black', label='Psi : Electrode[4]')
        plt.title("Psi   %s    #%s" % (Operation[4], Shot[4]), fontsize=Fontsize)
        plt.legend(bbox_to_anchor=(1, 1.00), loc='upper left', labelspacing=1, fontsize=16)
        Psi_Sub3()

        ax6 = fig.add_subplot(616)
        ax6.plot(Time, Xp6, 'red', marker='o', markersize='4', label='X-point')
        ax6.errorbar(Time, Limiter11, yerr=Limiter12, elinewidth=3, lw=0, ecolor='lightblue', label='Psi : Limiter')
        ax6.errorbar(Time, signal9,  yerr=signal10, elinewidth=3, lw=0, ecolor='black', label='Psi : Electrode[1]')
        ax6.errorbar(Time, signal11, yerr=signal12, elinewidth=3, lw=0, label='Psi : Electrode[2]')
        ax6.errorbar(Time, signal13, yerr=signal14, elinewidth=3, lw=0, label='Psi : Electrode[3]')
        ax6.errorbar(Time, signal15, yerr=signal16, elinewidth=3, lw=0, label='Psi : Electrode[4]')
        plt.title("Psi   %s    #%s" % (Operation[5], Shot[5]), fontsize=Fontsize)
        plt.legend(bbox_to_anchor=(1, 1.15), loc='upper left', labelspacing=1, fontsize=16)
        Psi_Sub3()

        plt.subplots_adjust(hspace=0.3)
        fig.text(0.51, 0.0925, 'Time  [μs]', fontsize=Fontsize, ha='center', va='center')
        plt.savefig("%s/Psi.png" %(savedir_Psi2), bbox_inches='tight')
        plt.close()

    else:()

#=======================================================
#  Return of magnetic probe
#=======================================================

Xp = np.zeros((Operation_Class, Timelen))
for i in range(Operation_Class):
    for shotname in [Shotlist[i]]:
        (f1, f2, f3, sample_period, full_range1, full_range2, full_range3, precision1, precision2, precision3, init_time, fin_time, dt, time, time_PF3) = File_Data(shotname)
        Xp[i] = Magnetic_Probe()
        Xp = Xp.reshape([Operation_Class, Timelen])

#=======================================================
#  Return of Psi muitiplot (1)
#=======================================================

    Psi1(Xp[0], Dupe_Limiter_ave[0][0], Dupe_Limiter_err[0][0], "Psi", "Operation", "Shotnumber")

#=======================================================
#  Return of Psi muitiplot (2)
#=======================================================

    for i in range(4):
        Psi2(Xp[i+1], Dupe_ave[i+1][i], Dupe_err[i+1][i], Dupe_Limiter_ave[i+1][0], Dupe_Limiter_err[i+1][0], "Psi", "Operation%s" % (i+1), "Shotnumber%s" % (i+1), "Electrode%s" % (i+1))

#=======================================================
#  Return of Psi muitiplot (3)
#=======================================================

    Psi3(Xp[5], Dupe_ave[5][0], Dupe_err[5][0], Dupe_ave[5][1], Dupe_err[5][1], Dupe_ave[5][2], Dupe_err[5][2], Dupe_ave[5][3], Dupe_err[5][3], Dupe_Limiter_ave[5][0], Dupe_Limiter_err[5][0], "Psi", "Operation", "Shotnumber")

#=======================================================
#  Return of Psi muitiplot (4)
#=======================================================

    Psi4(Xp[0], Xp[1], Xp[2], Xp[3], Xp[4], Xp[5], Dupe_ave[1][0], Dupe_err[1][0], Dupe_ave[2][1], Dupe_err[2][1], Dupe_ave[3][2], Dupe_err[3][2], Dupe_ave[4][3], Dupe_err[4][3],
         Dupe_ave[5][0], Dupe_err[5][0], Dupe_ave[5][1], Dupe_err[5][1], Dupe_ave[5][2], Dupe_err[5][2], Dupe_ave[5][3], Dupe_err[5][3],
         Dupe_Limiter_ave[0][0], Dupe_Limiter_err[0][0], Dupe_Limiter_ave[1][0], Dupe_Limiter_err[1][0], Dupe_Limiter_ave[2][0], Dupe_Limiter_err[2][0],
         Dupe_Limiter_ave[3][0], Dupe_Limiter_err[3][0], Dupe_Limiter_ave[4][0], Dupe_Limiter_err[4][0], Dupe_Limiter_ave[5][0], Dupe_Limiter_err[5][0], "Psi", " ")

#=======================================================