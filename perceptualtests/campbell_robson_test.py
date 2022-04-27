import numpy as np
from .color_matrices import *
from .utils import *

def campell_robson_test(N1=300, 
                        N2=750, 
                        fs=300,
                        L0=40,
                        Cmax=0.2,
                        CmaxT=0.2,
                        CmaxD=0.2,
                        expo=2.5,
                        C0=0,
                        ini=1,
                        angle=0,
                        delta_angle=25,
                        C_noise=0.15):

    # 1-degree domain (for each separate sinusoid)
    [x1,y1,t1,fx1,fy1,ft1] = spatio_temp_freq_domain(N2, N1, 1, fs, fs, 1)

    # 2.5-degree domain (for the whole test made from 0.5-deg pieces)
    [x2,y2,t2,fx2,fy2,ft2] = spatio_temp_freq_domain(N2, N2, 1, fs, fs, 1)

    # Contrast ramp  (initial contrast and slope -prepared for expo ramp-)
    kc = Cmax/np.max(y1**expo)

    # Frequencies
    freqs = np.array([2, 4, 8, 16, 32, 64])
    sins = (C0 + kc*y1**expo)*np.sin(2*np.pi*freqs[:-1,None,None]*x1)

    n_col = int(N1/2)
    # S = np.concatenate((sin1[:,0:n_col],sin2[:,0:n_col],sin3[:,0:n_col],sin4[:,0:n_col],sin5[:,0:n_col],), axis=1)
    S = np.concatenate(sins[:,:,0:n_col], axis=-1)

    S = control_lum_contrast(S,L0,Cmax)
    S = S - L0
    s = S.shape
    S_lum = np.zeros((s[0],s[1],3))
    S_lum[:,:,0] = S + L0
    S_lum[:,:,1] = 0*S
    S_lum[:,:,2] = 0*S

    S = control_lum_contrast(S,L0,CmaxT)
    S = S - L0
    S_T = np.zeros((s[0],s[1],3))
    S_T[:,:,0] = L0*(S - S + 1)
    S_T[:,:,1] = S
    S_T[:,:,2] = 0*S

    S = control_lum_contrast(S,L0,CmaxD)
    S = S - L0
    S_D = np.zeros((s[0],s[1],3))
    S_D[:,:,0] = L0*(S - S + 1)
    S_D[:,:,1] = 0*S
    S_D[:,:,2] = S

    #  Parameters of the noise
    #######################################

    # Frequency (one octave width)
    fm = freqs[ini]-(freqs[ini+1]-freqs[ini])/2
    fM = freqs[ini]+(freqs[ini+1]-freqs[ini])/2
    print(fm,fM)
    # Color weights in ATD channels
    color_noise = np.array([1, 0, 0])
    nx, nf, F_noise = noise(fx2,fy2,fm,fM,angle,delta_angle)
    nx = control_lum_contrast(nx,L0,C_noise)-L0

    S_lum[:,:,0] = S_lum[:,:,0] + color_noise[0]*nx
    S_lum[:,:,1] = S_lum[:,:,1] + color_noise[1]*nx
    S_lum[:,:,2] = S_lum[:,:,2] + color_noise[2]*nx

    S_T[:,:,0] = S_T[:,:,0] + color_noise[0]*nx
    S_T[:,:,1] = S_T[:,:,1] + color_noise[1]*nx
    S_T[:,:,2] = S_T[:,:,2] + color_noise[2]*nx

    S_D[:,:,0] = S_D[:,:,0] + color_noise[0]*nx
    S_D[:,:,1] = S_D[:,:,1] + color_noise[1]*nx
    S_D[:,:,2] = S_D[:,:,2] + color_noise[2]*nx

    S_lum_rgb = np.reshape(  np.sqrt(Mxyz2ng @ Matd2xyz @ np.reshape(S_lum,(s[0]*s[1],3), order='F').T).T , (s[0],s[1],3), order='F' )
    S_T_rgb = np.reshape(  np.sqrt(Mxyz2ng @ Matd2xyz @ np.reshape(S_T,(s[0]*s[1],3), order='F').T).T , (s[0],s[1],3), order='F' )
    S_D_rgb = np.reshape(  np.sqrt(Mxyz2ng @ Matd2xyz @ np.reshape(S_D,(s[0]*s[1],3), order='F').T).T , (s[0],s[1],3), order='F' )

    return S_lum_rgb, S_T_rgb, S_D_rgb