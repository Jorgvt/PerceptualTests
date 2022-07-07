import numpy as np
import torch
import math
import cmath

def spatio_temp_freq_domain(Ny, Nx, Nt, fsx, fsy, fst):
    int_x = Nx/fsx
    int_y = Ny/fsy
    int_t = Nt/fst

    x = torch.zeros(Ny, Nx*Nt)
    y = torch.zeros(Ny, Nx*Nt)
    t = torch.zeros(Ny, Nx*Nt)

    fot_x = torch.linspace(0, int_x, Nx+1)
    fot_x = fot_x[:-1]
    fot_x = fot_x.repeat(Ny, 1)

    fot_y = torch.linspace(0, int_y, Ny+1)
    fot_y = fot_y[:-1]
    fot_y = fot_y.repeat(Nx, 1).T

    fot_t = torch.ones(Ny, Nx)

    val_t = torch.linspace(0, int_t, Nt+1)
    val_t = val_t[:-1]

    for i in range(Nt):
        x = metefot(x, fot_x, i+1, 1)
        y = metefot(y, fot_y, i+1, 1)
        t = metefot(t, val_t[i]*fot_t, i+1, 1)

    [fx, fy] = freqspace([Ny, Nx])

    fx = fx*fsx/2
    fy = fy*fsy/2

    ffx = torch.zeros(Ny, Nx*Nt)
    ffy = torch.zeros(Ny, Nx*Nt)
    ff_t = torch.zeros(Ny, Nx*Nt)

    fot_fx = fx
    fot_fy = fy
    fot_t = torch.ones(Ny, Nx)

    [ft, ft2] = freqspace([Nt, Nt])
    val_t = ft*fst/2

    for i in range(Nt):
        ffx = metefot(ffx, fot_fx, i+1, 1)
        ffy = metefot(ffy, fot_fy, i+1, 1)
        ff_t = metefot(ff_t, val_t[i]*fot_t, i+1, 1)

    return np.array(x), np.array(y), np.array(t), np.array(ffx), np.array(ffy), np.array(ff_t)


def metefot(sec, foto, N, ma):
    ss = foto.size()
    fil = ss[0]
    col = ss[1]
    s = sec.size()
    Nfot = s[1] / col

    if N > Nfot:
        sec = [sec, foto]
    else:
        if ma == 1:
            sec[:, (N-1)*col:N*col] = foto
    # if incorrect results finish this function.
    return sec


def freqspace(N):
    # Returns 2-d frequency range vectors for N[0] x N[1] matrix

    f1 = (torch.arange(0, N[0], 1)-math.floor(N[0]/2))*(2/N[0])
    f2 = (torch.arange(0, N[1], 1)-math.floor(N[1]/2))*(2/N[1])
    F2, F1 = torch.meshgrid([f1, f2])
    return F1, F2


def control_lum_contrast(image, L, C):
    # CONTROL_LUM_CONTRAST sets the average luminance and RMSE sinus-like contrast for a natural image 

    img_mean = np.mean(image)
    img_std = np.std(image)*np.sqrt(2)
    if img_std == 0:
        img_std = 1
    new_image = (image - img_mean)/img_std
    new_image = L + C*L*new_image

    return new_image

def noise(fx2,fy2,fm,fM,angle,delta_a):

    # Noise in frequency domain
    #fm = 10
    #fM = 20
    #angle = 90
    a_m = -delta_a/2
    a_M = delta_a/2

    f = np.sqrt(fx2**2+fy2**2)
    a = 180*np.arctan2(fy2,fx2)/np.pi

    #print(np.min(a),np.max(a))

    F_noise_f = 1*((f>fm) & (f<fM))
    F_noise_a = 1*(((a>a_m+angle) & (a<a_M+angle)) | ((a>a_m+angle+180) & (a<a_M+angle+180)) | ((a>a_m+angle-180) & (a<a_M+angle-180)) )   

    F_noise = F_noise_f*F_noise_a

    #plt.figure(figsize = (20, 20))
    #plt.imshow(F_noise, cmap = 'gray')
    #plt.axis('off')
    #plt.show()

    # G = np.exp(-((fx-fx0)**2/sigmas_f[k]**2 +(fy-fy0)**2/sigmas_f[k]**2))
    nf = F_noise*np.exp(cmath.sqrt(-1)*(2*np.pi*np.random.rand(fx2.shape[0], fx2.shape[1])))
    nx = np.fft.ifft2(np.fft.ifftshift(nf)).real

    return nx, nf, F_noise  