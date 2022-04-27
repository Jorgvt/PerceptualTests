import numpy as np
from .color_matrices import *
from .utils import *

def gabor_noise(num_rows = 300,
                num_cols = 300,
                num_frames = 1,
                f_test = 4,
                c_test = 0.15,
                fs = 300,
                L0 = 40,
                sigma = 0.25,
                delta_angle = 30,
                angle = 0,
                delta_f = 2,
                c_noise = 0.1):

    # XY domain
    [x,y,t,fx,fy,ft] = spatio_temp_freq_domain(num_rows, num_cols, num_frames, fs, fs, num_frames)
    x0, y0 = np.max(x)/2, np.max(y)/2

    # Gabor stimulus
    test = np.zeros((num_rows, num_cols, 3))
    color_test = np.array([1, 0, 0])
    g = np.exp(-((x-x0)**2)/sigma**2-((y-y0)**2)/sigma**2)*np.sin(2*np.pi*f_test*x)
    gg = control_lum_contrast(g + L0, L0, c_test) - L0
    test[:,:,0] = L0 + color_test[0]*gg
    test[:,:,1] = color_test[1]*gg
    test[:,:,2] = color_test[2]*gg
    
    # Noise
    color_noise = np.array([1, 0, 0])
    fm = f_test - delta_f
    fM = f_test + delta_f
    nx, nf, F_noise = noise(fx, fy, fm, fM, angle, delta_angle)
    nx2 = control_lum_contrast(nx + L0, L0, c_noise) - L0

    # Add noise + stimulus
    stimulus = np.zeros((num_rows, num_cols, 3))
    stimulus[:,:,0] = test[:,:,0] + color_noise[0]*nx2
    stimulus[:,:,1] = test[:,:,1] + color_noise[1]*nx2
    stimulus[:,:,2] = test[:,:,2] + color_noise[2]*nx2

    stim_rgb = np.reshape(np.sqrt(Mxyz2ng @ Matd2xyz @ np.reshape(stimulus,(num_rows*num_cols,3), order = 'F').T).T, (num_rows,num_cols,3), order = 'F')

    return stim_rgb