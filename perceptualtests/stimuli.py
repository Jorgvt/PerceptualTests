import numpy as np
from .color_matrices import *
from .utils import *

__all__ = ['create_gabors_gs',
           'create_noises']

def create_gabors_gs(f_tests = np.array([1, 2, 4, 8, 16, 30]),
                     num_rows = 256, 
                     num_cols = 256,
                     num_frames = 1,
                     fs = 64,
                     L0 = 40,
                     c_noises = [0.1],
                     color_noise = np.array([1, 0, 0])[None,:],
                     delta_angle = 180,
                     angle = 0, #rad
                     phase = 0,
                     gs = None,
                     **kwargs):
                     
    [x,y,t,fx,fy,ft] = spatio_temp_freq_domain(num_rows, num_cols, num_frames, fs, fs, num_frames)
    x0 = np.max(x)/2
    y0 = np.max(y)/2

    if gs is None:
        gs = np.ones_like(x)
    elif callable(gs):
        gs = gs(x, y, **kwargs)

    sigma = num_rows/(fs*4)
    delta_f = f_tests/2

    gabors_atd = np.empty(shape=(len(f_tests),len(c_noises),num_rows,num_cols,3))
    gabors = np.empty(shape=(len(f_tests),len(c_noises),num_rows,num_cols,3))
    for l, f_test in enumerate(f_tests):
        fm = f_test - delta_f[l]
        fM = f_test + delta_f[l]
        nx = np.sin(2*np.pi*(f_test*np.cos(angle)*x+f_test*np.sin(angle)*y) + phase)
        nx = nx*gs + L0
        
        for j, c_noise in enumerate(c_noises):
            nx2 = control_lum_contrast(nx + L0, L0, c_noise) - L0
            nx2 = nx2[:,:,None]@color_noise
            nx2[:,:,0] = nx2[:,:,0] + L0
            gabors_atd[l,j,:,:,:] = nx2
            nx2 = nx2@Matd2xyz.T@Mxyz2ng.T
            if nx2.min()<0:
                print(f_test, (nx2<0).sum())
            nx2 = np.sqrt(nx2)
            gabors[l,j,:,:,:] = nx2
    
    return gabors_atd.squeeze(), gabors.squeeze()

def create_noises(f_tests = np.array([1, 2, 4, 8, 16, 30]),
                  num_rows = 256, 
                  num_cols = 256,
                  num_frames = 1,
                  fs = 64,
                  L0 = 40,
                  c_noises = [0.1],
                  color_noise = np.array([1, 0, 0])[None,:],
                  delta_angle = 180,
                  angle = 0,
                  phase = 0,
                  gs = None,
                  **kwargs):

    [x,y,t,fx,fy,ft] = spatio_temp_freq_domain(num_rows, num_cols, num_frames, fs, fs, num_frames)
    x0 = np.max(x)/2
    y0 = np.max(y)/2

    if gs is None:
        gs = np.ones_like(x)
    elif callable(gs):
        gs = gs(x, y, **kwargs)

    delta_f = f_tests/4

    noises_atd = np.empty(shape=(len(f_tests),len(c_noises),num_rows,num_cols,3))
    noises = np.empty(shape=(len(f_tests),len(c_noises),num_rows,num_cols,3))
    for l, f_test in enumerate(f_tests):
        fm = f_test - delta_f[l]
        fM = f_test + delta_f[l]
        nx, nf, F_noise = noise(fx, fy, fm, fM, angle, delta_angle)
        nx = nx*gs
        
        for j, c_noise in enumerate(c_noises):
            nx2 = control_lum_contrast(nx + L0, L0, c_noise) - L0
            nx2 = nx2[:,:,None]@color_noise
            nx2[:,:,0] = nx2[:,:,0] + L0
            noises_atd[l,j,:,:,:] = nx2
            nx2 = nx2@Matd2xyz.T@Mxyz2ng.T
            nx2 = np.sqrt(nx2)
            noises[l,j,:,:,:] = nx2
    return noises_atd.squeeze(), noises.squeeze()