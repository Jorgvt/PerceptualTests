import numpy as np
import scipy.io as sio
import os

__all__ = ['Mxyz2atd',
           'Matd2xyz',
           'Mng2xyz',
           'Mxyz2ng',
           'Mlms2xyz',
           'Mxyz2lms',
           'gamma',
           'T_lambda',
           'km']

Mng2xyz = np.array([[69.1661, 52.4902, 46.6052],
                    [39.0454, 115.8404, 16.3118],
                    [3.3467, 12.6700, 170.1090]])
 
Mxyz2lms = np.array([[0.2434, 0.8524, -0.0516],
                    [-0.3954, 1.1642, 0.0837],
                    [0, 0, 0.6225]])  

Mlms2xyz = np.linalg.inv(Mxyz2lms)
Mxyz2ng = np.linalg.inv(Mng2xyz)

# Jameson and Hurvich
Mxyz2atd_jh = np.array([[0, 1, 0],
                    [1, -1, 0],
                    [0, 0.4, -0.4]])
Matd2xyz_jh = np.linalg.inv(Mxyz2atd_jh)

# Ingling and tsou
Mxyz2atd_it = np.array([[-0.0121 ,  0.9771 ,   0.0025],
                    [0.9247,   -0.8398 ,   0.0532],
                    [0.0169 ,   0.3268 ,  -0.4393]])
Matd2xyz_it = np.linalg.inv(Mxyz2atd_it)

# J&H with scaled achromatic (*alpha)
alpha = 2
Matd2xyz = Matd2xyz_jh
# Matd2xyz[:,0] = alpha*Matd2xyz_jh[:,0]
Mxyz2atd = np.linalg.inv(Matd2xyz)

gamma = np.array([1/2.2, 1/2.2, 1/2.1])

path_T_lambda = os.path.dirname(__file__)
path_T_lambda = os.path.join(path_T_lambda, 'data/cmf_ciexyz.mat')
T_lambda = sio.loadmat(path_T_lambda)['T_lambda'].astype(np.float32)
km = 683 #lumens/watt