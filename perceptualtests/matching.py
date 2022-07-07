from abc import abstractmethod
from tqdm.auto import tqdm

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import scipy

from .color_matrices import *

__all__ = ['ColorMatching',
           'ColorMatchingInstance']

def interpolate_Ti_tf(lambdas):
    T_1 = T_lambda[:,1]
    T_1 = tfp.math.interp_regular_1d_grid(lambdas, T_lambda[0,0], T_lambda[-1,0], 
                                            T_1, fill_value='extrapolate')
    T_2 = T_lambda[:,2]
    T_2 = tfp.math.interp_regular_1d_grid(lambdas, T_lambda[0,0], T_lambda[-1,0], 
                                            T_2, fill_value='extrapolate')
    T_3 = T_lambda[:,3]
    T_3 = tfp.math.interp_regular_1d_grid(lambdas, T_lambda[0,0], T_lambda[-1,0], 
                                            T_3, fill_value='extrapolate')
    return T_1, T_2, T_3

def monochomatic_stimulus_2_tf(central_lambda, lambdas, width, max_radiance, background):
    """
    Generates a quasi-monochromatic spectrum. Gaussian centered in central lambda with desired width and peak on max_radiance over a background.
    """
    spectrum = max_radiance*tf.cast(tf.exp(-(lambdas-central_lambda)**2/width**2), tf.float32) + background_stimulus_tf(lambdas, background)
    return spectrum

def background_stimulus_tf(lambdas, background):
    """
    Generates a quasi-monochromatic spectrum. Gaussian centered in central lambda with desired width and peak on max_radiance over a background.
    """
    spectrum = tf.cast(tf.ones_like(lambdas, dtype=tf.float32)*background, tf.float32)
    return spectrum

def xyz2nrgb_tf(xyz, gamma, clip=False):
    """
    Changes from xyz to nrgb values (ready to be displayed in the screen). Expects and xyz color column vector or images.
    This function is TensorFlow's gradient friendly.

    Parameters
    ----------
    xyz: np.array
        Three-element array corresponding to a color in XYZ space.
    gamma: int or List
        Value(s) to be used in the transformation to digital values.
        The final result will be rgb = ng**(1/gamma).
    clip: bool
        Boolean determining if clipping will be used or not when the
        resultant values exceed the range [0,1].
    """
    rgb = Mxyz2ng @ xyz

    if clip and (tf.reduce_min(rgb) < 0 or tf.reduce_max(rgb) > 1):
        rgb = tf.clip_by_value(rgb, 0, 1)

    if type(gamma) == list:
        rgb = tf.pow(rgb, 1/gamma[:,None])
    elif type(gamma) == float:
        rgb = rgb**(1/gamma)
    else:
        raise TypeError('Gamma should be either a single value (float) or a list with 3.')
    
    return rgb

class ColorMatching():
    """
    Color matching experiment where different monochromatic lights
    are given and the user has to minimize the distance with respect
    to a white image.
    This process is performed in an iterative way by optimizing the
    ammount of each color that is put into the images from a set
    of four available wavelenghts.
    """
    
    def __init__(self,
                 wavelengths=np.linspace(380, 770, 50),
                 central_wavelengths=[475.0, 500.0, 580.0, 700.0],
                 max_radiance=1.5e-3,
                 background_radiance=0.5e-4,
                 norm_grads=False):
        self.wavelengths = wavelengths
        self.central_wavelengths = central_wavelengths
        self.max_radiance = max_radiance
        self.background_radiance = background_radiance
        self.norm_grads = norm_grads

    def fit(self, model, epochs):
        pass

class ColorMatchingInstance():
    """
    A color matching experiment instance, where instance means
    only a single wavelength. A full color matching experiment
    is compromised of a full run over a collection of wavelengths.
    """
    def __init__(self,
                 wavelength,
                 central_wavelengths,
                 lambdas,
                 max_radiance,
                 background_radiance,
                 img_size,
                 space_transform_fn,
                 initial_weights=[0.001, 0.001, 0.001, 0.001],
                 norm_grads=False):
        self.wavelength = wavelength
        self.central_wavelengths = central_wavelengths
        self.lambdas = lambdas
        self.max_radiance = max_radiance
        self.background_radiance = background_radiance
        self.img_size = img_size
        self.space_transform_fn = space_transform_fn
        self.norm_grads = norm_grads
        self.weights = tf.Variable(initial_weights,
                                   trainable=True,
                                   dtype=tf.float32)
        self._Ts = None
        self.loss = None
        self.optimizer = None
    @property
    def Ts(self):
        if self._Ts is None:
            self._Ts = interpolate_Ti_tf(self.lambdas)
        return self._Ts

    def compile(self, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer

    def generate_images(self):
        spectrum_lambda = monochomatic_stimulus_2_tf(self.wavelength, self.lambdas, width=10, max_radiance=0.75*2*self.max_radiance, background=5*self.background_radiance)
        Y_l = km*tf.reduce_sum(self.Ts[1]*spectrum_lambda)
        spectrum_white = tf.cast(tf.ones_like(self.lambdas, dtype=tf.float32)*self.background_radiance*10, tf.float32)
        Y_w = km*tf.reduce_sum(self.Ts[1]*spectrum_white)
        spectrum_white = spectrum_white*Y_l/Y_w

        for i in range(len(self.central_wavelengths)):
            spectrum = monochomatic_stimulus_2_tf(self.central_wavelengths[i], self.lambdas, width=5, max_radiance=0.75*self.max_radiance*tf.abs(self.weights[i]), background=self.background_radiance)
            if self.weights[i] >= 0:
                spectrum_lambda = spectrum_lambda + spectrum
            else:
                spectrum_white = spectrum_white + spectrum

        t1_l = km*tf.reduce_sum(self.Ts[0]*spectrum_lambda)
        t2_l = km*tf.reduce_sum(self.Ts[1]*spectrum_lambda)
        t3_l = km*tf.reduce_sum(self.Ts[2]*spectrum_lambda)
        t1_w = km*tf.reduce_sum(self.Ts[0]*spectrum_white)
        t2_w = km*tf.reduce_sum(self.Ts[1]*spectrum_white)
        t3_w = km*tf.reduce_sum(self.Ts[2]*spectrum_white)

        t_l = tf.convert_to_tensor([t1_l, t2_l, t3_l], dtype = tf.float32)[:,None]
        t_w = tf.convert_to_tensor([t1_w, t2_w, t3_w], dtype = tf.float32)[:,None]

        rgb_lambda = self.space_transform_fn(t_l)
        rgb_white = self.space_transform_fn(t_w)

        img_lambda = tf.ones((1,*self.img_size,3), dtype=tf.float32)*tf.cast(tf.transpose(rgb_lambda, perm=[1,0]), tf.float32)
        img_white = tf.ones((1,*self.img_size,3), dtype=tf.float32)*tf.cast(tf.transpose(rgb_white, perm=[1,0]), tf.float32)

        return img_lambda, img_white#, rgb_2_l, rgb_2_w

    def fit(self, model, epochs, verbose=True, use_tqdm=True):
        history = {'Loss':[], 'GradsL2':[]}
        pbar = tqdm(range(epochs)) if use_tqdm else range(epochs)
        for epoch in pbar:
            with tf.GradientTape() as tape:
                img_l, img_w = self.generate_images()
                imgs = tf.concat([img_l, img_w], axis=0)
                response_l, response_w = model.predict(imgs)
                response_l, response_w = response_l[None,:,:,:], response_w[None,:,:,:]
                loss = self.loss(response_l, response_w)
            
            grads = tape.gradient(loss, self.weights)
            self.optimizer.apply_gradients(zip([grads], [self.weights]))
            history['Loss'].append(loss.numpy().item())
            history['GradsL2'].append(tf.reduce_sum(grads**2).numpy().item())
            if verbose and not use_tqdm:
                print(f'Epoch {epoch+1} -> Loss: {history["Loss"][-1]} | GradsL2: {history["GradsL2"][-1]}')
        return history
