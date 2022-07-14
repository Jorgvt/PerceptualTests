import abc

import numpy as np
import matplotlib.pyplot as plt

from .color_matrices import *
from .colored_squares import create_colored_square
from .stimuli import *

__all__ = ['PsychoTest', 
           'CrispeningAchromaticTest', 
           'CrispeningRedGreenTest', 
           'CrispeningYellowBlueTest',
           'MaskingContrastFrequencyTest',
           'ContrastSensitivityFunctionTest'
           'MaskingFixedFrequencyTest',
           'MaskingFrequencyBackgroundTest',
           'MaskingOrientationBackgroundTest']

class PsychoTest(abc.ABC):
    
    @abc.abstractmethod
    def test(self, model):
        ...
    
    @property
    @abc.abstractmethod
    def stimuli(self):
        ...
    
    def show_stimuli(self, **kwargs):
        fig, axis = plt.subplots(*self.stimuli.shape[:2], **kwargs)
        for i,ax_bg in enumerate(axis):
            for j, ax_t in enumerate(ax_bg):
                ax_t.imshow(self.stimuli[i,j])
                ax_t.axis('off')

def atd2rgb(atd):
    """
    Expects and atd color column vector.
    """
    rgb = Mxyz2ng@Matd2xyz@atd
    rgb = np.power(rgb, gamma[:,None])
    return rgb

class CrispeningAchromaticTest(PsychoTest):
    """
    Test to check the nonlinearities in the achromatic channel.
    The test is performed by creating a set of images with two colors:
    background and central.
    Its result is obtained by measuring the visibility of each image
    with respect to an image with only the background color.
    """
    def __init__(self, N, img_size, square_size):
        self.N = N
        self.img_size = img_size
        self.square_size = square_size
        self._stimuli = None

    @property
    def stimuli(self):
        if self._stimuli is None:
            atd = np.array([[a,0,0] for a in np.linspace(0.1,120,self.N)]).T
            atd_bg = np.array([[a,0,0] for a in [1,40,80,120,160]]).T
            rgb = atd2rgb(atd)
            rgb_bg = atd2rgb(atd_bg)
            self._stimuli = np.empty(shape=(len(atd_bg.T),self.N,*self.img_size,3))
            for idx_bg, bgc in enumerate(rgb_bg.T):
                colored_squares = np.empty(shape=(self.N,*self.img_size,3))
                ## Generate all the colored squares with a fixed bg_color
                for i, sqc in enumerate(rgb.T):
                    colored_square = create_colored_square(img_size=self.img_size,
                                                           square_size=self.square_size,
                                                           square_color=sqc,
                                                           bg_color=bgc)
                    colored_squares[i] = colored_square
                self._stimuli[idx_bg] = colored_squares
        return self._stimuli

    def get_readouts(self, model):
        all_readouts = np.empty(shape=self.stimuli.shape[:2])
        for i, imgs_samebg in enumerate(self.stimuli):
            outputs = model.predict(imgs_samebg)
            readouts = (outputs-outputs[0])**2
            readouts = np.sqrt(np.sum(readouts, axis=(1,2,3)))
            all_readouts[i] = readouts
        return all_readouts

    def plot_result(self, readouts):
        ## Obtain the transform-corrected xaxis
        atd = np.array([[a,0,0] for a in np.linspace(0.1,120,self.N)]).T
        atd_bg = [1,40,80,120,160]
        a_ng = (Mxyz2ng @ Matd2xyz@ atd)**(1/2)
        dist = ((a_ng.T[:-1]-a_ng.T[1:])**2).sum(axis=1)**(1/2)
        dist = np.concatenate([np.array([0]), dist])
        for i, readout in enumerate(readouts):
            if i == 0:
                color = [0,0,1]
                alpha = 1
            elif i == len(readouts)-1:
                color = [1,0,0]
                alpha = 1
            else:
                color = 'black'
                alpha = 1 - (i/(len(readouts)-1))
            plt.plot(np.cumsum(dist), readout, 'o-', color=color, alpha=alpha, label=f'{atd_bg[i]} $cd/m^2$')
        plt.xlabel(r'$|\mathrm{\vec{n}}|$ (Digital Values)')
        plt.ylabel('Visibility')
        plt.legend(title = 'Background luminance')

    def test(self, model):
        readouts = self.get_readouts(model)
        self.plot_result(readouts)

class CrispeningRedGreenTest(PsychoTest):
    """
    Test to check the nonlinearities in the Red-Green channel.
    The test is performed by creating a set of images with two colors:
    background and central.
    Its result is obtained by measuring the visibility of each image
    with respect to an image with only the background color.
    """
    def __init__(self, N, img_size, square_size):
        self.N = N
        self.img_size = img_size
        self.square_size = square_size
        self._stimuli = None

    @property
    def stimuli(self):
        if self._stimuli is None:
            atd = np.array([[60,a,0] for a in np.linspace(-20,20,self.N)]).T
            bg_idx = np.array([6, 8, 11, 14, 16])-1
            atd_bg = np.array([[60*1.8,a,0] for a in np.linspace(-20,20,self.N)[bg_idx]]).T
            rgb = atd2rgb(atd)
            rgb_bg = atd2rgb(atd_bg)
            self._stimuli = np.empty(shape=(len(atd_bg.T),self.N,*self.img_size,3))
            for idx_bg, bgc in enumerate(rgb_bg.T):
                colored_squares = np.empty(shape=(self.N,*self.img_size,3))
                ## Generate all the colored squares with a fixed bg_color
                for i, sqc in enumerate(rgb.T):
                    colored_square = create_colored_square(img_size=self.img_size,
                                                           square_size=self.square_size,
                                                           square_color=sqc,
                                                           bg_color=bgc)
                    colored_squares[i] = colored_square
                self._stimuli[idx_bg] = colored_squares
        return self._stimuli

    def get_readouts(self, model):
        bg_idx = np.array([6, 8, 11, 14, 16])-1
        all_readouts = np.empty(shape=self.stimuli.shape[:2])
        for idx_bg, (i, imgs_samebg)in zip(bg_idx, enumerate(self.stimuli)):
            outputs = model.predict(imgs_samebg)
            readouts = (outputs-outputs[idx_bg])**2
            readouts = np.sqrt(np.sum(readouts, axis=(1,2,3)))
            signs = np.ones_like(readouts)
            signs[:idx_bg] = -1
            readouts = readouts*signs
            all_readouts[i] = readouts
        return all_readouts

    def plot_result(self, readouts):
        ## Obtain the transform-corrected xaxis
        atd = np.array([[60,a,0] for a in np.linspace(-20,20,self.N)]).T
        bg_idx = np.array([6, 8, 11, 14, 16])-1
        atd_bg = np.array([[60*1.8,a,0] for a in np.linspace(-20,20,self.N)[bg_idx]]).T
        a_ng = (Mxyz2ng @ Matd2xyz@ atd)**(1/2)
        dist = ((a_ng.T[:-1]-a_ng.T[1:])**2).sum(axis=1)**(1/2)
        dist = np.concatenate([np.array([0]), dist])
        color = ['green', 'green', 'black', 'red', 'red']
        alpha = [1.0, 0.5, 1.0, 0.5, 1.0]
        for i, readout in enumerate(readouts):
            plt.plot(np.cumsum(dist), readout, 'o-', 
                     color=color[i], alpha=alpha[i], 
                     label=' ')
        plt.xlabel(r'$|\mathrm{\vec{n}}|$ (Digital Values)')
        plt.ylabel('Visibility')
        plt.legend(title = 'Background color')

    def test(self, model):
        readouts = self.get_readouts(model)
        self.plot_result(readouts)

class CrispeningYellowBlueTest(PsychoTest):
    """
    Test to check the nonlinearities in the Red-Green channel.
    The test is performed by creating a set of images with two colors:
    background and central.
    Its result is obtained by measuring the visibility of each image
    with respect to an image with only the background color.
    """
    def __init__(self, N, img_size, square_size):
        self.N = N
        self.img_size = img_size
        self.square_size = square_size
        self._stimuli = None

    @property
    def stimuli(self):
        if self._stimuli is None:
            atd = np.array([[60,0,a] for a in np.linspace(-20,20,self.N)]).T
            bg_idx = np.array([6, 8, 11, 14, 16])-1
            atd_bg = np.array([[60*1.8,0,a] for a in np.linspace(-20,20,self.N)[bg_idx]]).T
            rgb = atd2rgb(atd)
            rgb_bg = atd2rgb(atd_bg)
            self._stimuli = np.empty(shape=(len(atd_bg.T),self.N,*self.img_size,3))
            for idx_bg, bgc in enumerate(rgb_bg.T):
                colored_squares = np.empty(shape=(self.N,*self.img_size,3))
                ## Generate all the colored squares with a fixed bg_color
                for i, sqc in enumerate(rgb.T):
                    colored_square = create_colored_square(img_size=self.img_size,
                                                           square_size=self.square_size,
                                                           square_color=sqc,
                                                           bg_color=bgc)
                    colored_squares[i] = colored_square
                self._stimuli[idx_bg] = colored_squares
        return self._stimuli

    def get_readouts(self, model):
        bg_idx = np.array([6, 8, 11, 14, 16])-1
        all_readouts = np.empty(shape=self.stimuli.shape[:2])
        for idx_bg, (i, imgs_samebg)in zip(bg_idx, enumerate(self.stimuli)):
            outputs = model.predict(imgs_samebg)
            readouts = (outputs-outputs[idx_bg])**2
            readouts = np.sqrt(np.sum(readouts, axis=(1,2,3)))
            signs = np.ones_like(readouts)
            signs[:idx_bg] = -1
            readouts = readouts*signs
            all_readouts[i] = readouts
        return all_readouts

    def plot_result(self, readouts):
        ## Obtain the transform-corrected xaxis
        atd = np.array([[60,0,a] for a in np.linspace(-20,20,self.N)]).T
        bg_idx = np.array([6, 8, 11, 14, 16])-1
        atd_bg = np.array([[60*1.8,0,a] for a in np.linspace(-20,20,self.N)[bg_idx]]).T
        a_ng = (Mxyz2ng @ Matd2xyz@ atd)**(1/2)
        dist = ((a_ng.T[:-1]-a_ng.T[1:])**2).sum(axis=1)**(1/2)
        dist = np.concatenate([np.array([0]), dist])
        color = ['blue', 'blue', 'black', 'yellow', 'yellow']
        alpha = [1.0, 0.5, 1.0, 0.5, 1.0]
        for i, readout in enumerate(readouts):
            plt.plot(np.cumsum(dist), readout, 'o-', 
                     color=color[i], alpha=alpha[i], 
                     label=' ')
        plt.xlabel(r'$|\mathrm{\vec{n}}|$ (Digital Values)')
        plt.ylabel('Visibility')
        plt.legend(title = 'Background color')

    def test(self, model):
        readouts = self.get_readouts(model)
        self.plot_result(readouts)

class Masking(PsychoTest):
    """
    Class dedicated to building masking tests.
    All of these tests are based on having a test signal
    over a background and calculating the visibility with
    respect to this background.
    """
    def get_readouts(self, model):
        all_readouts = np.empty(shape=self.stimuli.shape[:2])
        for i, imgs_samebg in enumerate(self.stimuli):
            outputs = model.predict(imgs_samebg)
            readouts = (outputs-outputs[0])**2
            readouts = np.sqrt(np.sum(readouts, axis=(1,2,3)))
            all_readouts[i] = readouts
        return all_readouts

    def test(self, model):
        readouts = self.get_readouts(model)
        self.plot_result(readouts)

class MaskingContrastFrequencyTest(Masking):
    """
    Test to check the behaviour when changing the contrast
    of the test at different frequencies.
    Its result is obtained by measuring the visibility of each image
    with respect to only the background.
    """
    def __init__(self, 
                 f_tests=np.array([1.5, 3, 6, 12, 24]), 
                 c_tests=np.concatenate([[0],np.logspace(-3, np.log10(0.09), 15)]),
                 img_size=(256, 256), 
                 L0=60, 
                 fs=64,
                 color=np.array([1, 0, 0])[None,:],
                 angle=0,
                 phase=0,
                 gs=None,
                 stimuli_type='gabor'):
        self.f_tests = f_tests
        self.c_tests = c_tests
        self.fs = fs
        self.img_size = img_size
        self.num_rows, self.num_cols = img_size
        self.L0 = L0
        self._stimuli = None
        self.color = color
        self.angle = angle
        self.phase = phase
        self.gs = gs
        self.stimuli_fn = create_gabors_gs if stimuli_type=='gabor' else create_noises

    @property
    def stimuli(self):
        if self._stimuli is None:
            gabors_atd, gabors = self.stimuli_fn(f_tests = self.f_tests,
                                                 num_rows = self.num_rows, 
                                                 num_cols = self.num_cols,
                                                 num_frames = 1,
                                                 fs = self.fs,
                                                 L0 = self.L0,
                                                 c_noises = self.c_tests,
                                                 color_noise = self.color,
                                                 angle = self.angle, #rad
                                                 phase = self.phase,
                                                 gs = self.gs)
            self._stimuli = gabors
        return self._stimuli

    def plot_result(self, readouts):
        for i, readout in enumerate(readouts):
            if i == 1:
                color = [0,0,1]
                alpha = 1
            elif i == len(readouts)-1:
                color = [1,0,0]
                alpha = 1
            else:
                color = 'black'
                alpha = 1 - (i/(len(readouts)-1))
            plt.plot(self.c_tests, readout, 
                     '-', color=color, alpha=alpha, 
                     label=f'{self.f_tests[i]} cdp')
        plt.xlabel('Test Contrast')
        plt.ylabel('Visibility')
        plt.legend(title = 'Test frequency')

class ContrastSensitivityFunctionTest(MaskingContrastFrequencyTest):
    """
    Test to obtain the Contrast Sensitivity Function of a model.
    It is calculated by passing images with no stimuli (only background)
    and images with low contrast and calculating their differences.
    This would be equivalent to calculating the threshold at which the model
    stops seeing the stimuli.
    """
    def __init__(self, **kwargs):
        super(ContrastSensitivityFunctionTest, self).__init__(**kwargs)
        self._background = None
        self.c_tests = [self.c_tests] if isinstance(self.c_tests, (int, float)) else [self.c_tests[0]]

    @property
    def background(self):
        if self._background is None:
            bg_atd, bg_rgb = self.stimuli_fn(f_tests=np.array([0]),
                                             c_noises=self.c_tests)
            self._background = bg_rgb[None,:]
        return self._background

    def get_readouts(self, model):
        outputs = model.predict(self.stimuli)
        readouts = (outputs-outputs[0])**2
        readouts = np.sqrt(np.sum(readouts, axis=(1,2,3)))
        return readouts

    def plot_result(self, readouts):
        plt.plot(self.f_tests, readouts)
        plt.xlabel('Test Frequency')
        plt.ylabel('Visibility')
        plt.xscale('log')

    def show_stimuli(self, **kwargs):
        fig, axes = plt.subplots(nrows=len(self.c_tests),
                                 ncols=len(self.f_tests),
                                 **kwargs)
        for ax, img in zip(axes, self.stimuli):
            ax.imshow(img)
            ax.axis('off')

class MaskingFixedFrequencyTest(Masking):
    """
    Test to check the behaviour when changing the contrast
    of the test and the background at fixed frequencies.
    Its result is obtained by measuring the visibility of each image
    with respect to only the background.
    The difference with respect to `MaskingContrastFrequency` is that
    here the background has a pattern that masks the test pattern.
    """
    def __init__(self, 
                 f_tests=np.array([3, 12]), 
                 f_backgrounds=np.array([3, 12]),
                 c_tests=np.concatenate([[0],np.logspace(-3, np.log10(0.09), 15)]),
                 c_backgrounds=np.linspace(0, 0.25, 3),
                 img_size=(256, 256), 
                 L0=60, 
                 fs=64,
                 color=np.array([1, 0, 0])[None,:],
                 angle=0,
                 phase=0,
                 gs=None,
                 stimuli_type='gabor'):
        self.f_tests = f_tests
        self.f_backgrounds = f_backgrounds
        self.c_tests = c_tests
        self.c_backgrounds = c_backgrounds
        self.fs = fs
        self.img_size = img_size
        self.num_rows, self.num_cols = img_size
        self.L0 = L0
        self._stimuli = None
        self.color = color
        self.angle = angle
        self.phase = phase
        self.gs = gs
        self.stimuli_fn = create_gabors_gs if stimuli_type=='gabor' else create_noises

    @property
    def stimuli(self):
        if self._stimuli is None:
            gabors_atd, gabors = self.stimuli_fn(f_tests = self.f_tests,
                                                 num_rows = self.num_rows, 
                                                 num_cols = self.num_cols,
                                                 num_frames = 1,
                                                 fs = self.fs,
                                                 L0 = self.L0,
                                                 c_noises = self.c_tests,
                                                 color_noise = self.color,
                                                 angle = self.angle, #rad
                                                 phase = self.phase,
                                                 gs = self.gs)
            gabors_atd_bg, gabors_bg = self.stimuli_fn(f_tests = self.f_tests,
                                                       num_rows = self.num_rows, 
                                                       num_cols = self.num_cols,
                                                       num_frames = 1,
                                                       fs = self.fs,
                                                       L0 = self.L0,
                                                       c_noises = self.c_backgrounds,
                                                       color_noise = self.color,
                                                       angle = self.angle, #rad
                                                       phase = self.phase,
                                                       gs = None)
            gabors_atd_sum = np.empty(shape=(len(self.f_tests),len(self.c_backgrounds),len(self.c_tests),*self.img_size,3))
            for i in range(len(self.f_tests)): #freqs
                for j in range(len(self.c_backgrounds)): #bg contrast
                    for k in range(len(self.c_tests)): #test contrast
                        gabors_atd_sum[i,j,k] = gabors_atd[i,k] + gabors_atd_bg[i,j]
                        gabors_atd_sum[i,j,k] = gabors_atd_sum[i,j,k] - gabors_atd_sum[i,j,k].mean(axis=(0,1))/2
            gabors_rgb_sum = (gabors_atd_sum @ Matd2xyz.T @ Mxyz2ng.T)**(1/2)
            self._stimuli = gabors_rgb_sum
        return self._stimuli

    def show_stimuli(self, **kwargs):
        for test_freq_idx, stimuli_fixed_freq in enumerate(self.stimuli):
            fig, axis = plt.subplots(*stimuli_fixed_freq.shape[:2], **kwargs)
            for i,ax_bg in enumerate(axis):
                for j, ax_t in enumerate(ax_bg):
                    ax_t.imshow(stimuli_fixed_freq[i,j])
                    ax_t.axis('off')
            plt.suptitle(f'Test frequency {self.f_tests[test_freq_idx]} cpd')
        
    def get_readouts(self, model):
        all_readouts = np.empty(shape=self.stimuli.shape[:3])
        for i, imgs_same_f in enumerate(self.stimuli):
            for j, imgs_same_bg in enumerate(imgs_same_f):
                outputs = model.predict(imgs_same_bg)
                readouts = (outputs-outputs[0])**2
                readouts = np.sqrt(np.sum(readouts, axis=(1,2,3)))
                all_readouts[i,j] = readouts
        return all_readouts
    
    def plot_result(self, readouts):
        fig, axis = plt.subplots(1, len(self.f_tests), sharey = True)
        for test_freq_idx, stimuli_fixed_freq in enumerate(readouts): 
            for i, readout in enumerate(stimuli_fixed_freq):
                if i == 1:
                    color = [0,0,1]
                    alpha = 1
                elif i == len(stimuli_fixed_freq)-1:
                    color = [1,0,0]
                    alpha = 1
                else:
                    color = 'black'
                    alpha = 1 - (i/(len(stimuli_fixed_freq)-1))
                axis[test_freq_idx].plot(self.c_tests, readout, 
                                         '-', color=color, alpha=alpha, 
                                         label=f'{self.c_backgrounds[i]}')
            axis[test_freq_idx].set_xlabel('Test Contrast')
            axis[test_freq_idx].set_ylabel('Visibility')
            axis[test_freq_idx].legend(title = 'Background contrast')
            axis[test_freq_idx].set_title(f'Test frequency {self.f_tests[test_freq_idx]} cpd')

class MaskingFrequencyBackgroundTest(Masking):
    """
    Test to check the behaviour when changing the contrast
    of the test and the frequency of the background.
    Its result is obtained by measuring the visibility of each image
    with respect to only the background.
    The difference with respect to `MaskingContrastFrequency` is that
    here the background has a pattern that masks the test pattern.
    """
    def __init__(self, 
                 f_tests=np.array([3, 12]), 
                 f_backgrounds=np.array([0, 1.5, 3, 6, 12, 24]),
                 c_tests=np.concatenate([[0],np.logspace(-3, np.log10(0.09), 15)]),
                 c_backgrounds=np.array([0.25]),
                 img_size=(256, 256), 
                 L0=60, 
                 fs=64,
                 color=np.array([1, 0, 0])[None,:],
                 angle=0,
                 phase=0,
                 gs=None,
                 stimuli_type='gabor'):
        self.f_tests = f_tests
        self.f_backgrounds = f_backgrounds
        self.c_tests = c_tests
        self.c_backgrounds = c_backgrounds
        self.fs = fs
        self.img_size = img_size
        self.num_rows, self.num_cols = img_size
        self.L0 = L0
        self._stimuli = None
        self.color = color
        self.angle = angle
        self.phase = phase
        self.gs = gs
        self.stimuli_fn = create_gabors_gs if stimuli_type=='gabor' else create_noises

    @property
    def stimuli(self):
        if self._stimuli is None:
            gabors_atd, gabors = self.stimuli_fn(f_tests = self.f_tests,
                                                 num_rows = self.num_rows, 
                                                 num_cols = self.num_cols,
                                                 num_frames = 1,
                                                 fs = self.fs,
                                                 L0 = self.L0,
                                                 c_noises = self.c_tests,
                                                 color_noise = self.color,
                                                 angle = self.angle, #rad
                                                 phase = self.phase,
                                                 gs = self.gs)
            gabors_atd_bg, gabors_bg = self.stimuli_fn(f_tests = self.f_backgrounds,
                                                       num_rows = self.num_rows, 
                                                       num_cols = self.num_cols,
                                                       num_frames = 1,
                                                       fs = self.fs,
                                                       L0 = self.L0,
                                                       c_noises = self.c_backgrounds,
                                                       color_noise = self.color,
                                                       angle = self.angle, #rad
                                                       phase = self.phase,
                                                       gs = None)
            gabors_atd_sum = np.empty(shape=(len(self.f_tests),len(self.f_backgrounds),len(self.c_tests),*self.img_size,3))
            for i in range(len(self.f_tests)): #test freq
                for j in range(len(self.f_backgrounds)): #bg freq
                    for k in range(len(self.c_tests)): #test contrast
                        gabors_atd_sum[i,j,k] = gabors_atd[i,k] + gabors_atd_bg[j]
                        gabors_atd_sum[i,j,k] = gabors_atd_sum[i,j,k] - gabors_atd_sum[i,j,k].mean(axis=(0,1))/2
            gabors_rgb_sum = (gabors_atd_sum @ Matd2xyz.T @ Mxyz2ng.T)**(1/2)
            self._stimuli = gabors_rgb_sum
        return self._stimuli

    def show_stimuli(self, **kwargs):
        for test_freq_idx, stimuli_fixed_freq in enumerate(self.stimuli):
            fig, axis = plt.subplots(*stimuli_fixed_freq.shape[:2], **kwargs)
            for i,ax_bg in enumerate(axis):
                for j, ax_t in enumerate(ax_bg):
                    ax_t.imshow(stimuli_fixed_freq[i,j])
                    ax_t.axis('off')
            plt.suptitle(f'Test frequency {self.f_tests[test_freq_idx]} cpd')
        
    def get_readouts(self, model):
        all_readouts = np.empty(shape=self.stimuli.shape[:3])
        for i, imgs_same_f in enumerate(self.stimuli):
            for j, imgs_same_bg in enumerate(imgs_same_f):
                outputs = model.predict(imgs_same_bg)
                readouts = (outputs-outputs[0])**2
                readouts = np.sqrt(np.sum(readouts, axis=(1,2,3)))
                all_readouts[i,j] = readouts
        return all_readouts
    
    def plot_result(self, readouts):
        fig, axis = plt.subplots(1, len(self.f_tests), sharey = True)
        for test_freq_idx, stimuli_fixed_freq in enumerate(readouts): 
            for i, readout in enumerate(stimuli_fixed_freq):
                if i == np.where(self.f_backgrounds==self.f_backgrounds.min())[0].item():
                    color = [0,0,1]
                    alpha = 1
                elif i == np.where(self.f_backgrounds==self.f_tests[test_freq_idx])[0].item():
                    color = [1,0,0]
                    alpha = 1
                else:
                    color = 'black'
                    alpha = 1 - (i/(len(stimuli_fixed_freq)-1))*0.8
                axis[test_freq_idx].plot(self.c_tests, readout, 
                                         '-', color=color, alpha=alpha, 
                                         label=f'{self.f_backgrounds[i]}')
            axis[test_freq_idx].set_xlabel('Test Contrast')
            axis[test_freq_idx].set_ylabel('Visibility')
            axis[test_freq_idx].legend(title = 'Background frequency')
            axis[test_freq_idx].set_title(f'Test frequency {self.f_tests[test_freq_idx]} cpd')

class MaskingOrientationBackgroundTest(Masking):
    """
    Test to check the behaviour when changing the contrast
    of the test and the orientation of the background.
    Its result is obtained by measuring the visibility of each image
    with respect to only the background.
    The difference with respect to `MaskingContrastFrequency` is that
    here the background has a pattern that masks the test pattern.
    """
    def __init__(self, 
                 f_tests=np.array([3, 12]), 
                 f_backgrounds=np.array([3, 12]),
                 c_tests=np.concatenate([[0],np.logspace(-3, np.log10(0.09), 15)]),
                 c_backgrounds=np.array([0.25]),
                 img_size=(256, 256), 
                 L0=60, 
                 fs=64,
                 color=np.array([1, 0, 0])[None,:],
                 angle=0,
                 angle_backgrounds=np.linspace(0, 180, 8),
                 phase=0,
                 gs=None,
                 stimuli_type='gabor'):
        self.f_tests = f_tests
        self.f_backgrounds = f_backgrounds
        self.c_tests = c_tests
        self.c_backgrounds = c_backgrounds
        self.fs = fs
        self.img_size = img_size
        self.num_rows, self.num_cols = img_size
        self.L0 = L0
        self._stimuli = None
        self.color = color
        self.angle = angle
        self.angle_backgrounds = angle_backgrounds
        self.phase = phase
        self.gs = gs
        self.stimuli_fn = create_gabors_gs if stimuli_type=='gabor' else create_noises

    @property
    def stimuli(self):
        if self._stimuli is None:
            gabors_atd, gabors = self.stimuli_fn(f_tests = self.f_tests,
                                                 num_rows = self.num_rows, 
                                                 num_cols = self.num_cols,
                                                 num_frames = 1,
                                                 fs = self.fs,
                                                 L0 = self.L0,
                                                 c_noises = self.c_tests,
                                                 color_noise = self.color,
                                                 angle = self.angle, #rad
                                                 phase = self.phase,
                                                 gs = self.gs)

            gabors_atd_bg = np.empty(shape=(len(self.f_backgrounds), len(self.angle_backgrounds), self.num_rows, self.num_cols, 3))
            gabors_bg = np.empty(shape=(len(self.f_backgrounds), len(self.angle_backgrounds), self.num_rows, self.num_cols, 3))
            for i in range(len(self.angle_backgrounds)):
                gabor_atd_bg, gabor_bg = create_gabors_gs(f_tests = self.f_backgrounds,
                                                          num_rows = self.num_rows, 
                                                          num_cols = self.num_cols,
                                                          num_frames = 1,
                                                          fs = self.fs,
                                                          L0 = self.L0,
                                                          c_noises = self.c_backgrounds,
                                                          color_noise = np.array([1, 0, 0])[None,:],
                                                          angle = self.angle_backgrounds[i]*np.pi/180, #rad
                                                          phase = np.random.uniform(0, 2*np.pi),
                                                          gs = None)
                gabors_atd_bg[:,i,:,:,:] = gabor_atd_bg
                gabors_bg[:,i,:,:,:] = gabor_bg

            gabors_atd_sum = np.empty(shape=(len(self.f_tests),len(self.angle_backgrounds),len(self.c_tests),*self.img_size,3))
            for i in range(len(self.f_tests)): #test freq
                for j in range(len(self.angle_backgrounds)): #bg angle
                    for k in range(len(self.c_tests)): #test contrast
                        gabors_atd_sum[i,j,k] = gabors_atd[i,k] + gabors_atd_bg[i,j]
                        gabors_atd_sum[i,j,k] = gabors_atd_sum[i,j,k] - gabors_atd_sum[i,j,k].mean(axis=(0,1))/2
            gabors_rgb_sum = (gabors_atd_sum @ Matd2xyz.T @ Mxyz2ng.T)**(1/2)

            gabors_atd_sum_0 = np.empty(shape=(len(self.f_tests),1,len(self.c_tests),*self.img_size,3))
            for i in range(len(self.f_tests)): #test freqs
                for j in range(1): #bg angle
                    for k in range(len(self.c_tests)): #test contrast
                        gabors_atd_sum_0[i,j,k] = gabors_atd[i,k]
                        gabors_atd_sum_0[i,j,k] = gabors_atd_sum_0[i,j,k] 
            gabors_rgb_sum_0 = (gabors_atd_sum_0 @ Matd2xyz.T @ Mxyz2ng.T)**(1/2)
            gabors_rgb_sum = np.concatenate([gabors_rgb_sum_0, gabors_rgb_sum], axis=1)
            self._stimuli = gabors_rgb_sum
        return self._stimuli

    def show_stimuli(self, **kwargs):
        for test_freq_idx, stimuli_fixed_freq in enumerate(self.stimuli):
            fig, axis = plt.subplots(*stimuli_fixed_freq.shape[:2], **kwargs)
            for i,ax_bg in enumerate(axis):
                for j, ax_t in enumerate(ax_bg):
                    ax_t.imshow(stimuli_fixed_freq[i,j])
                    ax_t.axis('off')
            plt.suptitle(f'Test frequency {self.f_tests[test_freq_idx]} cpd')
        
    def get_readouts(self, model):
        all_readouts = np.empty(shape=self.stimuli.shape[:3])
        for i, imgs_same_f in enumerate(self.stimuli):
            for j, imgs_same_bg in enumerate(imgs_same_f):
                outputs = model.predict(imgs_same_bg)
                readouts = (outputs-outputs[0])**2
                readouts = np.sqrt(np.sum(readouts, axis=(1,2,3)))
                all_readouts[i,j] = readouts
        return all_readouts
    
    def plot_result(self, readouts):
        fig, axis = plt.subplots(1, len(self.f_tests), sharey = True)
        for test_freq_idx, stimuli_fixed_freq in enumerate(readouts): 
            for i, readout in enumerate(stimuli_fixed_freq,-1):
                if i == -1:
                    color = [0,0,1]
                    alpha = 1
                    label = 'No background'
                elif i == np.where(self.angle_backgrounds==self.angle)[0].item():
                    color = [1,0,0]
                    alpha = 1
                    label=f'{self.angle_backgrounds[i]:.0f} deg'
                else:
                    color = 'black'
                    alpha = 1 - (i/(len(stimuli_fixed_freq)-1))*0.8
                    label=f'{self.angle_backgrounds[i]:.0f} deg'
                axis[test_freq_idx].plot(self.c_tests, readout, 
                                         '-', color=color, alpha=alpha, 
                                         label=label)
            axis[test_freq_idx].set_xlabel('Test Contrast')
            axis[test_freq_idx].set_ylabel('Visibility')
            axis[test_freq_idx].legend(title = 'Background angle')
            axis[test_freq_idx].set_title(f'Test frequency {self.f_tests[test_freq_idx]} cpd')