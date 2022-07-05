import abc

import numpy as np
import matplotlib.pyplot as plt

from .color_matrices import *
from .colored_squares import create_colored_square

__all__ = ['PsychoTest', 'CrispeningAchromaticTest', 'CrispeningRedGreenTest', 'CrispeningYellowBlueTest']

class PsychoTest(abc.ABC):
    
    @abc.abstractmethod
    def test(self, model):
        ...
    
    @abc.abstractmethod
    def show_stimuli(self):
        ...

    @property
    @abc.abstractmethod
    def stimuli(self):
        ...

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

    def show_stimuli(self, **kwargs):
        fig, axis = plt.subplots(*self.stimuli.shape[:2], **kwargs)
        for i,ax_bg in enumerate(axis):
            for j, ax_t in enumerate(ax_bg):
                ax_t.imshow(self.stimuli[i,j])
                ax_t.axis('off')

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

    def show_stimuli(self, **kwargs):
        fig, axis = plt.subplots(*self.stimuli.shape[:2], **kwargs)
        for i,ax_bg in enumerate(axis):
            for j, ax_t in enumerate(ax_bg):
                ax_t.imshow(self.stimuli[i,j])
                ax_t.axis('off')

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

    def show_stimuli(self, **kwargs):
        fig, axis = plt.subplots(*self.stimuli.shape[:2], **kwargs)
        for i,ax_bg in enumerate(axis):
            for j, ax_t in enumerate(ax_bg):
                ax_t.imshow(self.stimuli[i,j])
                ax_t.axis('off')

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