import numpy as np
from .color_matrices import *
from .utils import *

def create_square_mask(img_size, square_size):
    """
    Returns a boolean squared-shaped mask of square_size size inside 
    an image of img_size size.

    Parameters
    ----------
    img_size: tuple(int, int)
    square_size: tuple(int, int)

    Returns
    -------
    square_mask: array[bool]
        Boolean square-shaped mask of square_size size inside an
        image of img_size size.
    """

    x, y = np.meshgrid(range(img_size[0]), range(img_size[1]))
    square_mask = (x>=(img_size[0]//2 - square_size[0]//2)) &\
                  (x <(img_size[0]//2 + square_size[0]//2)) &\
                  (y>=(img_size[1]//2 - square_size[1]//2)) &\
                  (y <(img_size[1]//2 + square_size[1]//2))
    return square_mask.T

def create_colored_square(img_size,
                          square_size,
                          square_color,
                          bg_color):

    """
    Returns an image_size image with bg_color as background
    color and a square_size square of square_color in the center.

    Parameters
    ----------
    img_size: tuple(int, int)
        The output image will be (img_size, img_size).
    square_size: tuple(int, int)
        The center square will be (square_size, square_size).
    square_color: list[int]
        Color of the square.
    bg_color: list[int]
        Color of the background.

    Returns
    -------
    image: array
        Image of size img_size with a center square of size square_size
        with (potentially) different background and square colors.
    """
    
    image = np.ones(shape=img_size)
    image = np.ones(shape=(*img_size, 3))
    image = image*bg_color
    mask_square = create_square_mask(img_size=img_size, square_size=square_size)
    image[mask_square] = square_color
    return image