from tqdm.auto import tqdm

import numpy as np


def obtain_receptive_field(model, img_height, img_width, channels, fov=32, channel_to_keep=0):
    """
    This function calculates the receptive field of a model by building a set of deltas and
    calculating the response of the model for each delta. As of now, the calculations are
    made only for the center neuron using a predetermined FOV: only a range [-FOV, FOV] centered
    on the center neuron is used.

    Parameters
    ----------
    model: 
        Model we want to calculate the receptive field from.
    img_height:
        Height of the images that the model accepts.
    img_width:
        Width of the images that the model accepts.
    channels:
        Number of channels of the images that the model accepts.
    fov:
        Field Of View as explained before.
    channel_to_keep:
        Channel to keep when keeping the response.
    
    Returns
    -------
    receptive_field:
        Array containing the receptive field.
    """
    ## Define deltas and obtain their responses
    responses = []
    for i in tqdm(range(img_height//2-fov, img_height//2+fov), desc='outer'):
        for j in range(img_width//2-fov, img_width//2+fov):
            for k in range(channels):
                zero = np.zeros((img_height, img_width, channels))
                zero[i,j,k] = 1
                response = model.predict(zero[None,:,:,:])
                ## Now we keep only the center response of a specific channel
                response_height, response_width = response.shape[1:3] # [BatchDim, H, W, C]
                responses.append(response[0,response_height//2,response_width//2,channel_to_keep])
    
    ## Turn the list into an array
    receptive_field = np.array(responses)
    ## Reshape so it has image-like shape and is plotable
    receptive_field = receptive_field.reshape((fov*2, fov*2, channels))

    return receptive_field

def obtain_receptive_field_gen(img_height, img_width, channels, fov=32, bg_gain=0.5, delta_gain=0.05):
    """
    This generator yields the deltas needed to calculate the receptive
    field of a model.

    Parameters
    ----------
    img_height:
        Height of the images that the model accepts.
    img_width:
        Width of the images that the model accepts.
    channels:
        Number of channels of the images that the model accepts.
    fov:
        Field Of View as explained before.
    bg_gain:
        Intensity of the background.
    delta_gain:
        Value of the added delta.
    
    Returns
    -------
    receptive_field:
        Array containing the receptive field.
    """
    ## Define deltas
    for i in range(img_height//2-fov, img_height//2+fov):
        for j in range(img_width//2-fov, img_width//2+fov):
            for k in range(channels):
                zero = np.zeros((img_height, img_width, channels)) + bg_gain
                zero[i,j,k] += delta_gain

                yield zero

def normalization_fixed_0(responses, expo=0.5):
    """
    Normalization that tries to keep the 0 fixed as well as raising the lower values
    so that they are closer to the higher values.

    Parameters
    ----------
    responses: np.array
        Array of responses in any shape.
    expo: float
        How much we want to raise the lower values.
        A lower number implies more raising but can saturate the higher values.
        A value of 1 doesn't affect the values.
    
    Returns
    -------
    respo_abs_norm_e_signo_vis: np.array
        Array of normalized response with the same shape as the input.
    """
    signo = np.sign(responses)
    respo_abs = np.abs(responses)
    respo_abs_norm = (respo_abs - respo_abs.min()) / (respo_abs.max() - respo_abs.min())
    respo_abs_norm_e = respo_abs_norm ** expo
    respo_abs_norm_e_signo = signo * respo_abs_norm_e
    respo_abs_norm_e_signo_vis = respo_abs_norm_e_signo/2 + 0.5
    return respo_abs_norm_e_signo_vis