import numpy as np
from skimage.filters import sobel
from skimage import io
from typing import List


def maxIPstack(img_list: List[np.array]) -> np.array:
    """Takes maximum intensity projection of a list of images

    Parameters
    ----------
    img_list : List[np.array]
        list of images

    Returns
    -------
    np.array

    """
    parsed_list = img_list
    parsed_list = [img if isinstance(img, np.ndarray) else io.imread(img) for img in img_list]

    maxIP = np.maximum.reduce(parsed_list)
    return maxIP

def edf(img_list) -> np.array:
    ## Credist to the documentation of the mahotas package https://mahotas.readthedocs.io/en/latest/edf.html
    ## Important: image array has to have the z-stacks as first dimension
    if isinstance( img_list, list):
        img_array = np.zeros((len(img_list),*img_list[0].shape))
        for i, img in enumerate(img_list):
            img_array[i, :, :] = img
    else:
        img_array = img_list
    stack,h,w = img_array.shape

    focus = np.array([sobel(img) for img in img_array])
                      
    best = np.argmax(focus, 0)

    img_array = img_array.reshape((stack,-1)) # image is now (stack, nr_pixels)
    img_array = img_array.transpose() # image is now (nr_pixels, stack)
    r = img_array[np.arange(len(img_array)), best.ravel()] # Select the right pixel at each location
    r = r.reshape((h,w)) # reshape to get final result
    return r




    
