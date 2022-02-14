import numpy as np
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

