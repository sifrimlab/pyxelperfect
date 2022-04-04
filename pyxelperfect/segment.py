from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import cv2
from skimage import io
from skimage.util import img_as_ubyte
from skimage import color
import matplotlib.pyplot as plt
from skimage import measure
from csbdeep.utils import Path, normalize
from stardist.models import StarDist2D
from .display import showSegmentation
from cellpose import models


def otsuSegment(img: np.array) -> np.array :
    '''
        note to self: this code adapted from DigitalSreeni assumes that your input image is an 8-bit rgb, which makes it so that we have to do some image format transformation because:
        - cv2.shreshold accepts only 8-bit grayscale
        - cv2.watershed only accepts 8-bit rgb
    '''
    '''
    returns a labeled image, where 0 = background and all other integers an object number.
    These numbers don't have any actual image value, so the image isn't really used as an image object, but more as an array
    that stores which pixel belongs to which label. Also returns a csv that contains image properties of the given objects
    '''
    # Create an 8bit version of the image
    img_as_8 = img_as_ubyte(img)
    # Creat an RGB version that only has signal in the blue channel
    shape = img_as_8.shape
    empty = np.zeros(shape)
    img_as_8bit_RGB = cv2.merge([img_as_8,img_as_8,img_as_8])
    try:
        cells = img[:,:,0]
    except IndexError:
        cells = img_as_8

    ret1, thresh = cv2.threshold(cells, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Morphological operations to remove small noise - opening
    #To remove holes we can use closing
    kernel = np.ones((3,3),np.uint16)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # from skimage.segmentation import clear_border
    # opening = clear_border(opening) #Remove edge touching grains

    sure_bg = cv2.dilate(opening,kernel,iterations=10)
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret2, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)

    #Later you realize that 0.25* max value will not separate the cells well.
    #High value like 0.7 will not recognize some cells. 0.5 seems to be a good compromize

    # Unknown ambiguous region is nothing but bkground - foreground
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    #Now we create a marker and label the regions inside.
    # For sure regions, both foreground and background will be labeled with positive numbers.
    # Unknown regions will be labeled 0.
    #For markers let us use ConnectedComponents.
    ret3, markers = cv2.connectedComponents(sure_fg)

    #One problem rightnow is that the entire background pixels is given value 0.
    #This means watershed considers this region as unknown.
    #So let us add 10 to all labels so that sure background is not 0, but 10
    markers = markers+10

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    #Now we are ready for watershed filling.
    markers = cv2.watershed(img_as_8bit_RGB,markers)

    #The boundary region will be marked -1
    markers[markers==-1] = 10 # add the boundary images to the background

    label_image = measure.label(markers, background=10)
    return label_image


def stardistSegment(img: np.array, model:str = "2D_versatile_fluo") -> np.array:
    """Segments cells using pretrained stardist models.

    Parameters
    ----------
    img : np.array
        image to be segmented
    model : str
        String representing which pretrained stardist model to load. Choose one of: (2D_versatile_fluo, 2D_versatile_he, 2D_paper_dsb2018)

    Returns
    -------
    np.array

    """
    model_versatile = StarDist2D.from_pretrained(model)

    # extract number of channels in case the input image is an RGB image
    n_channel = 1 if img.ndim == 2 else img.shape[-1]
    # depending on that, we want to normalize the channels independantly
    axis_norm = (0,1)   # normalize channels independently
    # axis_norm = (0,1,2) # normalize channels jointly

    img_normalized = normalize(img, 1,99.8, axis=axis_norm)

    labeled_image, details = model_versatile.predict_instances(img_normalized)
    return labeled_image 

def cellPoseSegment(img: np.array, model:str = "nuclei", channels: np.array = [0,0]) -> np.array:
    """Segments cells using pretrained Cellpose models. Can do both cytoplasm and nuclei at the same time.

    Parameters
    ----------
    img : np.array
        img
    model : str
        String representing which model to use. Either 'nuclei', 'cyto'
    channels : np.array
        Array depicting which channel is either nucleus or DAPI, default = [0,0] -> grayscale for all images
        How to indicate:

        Grayscale=0, R=1, G=2, B=3
        channels = [cytoplasm, nucleus]
        if NUCLEUS channel does not exist, set the second channel to 0
        IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
        channels = [0,0] # IF YOU HAVE GRAYSCALE
        channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
        channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus
        If they have different orders -> make it a 2D array, e.g: 3 images:
        channels = [[2,3], [1,2], [3,1]]

    Returns
    -------
    np.array

    """
    # model_type='cyto' or model_type='nuclei'
    model = models.Cellpose(gpu=False, model_type=model)
    masks, flows, styles, diams = model.eval(img, diameter=None, channels=channels)
    return masks

