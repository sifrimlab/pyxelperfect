import numpy as np
from skimage import io
from skimage.util import img_as_ubyte
from skimage import color
import matplotlib.pyplot as plt
from skimage import measure
import cv2

def otsuThreshold(image: np.array) -> float:
    """Use otsu's thresholding to find the optimal threshold divind the histogram of the image into 2 distributions.

    Parameters
    ----------
    image : np.array
        image

    Returns
    -------
    float

    """
    bins_num = 256

    # Get the image histogram
    hist, bin_edges = np.histogram(image, bins=bins_num)

    hist = hist.astype(float)

    # # Get normalized histogram if it is required
    # hist = np.divide(hist.ravel(), hist.max())

    # # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.

    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    # Get the class means mu0(t)
    mean1 = np.cumsum(hist * bin_mids) / weight1
    # Get the class means mu1(t)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)

    threshold = bin_mids[:-1][index_of_max_val]
    return threshold


def otsuSegment(img: np.array, tile_nr="") -> np.array :
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





if __name__ == "__main__":
    image_path = "/media/gojira/MERFISH_datasets/download.brainimagelibrary.org/02/26/02265ddb0dae51de/mouse1_sample2_raw/extracted_aligned_images/aligned_images_tile101_DAPI.tiff"
    image = io.imread(image_path)
    label_image = otsuSegment(image)
    showSegmentation(label_image, plot=True)
    # plt.imshow(label_image)
    # plt.show()
    # print(attribute_df)

