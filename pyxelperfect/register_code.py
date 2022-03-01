import os
import pandas as pd
import cv2 as cv
import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.util import invert,img_as_ubyte, img_as_uint
from skimage.transform import rotate
import matplotlib.pyplot as plt
from manipulate import equalizeImageSize
from register import performRegistration
from measure import measureLabeledImage
from decorators import measureTime
from icecream import ic



"""
    0: rotate? Might just leave that to the registration, cause not all of them will be rotated
    1: grayscale
    2: invert grayscale values to make black on white instead of the other way around?
    3: crop matrix out of ST
    4: downscale ST to MSI size
    5: register
    6: make overlays?
"""
def expandBbox(bbox, size: int):
    min_row, min_col, max_row, max_col = bbox
    min_row -= size
    max_row += size
    min_col -= size
    max_col += size
    bbox = min_row, min_col, max_row, max_col
    return bbox

def cutBboxFromImage(bbox, image: np.ndarray):
        min_row, min_col, max_row, max_col = bbox

        cut_image = image[min_row:max_row, min_col:max_col]
        #io.imsave(filename, cut_image)
        return cut_image

def isolateForeground(input_image: np.array, kernel_size:int = 100, bbox_expansion:int= 1000):
    # create labeled image
    # gray_correct = np.array(255 * (input_image / 255) ** 1.2 , dtype='uint8')
    gray_correct = img_as_ubyte(input_image)
    # Local adaptative threshold

    thresh = cv.adaptiveThreshold(gray_correct, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 255, 19)
    thresh = cv.bitwise_not(thresh)

    # # Dilatation et erosion
    kernel = np.ones((kernel_size,kernel_size), np.uint8)
    img_dilation = cv.dilate(thresh, kernel, iterations=1)
    img_erode = cv.erode(img_dilation,kernel, iterations=1)
    # # clean all noise after dilatation and erosion
    img_erode = cv.medianBlur(img_erode, 7)

    ret, labeled_image = cv.connectedComponents(img_erode)


    # io.imsave("tmp_labeledimg.tif", labeled_image)
    # Meaure imageprops
    df = measureLabeledImage(labeled_image)

    # get max area row
    max_area_label_number = df.iloc[df['Area'].idxmax()]['image_label']

    # check if the object is horizontal or vertical, horizontal if orientation < 0
    bool_rotate = df.iloc[df['Area'].idxmax()]['orientation']

    # Set everything that isnt max label to zero
    labeled_image[labeled_image!=max_area_label_number] = 0
    image_bbox =  df.iloc[df['Area'].idxmax()]['bbox']
    image_bbox = expandBbox(image_bbox, bbox_expansion)

    foreground_image = cutBboxFromImage(image_bbox, input_image)

    return foreground_image, bool_rotate



@measureTime
def perform_registration(ref_image_path: str, target_image_path: str):
    target_base_name = os.path.splitext(target_image_path)[0]
    original_ref_image = io.imread(ref_image_path) 
    original_target_image = io.imread(target_image_path) 


    # grayscale
    ref_image = rgb2gray(original_ref_image)

    target_image = rgb2gray(original_target_image)

    # invert grayscale image
    ref_image = invert(ref_image)

    target_image = invert(target_image)


    # Crop foreground from the weird Spatial H&E stain
    target_image, bool_rotate = isolateForeground(target_image, bbox_expansion= 500)

    target_image = rotate(target_image, -90, resize=True)

    # Downscale
    target_image = equalizeImageSize(ref_image, target_image)


    # Equalize ref image dtype, cause target image comes out as uint
    ref_image = img_as_uint(ref_image)

    registered_image = performRegistration(ref_image, target_image, method="bspline", out_name=f"{target_base_name}_registered.tif")

    fig, axs = plt.subplots(1,2)
    axs[0].imshow(ref_image, cmap="gray")
    axs[1].imshow(registered_image, cmap="gray")
    plt.show()

if __name__ == '__main__':
    ref_image_path = "/home/david/Documents/prostate_cancer/testing_data/PWB929_normal_HE_min_cmc_10X.tif"
    target_image_path = "/home/david/Documents/prostate_cancer/testing_data/PWB929_DLC1.tif"
    perform_registration(ref_image_path, target_image_path)
    # isolateForeground(io.imread("/home/david/Documents/prostate_cancer/testing_data/PWB929_DLC1.tif"))




