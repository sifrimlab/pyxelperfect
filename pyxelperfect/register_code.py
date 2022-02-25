import os
import pandas as pd
import cv2 as cv
import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.util import invert,img_as_ubyte
import matplotlib.pyplot as plt
from manipulate import equalizeImageSize
from register import performRegistration
from measure import measureLabeledImage









"""
    0: rotate
    1: grayscale
    2: invert grayscale values to make black on white instead of the other way around?
    3: crop matrix out of ST
    4: downscale ST to MSI size
    5: register
    6: make overlays?
"""

def cutBboxFromImage(bbox, image: np.ndarray):
        top_left = bbox[0]
        bot_right = bbox[1]
        row_min, col_min = [coordinate if coordinate > 0 else 0 for coordinate in top_left]
        row_max, col_max = [coordinate if coordinate > 0 else 0 for coordinate in bot_right]

        cut_image = image[row_min:row_max, col_min:col_max]
        io.imsave(filename, cut_image)


def isolateForeground(input_image: np.array):
    # create labeled image
    gray_correct = np.array(255 * (input_image / 255) ** 1.2 , dtype='uint8')
    # Local adaptative threshold

    thresh = cv.adaptiveThreshold(gray_correct, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 255, 19)
    thresh = cv.bitwise_not(thresh)

    # # Dilatation et erosion
    kernel = np.ones((15,15), np.uint8)
    img_dilation = cv.dilate(thresh, kernel, iterations=1)
    img_erode = cv.erode(img_dilation,kernel, iterations=1)
    # # clean all noise after dilatation and erosion
    img_erode = cv.medianBlur(img_erode, 7)

    ret, labeled_image = cv.connectedComponents(img_erode)

    io.imsave("tmp_labeledimg.tif", labeled_image)
    # Meaure imageprops
    df = measureLabeledImage(labeled_image)

    # get max area row
    print(df.iloc[df['Area'].idxmax()]['image_label'])
    # image_label = df.iloc[df['Area'].idxmax()]['image_label']
    # image_bbow =  df.iloc[df['Area'].idxmax()]['bbox']

    # convert bbox row to a tuple so that it can serve as input for cutBboxFromImage

    # extract based on bounding box



def perform_registration(ref_image_path: str, target_image_path: str):
    target_base_name = os.path.splitext(target_image_path)[0]
    original_ref_image = io.imread(ref_image_path) 
    # target_image = io.imread(target_image_path) 


    # grayscale
    ref_image = rgb2gray(original_ref_image)
    # target_image = rgb2gray(target_image)

    # invert grayscale image
    ref_image = invert(ref_image)
    # target_image = invert(target_image)

    # crop out unecesary pixels
    # im[~np.all(im == 0, axis=1)] then im[~np.all(im == 0, axis=2)]

    # Downscale
    target_image = equalizeImageSize(ref_image, target_image)

    registered_image = performRegistration(ref_image, target_image, method="bspline")

    fig, axs = plt.subplots(1,2)
    axs[0].imshow(original_ref_image, cmap="gray")
    axs[1].imshow(ref_image, cmap="gray")
    plt.show()

if __name__ == '__main__':
    # ref_image_path = "./PWB929_normal_HE_minus_cmc_10X.tif"
    # target_image_path = "./PWB929_DLC1.tif"
    # perform_registration(ref_image_path, target_image_path)
    isolateForeground(io.imread("/home/david/Documents/prostate_cancer/testing_data/PWB929_DLC1_grey.tif"))




