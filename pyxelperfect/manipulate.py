import cv2
import numpy as np
from skimage.transform import resize
from skimage.util import img_as_uint
from diagnose import getImageStats
from skimage import io
from icecream import ic


# Automatic brightness and contrast optimization with optional histogram clipping
def automaticBrightnessAndContrast(image: np.array, clip_hist_percent: int =1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)
    
    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))
    
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0
    
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    
    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    
    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    
    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)

def equalizeImageSize(ref_image: np.array, target_image: np.array, save = False, out_name = "") -> np.array:
    target_resized = resize(target_image,ref_image.shape)
    target_resized = img_as_uint(target_resized)
    if save and out_name:
        io.imsave(f"{out_name}_resized.tif", target_resized)
    return target_resized


if __name__ == "__main__":

    image = io.imread("/home/david/Documents/prostate_cancer/PWB929_normal_HE_minus_cmc_10X_grey.tif")
    image2 = io.imread("/home/david/Documents/prostate_cancer/PWB929_DLC1_grey_cropped.tif")
    equalizeImageSize(image, image2, save=True, out_name = "/home/david/Documents/prostate_cancer/PWB929_DLC1_grey_cropped")
    
    # auto_result, alpha, beta = automaticBrightnessAndContrast(image)
    # print('alpha', alpha)
    # print('beta', beta)
    # cv2.imwrite("/home/david/Documents/communISS/data/merfish/merfish_5_contrast.tif", auto_result)
    # cv2.imshow('auto_result', auto_result)
    # cv2.waitKey()