import cv2
import numpy as np
from skimage.transform import resize
from skimage.util import img_as_uint
from skimage import io


# testing imports
import matplotlib.pyplot as plt
from skimage.draw import ellipse


# Automatic brightness and contrast optimization with optional histogram clipping
def automaticBrightnessAndContrast(image: np.array, clip_hist_percent: int =1):
    if np.amin(image) == 0 and np.amax(image) == 0:
        return image, 0, 0
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else: 
        gray = image
    
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
    
    try:
        # Locate left cut
        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1
        
        # Locate right cut
        maximum_gray = hist_size -1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1
    except:
        return image, 0, 0
    
    # If the image was empty, the alpha calculation will raise an error, so just return the original image
    if maximum_gray - minimum_gray == 0:
        return image, 0, 0
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
    # Gross ifelse statement to check if one of the image is globally larger or smaller than the other
    ref_shape, target_shape = ref_image.shape, target_image.shape
    # if (ref_shape[0] > target_shape[0] and ref_shape[1] < target_shape[1]) or (target_shape[0] > ref_shape[0] and target_shape[1] < ref_shape[1]):
    #     return ref_image, target_image
    # else:
    target_resized = resize(target_image,ref_image.shape)
    target_resized = img_as_uint(target_resized)
    if save and out_name:
        io.imsave(f"{out_name}_resized.tif", target_resized)
    return ref_image, target_resized

if __name__ == '__main__':
    ref_image = np.zeros((1600,500), dtype=np.uint)
    ref_image[ellipse(800,250, 700, 100)] = 255
    target_image = np.zeros((1000,1000), dtype=np.uint)
    target_image[ellipse(500,500, 400,200)] = 255

    warped_ref_image, warped_target_image = equalizeImageSize(ref_image, target_image)

    fig, axs = plt.subplots(2,2)
    axs[0,0].imshow(ref_image)
    axs[0,1].imshow(target_image)
    axs[1,0].imshow(warped_ref_image)
    axs[1,1].imshow(warped_target_image)
    plt.show()

