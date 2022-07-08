import cv2
import numpy as np
from skimage.transform import resize
from skimage.morphology import white_tophat, disk
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
        return image
    
    # If the image was empty, the alpha calculation will raise an error, so just return the original image
    if maximum_gray - minimum_gray == 0:
        return image
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
    return auto_result

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

def whiteTophat(image, radius):
    """Wrapper for skimage's white tophat filter

    Parameters
    ----------
    image : np.ndarray
        Image to be filtered.
    radius: int
        Radius of the morphological disk.
    """
    selem = disk(radius)
    return white_tophat(image,selem)

def changeRGBcolor(image, original_color, target_color):
    """ Change a specific color in an image to another color

    Parameters
    ----------
    image :
        input image
    original_color :
        tuple representing the rgb color to change
    target_color :
        tuple representing the rgb color to change to 
    """
    image = image.copy()
    if len(image.shape) > 3:
        raise ValueError("not an RGB image")
    if image.shape[2] > 3:
        image = image[:,:,0:3]
    # print(image[:,:,0])
    r1, g1, b1 = original_color
    r2, g2, b2 = target_color

    red, green, blue = image[:,:,0], image[:,:,1], image[:,:,2]
    mask = (red == r1) & (green == g1) & (blue == b1)
    image[:,:,:3][mask] = [r2, g2, b2]
    return image

    # io.imsave("./image_fixed.png", data, check_contrast=False)

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

