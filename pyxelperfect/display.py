import matplotlib.pyplot as plt
from skimage import io
import numpy as np
from skimage.util import img_as_ubyte
from skimage import color
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
import cv2

def showSegmentation(labeled_image: np.array, original_image: np.array = None, save=True, plot=False):
    colored_image = color.label2rgb(labeled_image, bg_label=0)
    if original_image is not None:
        original_image = img_as_ubyte(original_image)

    colored_image_on_DAPI = color.label2rgb(labeled_image, original_image, bg_label=0)
    if save:
        io.imsave("labeled_image.tif", colored_image_on_DAPI)

    if plot:
        plt.imshow(colored_image_on_DAPI)
        plt.axis("off")
        plt.show()

def showImage(image: np.array):
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def createComposite(img1, img2):
    A = cv2.cv2.imread(img1, cv2.cv2.IMREAD_ANYDEPTH)
    B = cv2.cv2.imread(img2, cv2.cv2.IMREAD_ANYDEPTH)
    bitDepth = A.dtype
    # print(A.dtype, B.dtype)
    zeros = np.zeros(A.shape[:2], dtype=bitDepth)

    ##Don't forget, cv2.cv2 works with a different order sometimes, so it's not RGB it's BGR
    merged = cv2.cv2.merge((A,B,zeros))
    
    return merged

def evaluateRegistration(ref_image, target_image, registered_target, identifier: str = "", plot=True):
    original_situation = createComposite(ref_image, target_image)
    original_situation = img_as_ubyte(original_situation)
    registered_situation = createComposite(ref_image, registered_target)
    registered_situation = img_as_ubyte(registered_situation)

    if identifier:
        io.imsave(f"overlay_registered_{identifier}.tif", registered_situation)
    if plot:
        fig, axs = plt.subplots(1,2)
        axs[0] = plt.imshow(original_situation)
        axs[1] = plt.imshow(registered_situation)
        plt.show()



