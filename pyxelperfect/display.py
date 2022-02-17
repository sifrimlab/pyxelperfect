import matplotlib.pyplot as plt
from skimage import io
import numpy as np
from skimage.util import img_as_ubyte
from skimage import color
import matplotlib.pyplot as plt

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

