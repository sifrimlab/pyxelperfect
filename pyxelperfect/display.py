import matplotlib.pyplot as plt
from skimage import io
from skimage.util import img_as_ubyte
import matplotlib.plt as plt

def showSegmentation(labeled_image: np.array, original_image: np.array = None, save=True, plot=False):
    colored_image = color.label2rgb(label_image, bg_label=0)
    if original_image:
        original_image = img_as_ubyte(original_image)

    colored_image_on_DAPI = color.label2rgb(labeled_image, original_image, bg_label=0)
    if save:
        io.imsave("labeled_image.tif", colored_image_on_DAPI)

    if plot:
        plt.imshow(colored_image_on_DAPI)
        plt.axis("off")
        
