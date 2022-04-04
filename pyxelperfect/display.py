import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import io
import numpy as np
from skimage.util import img_as_ubyte
from skimage import color
import numpy as np
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


def createComposite(img1_path, img2_path):
    A = cv2.cv2.imread(img1_path, cv2.cv2.IMREAD_ANYDEPTH)
    B = cv2.cv2.imread(img2_path, cv2.cv2.IMREAD_ANYDEPTH)
    bitDepth = A.dtype
    # print(A.dtype, B.dtype)
    zeros = np.zeros(A.shape[:2], dtype=bitDepth)

    ##Don't forget, cv2.cv2 works with a different order sometimes, so it's not RGB it's BGR
    merged = cv2.cv2.merge((A,B,zeros))
    merged = img_as_ubyte(merged)
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
        axs[0].imshow(original_situation)
        axs[0].set_title("Before Registration")
        red_patch = mpatches.Patch(color='red', label='MS ref')
        green_patch = mpatches.Patch(color='green', label='ST target')
        axs[0].legend(handles=[red_patch, green_patch])
        axs[1].imshow(registered_situation)
        axs[1].set_title("After Registration")
        green_patch_2 = mpatches.Patch(color='green', label='ST moved')
        axs[1].legend(handles=[red_patch, green_patch_2])
        plt.show()

if __name__ == '__main__':
    io.imsave("composite_registered.tif",createComposite("/home/david/.config/nnn/mounts/nacho@10.38.76.144/tool/prostate_cancer/david_test/PWB950_cancer/affine_registered/PWB950_cancer_HE_minus_cmc_10X_processed.tif", "/home/david/.config/nnn/mounts/nacho@10.38.76.144/tool/prostate_cancer/david_test/PWB950_cancer/affine_registered/PWB950_DLC2_affine_registered.tif"))
