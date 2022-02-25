import os
import pandas as pd
from skimage import io
from skimage.color import rgb2gray
from skimage.util import invert
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

def cutBboxFromImage(bbox: Tuple, image: np.ndarray):
        top_left = bbox[0]
        bot_right = bbox[1]
        row_min, col_min = [coordinate if coordinate > 0 else 0 for coordinate in top_left]
        row_max, col_max = [coordinate if coordinate > 0 else 0 for coordinate in bot_right]

        cut_image = image[row_min:row_max, col_min:col_max]
        io.imsave(filename, cut_image)


def isolateForeground(input_image: np.array):
    # create labeled image

    # Meaure imageprops
    df = measureLabeledImage(labeled_image)

    # get max area row
    image_label = df.iloc[df['Area'].idxmax()]['image_label']
    image_bbow =  df.iloc[df['Area'].idxmax()]['bbox']

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
    df = pd.read_csv("/home/david/Documents/prostate_cancer/cv2labeled.csv")
    isolateForeground(df)




