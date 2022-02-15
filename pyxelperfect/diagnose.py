import os
from functools import reduce
from skimage import io
import numpy as np
import pandas as pd
import glob
from skimage.filters import laplace
from skimage.util import dtype_limits
from segment import otsuThreshold
from typing import List

import PIL # this is needed to read in the giant images with skimage.io.read_collection
PIL.Image.MAX_IMAGE_PIXELS = 933120000

def getImageStats(image: np.ndarray, out_print: bool = False, save: bool = False, result_prefix: str = "results") -> pd.DataFrame:
    """Returns general stats about the input image.

    Parameters
    ----------
    image : np.ndarray
        input image
    out_print : bool
        Determines whether to print out the results to stdout, default = False
    save : bool
        Determines whether to save the results to a csv, default = False
    result_prefix : str
        prefix to use when saving the csv: output file name will be {result_prefix}.csv

    Returns
    -------
    pd.DataFrame

    """
    attribute_col = []
    value_col = []

    attribute_col.append("Shape")
    value_col.append(image.shape)

    attribute_col.append("Dtype")
    value_col.append(image.dtype)

    attribute_col.append("Dtype range")
    value_col.append(dtype_limits(image))

    attribute_col.append("Actual range")
    value_col.append((np.amin(image), np.amax(image)))

    attribute_col.append("Mean")
    value_col.append(np.mean(image))

    attribute_col.append("Median")
    value_col.append(np.median(image))

    attribute_col.append("Laplace std")
    value_col.append(np.std(laplace(image)))

    otsu_threshold = otsuThreshold(image)
    attribute_col.append("Otsu threshold")
    value_col.append(otsu_threshold)

    above_otsu = np.sum(np.where(image > otsu_threshold, 1, 0))
    below_otsu =(image.shape[0] * image.shape[1]) - above_otsu
    perc_above = above_otsu / (image.shape[0] * image.shape[1]) * 100

    attribute_col.append("Pixels above otsu")
    value_col.append(above_otsu)

    attribute_col.append("Pixels below otsu")
    value_col.append(below_otsu)

    attribute_col.append("Percentage above otsu")
    value_col.append(perc_above)

    # Make result dataframe
    zipped = list(zip(attribute_col, value_col))
    result_df = pd.DataFrame(zipped, columns=['Attribute', f'{result_prefix}'], index=None)

    if out_print:
        print(result_df.to_markdown())
    if save:
        result_df.to_csv(f"{result_prefix}.csv", index=None)
    return result_df

def compareImageStats(glob_pattern: str = None, result_prefix = "result") -> pd.DataFrame:
    """Create a dataframe comparing the general stats of a collection of images, defined by a glob pattern.

    Parameters
    ----------
    glob_pattern : str
        glob_pattern
    result_prefix :
        prefix to use when saving the csv: output file name will be {result_prefix}.html

    Returns
    -------
    pd.DataFrame

    """
    # if glob_pattern is None and image_list is None:
    #     raise TypeError("Function requires either glob_pattern or image_list to be given, not both.") 
    
    
    name_image_dict = {os.path.basename(file): io.imread(file) for file in glob.glob(glob_pattern)}
    # image_list = io.imread_collection(glob_pattern)

    df_list  = []
    for k,v in name_image_dict.items():
        df_list.append(getImageStats(v, result_prefix = k, save=False, out_print=False))

    merged_df = reduce(lambda x, y: pd.merge(x, y, on = 'Attribute'), df_list)
    merged_df.to_html(f"{result_prefix}.html")

        


if __name__ == '__main__':
    # image = io.imread("/media/tool/enteric_neurones/slidescanner_examples/Good/processed_Slide2-2-2_Region0000_Channel647,555,488_Seq0017/Slide2-2-2_Region0000_Channel647,555,488_Seq0017_c1_z0_tile14.tif")
    # getImageStats(image, out_print = True)
    # image_list = [io.imread(file) for file in ("/media/tool/enteric_neurones/slidescanner_examples/Good/processed_Slide2-2-2_Region0000_Channel647,555,488_Seq0017/Slide2-2-2_Region0000_Channel647,555,488_Seq0017_c1_z0_tile4.tif", "/media/tool/enteric_neurones/slidescanner_examples/Good/processed_Slide2-2-2_Region0000_Channel647,555,488_Seq0017/Slide2-2-2_Region0000_Channel647,555,488_Seq0017_c1_z0_tile5.tif")]
    compareImageStats(glob_pattern = "/media/tool/enteric_neurones/slidescanner_examples/Good/processed_Slide2-2-2_Region0000_Channel647,555,488_Seq0017/*tile*.tif")


    
