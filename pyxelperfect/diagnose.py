import os
from functools import reduce
from skimage import io
import numpy as np
import pandas as pd
import glob
from skimage.filters import laplace
from skimage.util import dtype_limits
from .threshold import otsuThreshold
from typing import List, Dict

import PIL # this is needed to read in the giant images with skimage.io.read_collection
PIL.Image.MAX_IMAGE_PIXELS = 933120000

def getImageStats(image: np.ndarray, out_print: bool = True, save: bool = False, result_prefix: str = "results", add_props: Dict = {}) -> pd.DataFrame:
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

    if add_props:
        for key, value in add_props.items():
            attribute_col.append(key)
            value_col.append(value)

    # Make result dataframe
    zipped = list(zip(attribute_col, value_col))
    result_df = pd.DataFrame(zipped, columns=['Attribute', f'{result_prefix}'], index=None)
    result_df.index=result_df['Attribute']
    result_df = result_df.drop('Attribute', axis=1)

    if out_print:
        print(result_df.to_markdown())
    if save:
        result_df.T.to_csv(f"{result_prefix}.csv", index=None)
    return result_df.T

def compareImageStats(glob_pattern: str = None, result_prefix = "result", add_props: Dict = {}) -> pd.DataFrame:
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
        df_list.append((getImageStats(v, result_prefix = k, save=False, out_print=False, add_props = add_props))) # Transpose to make concatenating possible

    # merged_df = reduce(lambda x, y: pd.merge(x, y, on = 'Attribute'), df_list)
    merged_df = pd.concat(df_list)
    merged_df = merged_df.sort_index() # sort for readability
    merged_df.to_csv(f"{result_prefix}.csv")
    return merged_df
        
