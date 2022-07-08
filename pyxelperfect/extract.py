import os
from skimage import io
from skimage.filters import laplace
import numpy as np
from typing import List, Tuple
import aicspylibczi
from nd2reader import ND2Reader
from itertools import product
from .utils import getLastDimensions

def getMostInFocusImage(image_array_list: List[np.array]) -> Tuple[np.array, int]:  
    """Gets most in focus image and its index from a list of image-arrays.

    Parameters
    ----------
    image_array_list : List[np.array]
        image_array_list

    Returns
    -------
    Tuple[np.array, int]

    """
    stdev_list = []
    for image in image_array_list:
        # calculate edges in image
        edged = laplace(image)
        # Calculate stdev of edges
        stdev = np.std(edged)
        stdev_list.append(stdev)
    # Find largest stdev in list
    largest = max(stdev_list)
    # Fidn which index it is to link back to the original list
    index = stdev_list.index(largest)
    return image_array_list[index], index

def readImage(file_path: str) -> Tuple[np.array, str]:
    """Reads in image-file and returns both the image, its file type and its shape.

    Parameters
    ----------
    file_path : str
        path to image

    Returns
    -------
    Tuple[np.array, str]

    """
    if file_path.lower().endswith((".tif", ".tiff")):
        image = io.imread(file_path)
        img_type = "tif"
        img_shape = image.shape
        return image, img_type, img_shape
    if file_path.lower().endswith(".czi"):
        image = aicspylibczi.CziFile(file_path)
        img_type = "czi"
        img_shape  = image.get_dims_shape()[0]
        return image, img_type, img_shape
    if file_path.lower().endswith(".nd2"):
        with ND2Reader(file_path) as image:
            img_type = "nd2"
            img_shape = image.sizes
            return file_path, img_type, img_shape


def extractMostInFocusZstack(image: np.ndarray, z_shape_index:int = -1) -> np.ndarray:
    """Extract the most in focus z-stack of an image.

    Parameters
    ----------
    image : np.ndarray
        image
    z_shape_index : int
        index that indicates which position in the image.shape contains the z-axis.

    Returns
    -------
    np.ndarray

    """
    if not len(image.shape) > 2:
        raise ValueError(f"Image shape expected to be larger than 2, image shape is {image.shape}")
    img_list = []
    for i in range(image.shape[z_shape_index]):
        img_list.append(np.take(image, i, z_shape_index))
    mostInFocusImage, index = getMostInFocusImage(img_list)
    return mostInFocusImage, index



def extractSingleImages(image, image_type: str, indexes_dict: dict = None, filename: str = ""):
    """Extract from multi-dimensional input images all singe-stack images indicated by the indexes dict.

    Parameters
    ----------
    image : np.ndarray
        multi-dimensional input image
    indexes_dict : dict
        dict where keys are the shape-indexes and their values are indexes of which channels in that axis to extract. The combination of all key-value pairs will be extracted into single-channel images. Default = all combinations of axises that aren't the first 2 (which usually represents X and Y).
    image_type : str
        image_type, choose either "tif", "czi" or "nd2"
    filename : str
        filename
    """
    if image_type == "tif":
        # First check if it's a string, if so we gotta read the image still
        if isinstance(image, str):
            if not filename: # If no filename is given, we'll take the input image string
                filename = image
            image = io.imread(image)
        if indexes_dict == None: # if no indexes_dict is given, calculate all combinations of no X and Y axises
            indexes_list = [list(range(image.shape[i])) for i in range(len(image.shape))[2:]]
            # Create combinations of all channels
            all_combinations =list(product(*indexes_list))
            for combo in all_combinations:
                tmp_extracted_image = np.copy(image) # create copy to iteratively extract dimensions
                append_string = "" # String to append to the end of this extracted img
                for i, entry in enumerate(combo,2):
                    tmp_extracted_image = np.take(tmp_extracted_image, entry, 2)
                    append_string += f"i{i}_c{entry}"
                io.imsave(f"{os.path.splitext(filename)[0]}_{append_string}.tif", tmp_extracted_image, check_contrast=False)
        else:
            for shape_index, channel_indexes in indexes_dict.items():
                if isinstance(channel_indexes, int): # if it's only one channel, it'ss give an iterable error, so we gotta make it a list
                    channel_indexes = [channel_indexes] 
                for channel_index in channel_indexes:
                    if shape_index == None:
                        extracted_image = np.take(image, channel_index, axis=-1)
                    else:
                        extracted_image = np.take(image, channel_index, axis=shape_index)
                    io.imsave(f"{os.path.splitext(filename)[0]}_i{shape_index}_c{channel_index}.tif", extracted_image)

            
    elif image_type == "czi":
        if isinstance(image, str):
            if not filename: # If no filename is given, we'll take the input image string
                filename = image
            image = aicspylibczi.CziFile(image)
        if indexes_dict == None: # if no indexes_dict is given, calculate all combinations of no X and Y axises
            channel_indices = range(*image.get_dims_shape()[0]["C"])
            for channel_index in channel_indices:
                if image.is_mosaic():

                    tmp_image = image.read_mosaic(C=channel_index,scale_factor=1)
                else:
                    tmp_image = image.read_image(C=channel_index)

                tmp_image = getLastDimensions(tmp_image, nr_dims = 2)# extract last 2 dims, first or second dim might be T or S
                io.imsave(f"{os.path.splitext(filename)[0]}_c{channel_index}.tif", tmp_image, check_contrast=False)

        else:
            for shape_index, channel_indexes in indexes_dict.items():
                # now we assume the shape indexes are letters referring to czi stuff
                if isinstance(channel_indexes, int): # if it's only one channel, it'ss give an iterable error, so we gotta make it a list
                    channel_indexes = [channel_indexes] 
                for channel_index in channel_indexes:
                    extracted_image,shape = image.read_image(shape_index=channel_index)
                    extracted_image = getLastDimensions(extracted_image, nr_dims = 2)
                    io.imsave(f"{os.path.splitext(filename)[0]}__{shape_index}{channel_index}.tif", extracted_image )

    elif image_type == "nd2":
        with ND2Reader(image) as images:
            for shape_index, channel_indexes in indexes_dict.items():
                if isinstance(channel_indexes, int): # if it's only one channel, it'ss give an iterable error, so we gotta make it a list
                    channel_indexes = [channel_indexes] 
                for channel_index in channel_indexes:
                    extracted_image = images.get_frame_2D(c=channel_index)
                    io.imsave(f"{os.path.splitext(filename)[0]}_i{shape_index}_c{channel_index}.tif", extracted_image)
