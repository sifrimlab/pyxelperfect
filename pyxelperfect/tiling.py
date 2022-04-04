from skimage.io import imread_collection, imsave, imread
import numpy as np
from tifffile import imsave

from typing import List, Tuple
import glob
import os

def calculateOptimalLargestResolution(glob_pattern: str, target_tile_height: int, target_tile_width: int) -> List[int]: 
    """Calculates the optimal maximum resolution based on a set of images assuming that it will be used to tile the images into equal tiles of a certain size, given by the input variables.


    Parameters
    ----------
    glob_pattern : str
        glob pattern hitting all images to be taken into account for the calculation.
    target_tile_height : int
        target_tile_height
    target_tile_width : int
        target_tile_width

    Returns
    -------
    List[int]

    """
    images_array = np.array(imread_collection(glob_pattern), dtype = object)

    heights = []
    widths = []
    for i in range(len(images_array)):
        heights.append(images_array[i].shape[0])
        widths.append(images_array[i].shape[1])
    max_rows = max(heights) 
    max_columns = max(widths)

    ydiv = np.ceil(max_rows / target_tile_height)
    xdiv = np.ceil(max_columns / target_tile_width)

    target_full_rows = ydiv * target_tile_height
    target_full_columns = xdiv * target_tile_width

    return target_full_rows, target_full_columns, ydiv, xdiv

def padImage(image: np.array, target_full_rows: int, target_full_columns: int) -> np.ndarray:
    """Pads an image with black pixels up until a certain number of rows and columns

    Parameters
    ----------
    image : np.array
        image to be padded
    target_full_rows : int
        Number of rows to be achieved by padding
    target_full_columns : int
        Number of columns to be achieved by padding

    Returns
    -------
    np.ndarray

    """
    rowdiff = target_full_rows - image.shape[0]
    columndiff = target_full_columns - image.shape[1]
    padded_img = np.pad(image, ((0, rowdiff), (0, columndiff)))
    return padded_img

def tileImage(image: np.ndarray, ydiv: int, xdiv: int, image_prefix: str="image_tile_"):
    """Tile an image into smaller tiles based on divisions in x and y axes, and saving them to tif files.

    Parameters
    ----------
    image : np.ndarray
        image
    ydiv : int
        Nr of divisions to make in the y axis
    xdiv : int
        Nr of divisions to make in the x axis
    image_prefix : str
        prefix to add to the new filenames of the tiles. default = image_tile
    """
    if not image_prefix.endswith("_"):
        image_prefix += "_"

    temp_split = np.array_split(image, ydiv, axis = 0)
    # Item sublist part is just to unpack a list of lists into one list
    final_split = [item for sublist in [np.array_split(row, xdiv, axis = 1) for row in temp_split] for item in sublist]

    for i, img in enumerate(final_split, 1):
        imsave(f"{image_prefix}tile{i}.tif", img)

def tile(glob_pattern: str, target_tile_width: int, target_tile_height: int, out_dir: str = "") -> Tuple[int]:
    """Tile the images caught by the glob pattern by first padding them to a global image size needed to tile them all into the same tile size, given by the input values. Tiles are written to tif files in the basedir of the original image, or to out_dir if given, with naming convention {out_dir | basedir}/{image_name}_tile{i}.tif

    Parameters
    ----------
    glob_pattern : str
        glob_pattern that catches all images to be tiled.
    target_tile_width : int
        Width of the tiles
    target_tile_height : int
        Height of the tiles

    Returns
    -------
    Tuple[int]

    """
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    target_full_rows, target_full_columns, ydiv, xdiv = calculateOptimalLargestResolution(glob_pattern, target_tile_height, target_tile_width)

    padded_imgs = {}
    if not out_dir:
        for image_path in glob.glob(glob_pattern):
            padded_imgs[os.path.splitext(image_path)[0]] = padImage(imread(image_path), int(target_full_rows), int(target_full_columns))
    else:
        for image_path in glob.glob(glob_pattern):
            padded_imgs[os.path.join(out_dir, os.path.splitext(os.path.basename(image_path))[0])] = padImage(imread(image_path), int(target_full_rows), int(target_full_columns))

    for k, padded_img in padded_imgs.items():
        tileImage(padded_img,ydiv = ydiv, xdiv = xdiv, image_prefix = k)
    return xdiv, ydiv

if __name__ == '__main__':
    tile("/home/david/.config/nnn/mounts/nacho@10.38.76.144/tool/enteric_neurones/slidescanner_examples/Good/Slide2-2-2_Region0000_Channel647,555,488_Seq0017/Slide2-2-2_Region0000_Channel647,555,488_Seq0017_555.tif", 2000, 2000, out_dir = "/home/david/.config/nnn/mounts/nacho@10.38.76.144/tool/enteric_neurones/slidescanner_examples/Good/Slide2-2-2_Region0000_Channel647,555,488_Seq0017/Slide2-2-2_Region0000_Channel647,555,488_Seq0017_555_tiles/")
