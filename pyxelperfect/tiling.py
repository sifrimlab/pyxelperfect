from skimage.io import imread_collection, imsave, imread
import numpy as np
from tifffile import imsave
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Tuple
from skimage import io
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
    if isinstance(glob_pattern, str):
        images_array = np.array(imread_collection(glob_pattern))
    elif isinstance(glob_pattern, np.ndarray):
        images_array = glob_pattern

    heights = []
    widths = []
    if len( images_array.shape ) > 2:
        for i in range(len(images_array)):
            heights.append(images_array[i].shape[0])
            widths.append(images_array[i].shape[1])
        max_rows = max(heights) 
        max_columns = max(widths)
    else:
        max_rows=images_array.shape[0]
        max_columns=images_array.shape[1]


    rowdiv = np.ceil(max_rows / target_tile_height)
    coldiv = np.ceil(max_columns / target_tile_width)

    target_full_rows = rowdiv * target_tile_height
    target_full_columns = coldiv * target_tile_width

    return target_full_rows, target_full_columns, rowdiv, coldiv

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

def padCoordinateDf(df, target_full_rows, target_full_columns, original_resolution, rowname="row", colname="col"):
    row_diff = (target_full_rows - original_resolution[0] )  / 2
    col_diff = (target_full_columns - original_resolution[1] )  / 2

    df[rowname] = df[rowname] + row_diff
    df[colname] = df[colname] + col_diff
    return df




def tileImage(image: np.ndarray, rowdiv: int, coldiv: int, image_prefix: str="image_tile_"):
    """Tile an image into smaller tiles based on divisions in x and y axes, and saving them to tif files.

    Parameters
    ----------
    image : np.ndarray
        image
    rowdiv : int
        Nr of divisions to make in the y axis
    coldiv : int
        Nr of divisions to make in the x axis
    image_prefix : str
        prefix to add to the new filenames of the tiles. default = image_tile
    """
    if not image_prefix.endswith("_"):
        image_prefix += "_"

    temp_split = np.array_split(image, rowdiv, axis = 0)
    # Item sublist part is just to unpack a list of lists into one list
    final_split = [item for sublist in [np.array_split(row, coldiv, axis = 1) for row in temp_split] for item in sublist]

    for i, img in enumerate(final_split, 1):
        imsave(f"{image_prefix}tile{i}.tif", img)

def tile(glob_pattern: str, target_tile_width: int, target_tile_height: int, out_dir: str = "", calc_only=False) -> Tuple[int]:
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
    if out_dir and not calc_only:
        os.makedirs(out_dir, exist_ok=True)

    target_full_rows, target_full_columns, rowdiv, coldiv = calculateOptimalLargestResolution(glob_pattern, target_tile_height, target_tile_width)

    if calc_only:
        return rowdiv, coldiv, target_full_rows, target_full_columns

    padded_imgs = {}
    if not out_dir:
        for image_path in glob.glob(glob_pattern):
            padded_imgs[os.path.splitext(image_path)[0]] = padImage(imread(image_path), int(target_full_rows), int(target_full_columns))
    else:
        for image_path in glob.glob(glob_pattern):
            padded_imgs[os.path.join(out_dir, os.path.splitext(os.path.basename(image_path))[0])] = padImage(imread(image_path), int(target_full_rows), int(target_full_columns))

    for k, padded_img in padded_imgs.items():
        tileImage(padded_img,rowdiv = rowdiv, coldiv = coldiv, image_prefix = k)
    return rowdiv,coldiv, target_full_rows, target_full_columns

class tileGrid:
    ## Tile naming starts at 1
    def __init__(self, rowdiv, coldiv, n_rows, n_cols, image_list = []):
        # basic vars
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.coldiv = int( coldiv )
        self.rowdiv = int( rowdiv )
        self.image_list = image_list

        # calculated stuff
        self.tile_size_row =int(  n_rows / rowdiv )
        self.tile_size_col =int(  n_cols / coldiv )
        self.n_tiles =int(rowdiv * coldiv)

        # numerical representation of the tiles
        self.tile_grid = np.arange(1, self.n_tiles + 1).reshape(self.rowdiv, self.coldiv)

        # Calculate boundaries of tiles
        self.tile_boundaries = {}

        # start at one to make the boundary math check out
        for i in range(1, self.n_tiles + 1):
            idx = np.where(self.tile_grid == i)
            self.tile_boundaries[i] = np.s_[idx[0][0] * self.tile_size_row : (idx[0][0] + 1) * self.tile_size_row, idx[1][0] * self.tile_size_col: (idx[1][0] + 1) * self.tile_size_col]

    def getTileGrid(self):
        print(self.tile_grid)

    def addImage(self, image: np.ndarray):
        self.image_list.append(image)

    def plotTile(self, tile_nr = 1, image_nr = 0):
        plt.imshow(self.image_list[image_nr][self.tile_boundaries[tile_nr]])
        plt.show()


# def getTileGridStats(rowdiv, coldiv, n_rows,n_cols ):
#     grid = {name:value for name, value in locals().items()}
    # grid["tile_size_row"] = n_rows / rowdiv
    # grid["tile_size_col"] = n_cols / coldiv
    # grid["n_tiles"] = grid["rowdiv"] * grid["coldiv"] 
    # # Make a variable that contains all bounds 
    # grid["tile_bounds"] = []
    # return grid


def tileCoordinateTable(df, rowdiv, coldiv, target_full_rows, target_full_columns, original_resolution):
    grid = getTileGridStats(rowdiv, coldiv, target_full_rows, target_full_columns)
    df = padCoordinateDf(df, target_full_rows, target_full_columns, original_resolution)
    #TODO implement actual tile coordinate boxing using the grid object
    # for i in range()
    


if __name__ == '__main__':
    # test_image = np.zeros((75663,114245),dtype=np.bool) 
    test_image = np.diagflat(range(100)) 
    # test_image =io.imread("/home/david/Documents/segmentation_benchmark/test_data/merfish/labeled1_MERFISH_nuclei.tif")
    # print(np.diagonal(test_image))
    # test_image[np.diagonal(test_image)] = range(100)
    test_df = pd.read_csv("./test_df.csv")
    rowdiv, coldiv, target_full_rows, target_full_columns = tile(test_image, 30, 30,calc_only=True)
    grid = tileGrid(rowdiv, coldiv, target_full_rows, target_full_columns)
    
    # tileCoordinateTable(test_df, rowdiv, coldiv, target_full_rows, target_full_columns)
