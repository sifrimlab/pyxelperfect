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

def padCoordinateDf(df, grid, rowname="row", colname="col"):
    row_diff = (grid.n_rows - grid.original_resolution[0] )  / 2
    col_diff = (grid.n_cols - grid.original_resolution[1] )  / 2

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
    def __init__(self, rowdiv, coldiv, n_rows, n_cols, original_resolution, image_list = [], data_coordinates_list=[]):
        # basic vars
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.coldiv = int( coldiv )
        self.rowdiv = int( rowdiv )
        self.original_resolution = original_resolution
        self.image_list = image_list
        self.data_coordinates_list = data_coordinates_list

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

    def addDataCoordinates(self, data_df):
        self.data_coordinates_list.append(data_df)

    def getTileDataCoordinates(self, tile_nr, data_index=0,rowname="row", colname="col"):
        """getTileDataCoordinates.

        Parameters
        ----------
        tile_nr :
            nr of tile to fetch coordinates from. (indexing starts at 1, not at 0)
        data_index :
            index of which dataframe to fetch, in case multiple dataframes are linked with the tiling coordinate. (indexing starts at 0)
        rowname :
            column name that refers to the row dimension
        colname :
            column name that refers to the col dimension
        """

        df = self.data_coordinates_list[data_index]
        # padded_df = self._padCoordinateDf(df)
        cropped_df = self._cropCoordinateDf(df, tile_nr)

        # Now we have all spots belonging to this tile, but we still need to add local coords to them
        return cropped_df

    def _cropCoordinateDf(self, df, tile_nr, rowname="row", colname="col"):
        row_slice, col_slice =  self.tile_boundaries[tile_nr][0], self.tile_boundaries[tile_nr][1]

        tmp_df = df.loc[(df[rowname] >= row_slice.start) & (df[rowname] < row_slice.stop)]
        cropped_df = tmp_df.loc[(tmp_df[colname] > col_slice.start) & (tmp_df[colname] <= col_slice.stop)]

        local_rows = [el - row_slice.start for el in cropped_df[rowname]]
        local_cols = [el - col_slice.start for el in cropped_df[colname]]

        # copy line added since assigning new columns to the normal cropped_df (which is technically a slice of the original df) raises a warning.
        # In this usecase the warning is a false positive, since we don't care about tracing the new column back to the original dataframe
        cropped_df = cropped_df.copy()
        cropped_df["local_row"] = local_rows
        cropped_df["local_col"] = local_cols
        return cropped_df 

    def __str__(self):
        return f"Tile grid of size {self.rowdiv} by {self.coldiv}, {self.n_tiles} in total.\nTiles are {self.tile_size_row} rows by {self.tile_size_col} cols.\n {self.tile_grid}"

if __name__ == '__main__':
    test_df = pd.read_csv("./test_input/own_decoded_intensities.csv")

    rowdiv, coldiv, target_full_rows, target_full_columns = tile("./test_input/MERFISH_dapi2.tif", 500, 500, out_dir = "./test_output/", calc_only = True)
    test_image = io.imread("./test_input/MERFISH_dapi2.tif")

    grid = tileGrid(rowdiv, coldiv, target_full_rows, target_full_columns, test_image.shape)
    grid.addDataCoordinates(test_df)

    for i in range(1,grid.n_tiles + 1 ):
        tmp_df = grid.getTileDataCoordinates(i)
        tmp_df.to_csv(f"./test_output/data_coords_tile{i}.csv")
