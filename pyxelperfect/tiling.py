from skimage.io import imread_collection, imsave, imread
import numpy as np
from tifffile import imsave
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Tuple
from skimage import io
import glob
import os

def calculateOptimalLargestResolution(glob_pattern: str, target_tile_height: int, target_tile_width: int) -> Tuple[int]: 
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
    Tuple[int]

    """
    
    def calcMaxes(images_array):
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
        return max_rows, max_columns

    if isinstance(glob_pattern, str):
        images_array = np.array(imread_collection(glob_pattern))
        max_rows, max_columns = calcMaxes(images_array)
    elif isinstance(glob_pattern, np.ndarray):
        images_array = glob_pattern
        max_rows, max_columns = calcMaxes(images_array)
    elif isinstance(glob_pattern, tuple):
        max_rows, max_columns = glob_pattern

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
    calc_only: bool
        If true, only calculates the padded image size and number of tiles in both dimensions, without creating the tiles. 
        Output can be used to create a tileGrid object. 
        

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
    # """ Class to represent the tiling grid of the tile function
    # Important for usage: tile names start at indexin 1, so for self.tile_boundaries (which is a dict), keys start at 1
    # """
    def __init__(self, rowdiv, coldiv, n_rows, n_cols, original_resolution, image_list = [], data_coordinates_list=[]):
# basic vars
        self.n_rows = n_rows # Number rows in the complete array, untiled
        self.n_cols = n_cols # Number cols in the complete array, untiled
        self.coldiv = int( coldiv ) # Number of tiles in the col-dimension
        self.rowdiv = int( rowdiv ) # Number of tiles in the row dimension
        self.original_resolution = original_resolution # tuple, resolution of the original image, before padding
        self.image_list = image_list # List of images of the original image-stack ,in case the tiling tiles more than 1 image in unison
        self.data_coordinates_list = data_coordinates_list # List of potential dataframes with point coordinates, such as spatial transcriptomics

        # calculated stuff
        self.tile_size_row =int(  n_rows / rowdiv ) # Size of each individual tile in row dimension
        self.tile_size_col =int(  n_cols / coldiv ) # Size of each individual tile in col dimension
        self.n_tiles = int(rowdiv * coldiv) # total number of tiles created

        # numerical representation of the tiles
        self.tile_grid = np.arange(1, self.n_tiles + 1).reshape(self.rowdiv, self.coldiv) # representation of the tile grid in numbers, for if you want to search for the position of a specific tile

        # Calculate boundaries of tiles
        self.tile_boundaries = {} # Dict that stores the boundaries of the tiles (with respect to the padded image, not the original)

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
    def getTile(self, tile_nr = 1, image_nr = 0):
        return self.image_list[image_nr][self.tile_boundaries[tile_nr]]

    def addDataCoordinates(self, data_df):
        self.data_coordinates_list.append(data_df)

    def getTileDataCoordinates(self, tile_nr, data_index=0,rowname="row", colname="col"):
        """getTileDataCoordinates.
        Note to self: because padding happens only at the end of the dimensions, it's not included in the tiling of coordinates, since their coordinate is the same relative to the padded image.

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
        cropped_df = self._cropCoordinateDf(df, tile_nr, rowname=rowname, colname=colname)

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

class tileBorder:
    """ Class to represent the border of a tile. Requires the labeled image of that tile for initialization, since this class is meant to compare
    {orientation: {(row,col): label} }
    """
    def __init__(self, labeled_image:np.ndarray, tile_nr:int=0):
        self.labeled_image = labeled_image

        self.border_mask = self._getArrayBorder(self.labeled_image)
        tmp_list = list( np.unique(self.labeled_image[self.border_mask]))
        # Remove 0, which is background, not an actual label
        tmp_list.remove(0)
        self.border_labels = tmp_list 
        self.nr_border_objects = len(self.border_labels)

        ## only for in-class use
        self._barDict = {"left": self.labeled_image[:, 0], "top": self.labeled_image[0, :], "right" : self.labeled_image[:, -1], "bot": self.labeled_image[-1, :]}

        self.orientation_label_centers = self._getLabelCentersPerOrientation()

        if tile_nr != 0:
            self.tile_nr = tile_nr

    def _getLabelCentersPerOrientation(self):
        """ 
        Creates a dict represention of at which border is each label, in what its center is.

        returns: {orientation: {(row,col): label} }  with orientation either top, right, left of bot
        """ 
        orientation_dict = {}
        for orientation, bar  in self._barDict.items():
            labels = list(np.unique(bar))
            labels.remove(0)
            orientation_dict[orientation] = {}
            for label in labels:
                label_indexes = np.where(bar == label)[0]
                middle = label_indexes[round(len(label_indexes)/2)]
                orientation_dict[orientation][middle] = label
        return orientation_dict

    def _getArrayBorder(self,image):
        mask = np.ones(image.shape, dtype = bool)
        mask[image.ndim * (slice(1, -1), )] = False
        return mask

    def __eq__(self, other):
        return self.tile_nr == other.tile_nr

    def __lt__(self, other):
        return self.tile_nr < other.tile_nr

    def __gt__(self, other):
        return self.tile_nr > other.tile_nr

    def plotBorderCenters(self):
        plt.imshow(self.labeled_image)
        for k, v in self.orientation_label_centers["left"].items():
            plt.scatter(0,k)
        for k, v in self.orientation_label_centers["top"].items():
            plt.scatter(k,0)
        for k, v in self.orientation_label_centers["right"].items():
            plt.scatter((self.labeled_image).shape[1]-1,k)
        for k, v in self.orientation_label_centers["bot"].items():
            plt.scatter(k,(self.labeled_image).shape[0]-1)
        plt.show()

    def matchTileBorders(self, tileBorder2, tile_grid: tileGrid):
        ## we take the smaller number as viewpoint, which means it will be bordering the other tile 
        ref, other = (self, tileBorder2) if self < tileBorder2 else (tileBorder2, self)

        # if other is next to this one, look right to left
        if other.tile_nr == (ref.tile_nr + 1):
            ref_border = "right"
            other_border = "left"
        # if other is one row further, look bot to top
        elif other.tile_nr == (ref.tile_nr + tile_grid.coldiv):
            ref_border = "bot"
            other_border = "top"
        else:
            return {}

        labels_matching_dict = {} # this will store mapping from labels ref to other

        # get all centers of the ref
        center_dict = ref.orientation_label_centers[ref_border]
        if center_dict: # if there are any
            for ref_key in center_dict.keys():
                # for each one, check which centers there are on the other border border 
                other_key_array = np.array(list(other.orientation_label_centers[other_border].keys()))
                # build in an error margin since rounding errors are possible on the center detection
                matching_label = other.orientation_label_centers[other_border][other_key_array[np.where((other_key_array >= (ref_key - 3)) & (other_key_array <= (ref_key + 3)))][0]]
                labels_matching_dict[center_dict[ref_key]] = matching_label

        return labels_matching_dict

    def __str__(self):
        return f"Borders of tile {self.tile_nr}. {self.nr_border_objects} unique objects at borders."


def plotLabeledImageOverlap(labeled_image_paths, tile_grid):
    """Plots labeled images of a tile grid in the correct axis, with overlapping labels as their title
    Pure for visualization of the tileBorder class
    """

    r = re.compile(r"tile(\d+)")
    def key_func(m):
        return int(r.search(m).group(1))
    labeled_image_paths.sort(key=key_func)
    print(labeled_image_paths)

    fig, axs = plt.subplots(tile_grid.rowdiv, tile_grid.coldiv)
    axs = axs.flatten()
    for i in range(1,tile_grid.n_tiles + 1):
        img = io.imread(labeled_image_paths[i-1])
    
        tile = tileBorder(img, i)
        try:
            img2 =  io.imread(labeled_image_paths[i])
            tile2 = tileBorder(img2, i+1)
            tile2_matches = tile.matchTileBorders(tile2, tile_grid)
        except:
            tile2_matches = {}
        try:
            img3 =  io.imread(labeled_image_paths[i + tile_grid.coldiv - 1])
            tile3 = tileBorder(img3, i+tile_grid.coldiv)
            tile3_matches = tile.matchTileBorders(tile3, tile_grid)
        except:
            tile3_matches = {}
            
        axs[i-1].imshow(img)
        axs[i-1].set_title(f"with {i+1}: {list(tile2_matches.keys())} ; with {i + tile_grid.coldiv}: {list(tile3_matches.keys())}")

    plt.show()
