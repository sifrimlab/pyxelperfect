import os
import glob
import re
import numpy as np
import pandas as pd
from skimage import io
from tifffile import imsave
from typing import List, Tuple
import matplotlib.pyplot as plt
from skimage.io import imread_collection, imsave, imread




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
        grid = tileGrid(rowdiv, coldiv, target_full_rows, target_full_columns)
        return grid

    padded_imgs = {}
    if not out_dir:
        for image_path in glob.glob(glob_pattern):
            padded_imgs[os.path.splitext(image_path)[0]] = padImage(imread(image_path), int(target_full_rows), int(target_full_columns))
    else:
        for image_path in glob.glob(glob_pattern):
            padded_imgs[os.path.join(out_dir, os.path.splitext(os.path.basename(image_path))[0])] = padImage(imread(image_path), int(target_full_rows), int(target_full_columns))

    for k, padded_img in padded_imgs.items():
        tileImage(padded_img,rowdiv = rowdiv, coldiv = coldiv, image_prefix = k)
    grid = tileGrid(rowdiv, coldiv, target_full_rows, target_full_columns)
    return rowdiv,coldiv, target_full_rows, target_full_columns

class tileGrid:
    # """ Class to represent the tiling grid of the tile function
    # Important for usage: tile names start at indexin 1, so for self.tile_boundaries (which is a dict), keys start at 1
    # image list and data coordinates are there to be able to tile them after the fact
    # """
    def __init__(self, rowdiv, coldiv, n_rows, n_cols, image_list = [], data_coordinates_list=[], tiles={}):
# basic vars
        self.n_rows = n_rows # Number rows in the complete array, untiled
        self.n_cols = n_cols # Number cols in the complete array, untiled
        self.coldiv = int( coldiv ) # Number of tiles in the col-dimension
        self.rowdiv = int( rowdiv ) # Number of tiles in the row dimension
        self.image_list = image_list # List of images of the original image-stack ,in case the tiling tiles more than 1 image in unison
        self.data_coordinates_list = data_coordinates_list # List of potential dataframes with point coordinates, such as spatial transcriptomics
        self.tiles = tiles

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

    def addImage(self, image: np.ndarray):
        self.image_list.append(image)

    def addTile(self, tile: "Tile"):
        self.tiles[tile.tile_nr] = tile

    def getTile(self, tile_nr):
        return self.tiles[tile_nr]

    def plotImageTile(self, tile_nr = 1, image_nr = 0):
        plt.imshow(self.image_list[image_nr][self.tile_boundaries[tile_nr]])
        plt.show()

    def getImageTile(self, tile_nr = 1, image_nr = 0):
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

    def getNeighbouringTiles(self, tile_nr):
        flattened_grid = self.tile_grid.flatten()
        idx = np.argwhere(self.tile_grid == tile_nr)[0]
        left = tile_nr - 1 if (tile_nr - 1  in flattened_grid) and (idx[1] !=  0) else None
        right =  tile_nr + 1 if (tile_nr + 1 in flattened_grid) and (idx[1] != self.coldiv - 1) else None
        top =  tile_nr - self.coldiv if (tile_nr - self.coldiv in flattened_grid) and (idx[0] != 0) else None
        bot =  tile_nr + self.coldiv if (tile_nr + self.coldiv in flattened_grid) and (idx[0] != self.rowdiv-1) else None
        return {"left": left, "top": top, "bot": bot, "right": right}

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
        self.tile_nr = tile_nr

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

    def matchTileBorders(self, tileBorder2, tile_grid: "tileGrid", error_margin = 5):
        ## we take the largest number as viewpoint, which means it will be bordering the other tile 
        ref, other = (self, tileBorder2) if self > tileBorder2 else (tileBorder2, self)
        # print(ref.orientation_label_centers, other.orientation_label_centers)

        # if other is next to this one, look right to left
        if other.tile_nr == (ref.tile_nr - 1):
            ref_border = "left"
            other_border = "right"
        # if other is one row further, look bot to top
        elif other.tile_nr == (ref.tile_nr - tile_grid.coldiv):
            ref_border = "top"
            other_border = "bot"
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
                matching_label = other.orientation_label_centers[other_border][other_key_array[np.where((other_key_array >= (ref_key - error_margin)) & (other_key_array <= (ref_key + error_margin)))][0]]
                labels_matching_dict[center_dict[ref_key]] = matching_label

        return labels_matching_dict

    def __str__(self):
        return f"Borders of tile {self.tile_nr}. {self.nr_border_objects} unique objects at borders."




class Tile:
    def __init__(self, labeled_image, detected_genes_df, measured_df, grid, tile_nr):
        self.labeled_image = labeled_image
        self.detected_genes_df = detected_genes_df
        self.grid = grid
        self.measured_df = measured_df
        self.tile_nr = tile_nr

    def getGenesOfLabel(self, label):
        gene_dict = {} # keys = label integers, values = dict{gene: count}
        gene_dict[label] = {}

        for row in self.detected_genes_df.itertuples():
            try:
                this_label = self.labeled_image[row.local_row, row.local_col]
            except IndexError:
                continue
            if label == this_label:
                gene_dict[label][row.Gene] = gene_dict[label].get(row.Gene, 0) + 1 
        return gene_dict

    
    def createCountMatrix(self, left_tile: "tileBorder" = None, top_tile: "tileBorder" = None):

        # utility function to concat the gene lists of two labels
        def concatGeneDicts(ref_gene_dict, target_gene_dict, ref_label, target_label):
            # iterate over target and add their counts to the ref
            for gene, count in target_gene_dict[target_label].items():
                ref_gene_dict[ref_label][gene] = ref_gene_dict[ref_label].get(gene, 0) + count 

        # Actual count matrix creation, including getting the counts from top and left
        def assignGenesToSpots(labeled_image, decoded_df):
            # First get all genes for this tile
            gene_dict = {} # keys = label integers, values = dict{gene: count}
            n_labels = np.unique(labeled_image)
            for label in n_labels:
                gene_dict[label] = {}
            for row in decoded_df.itertuples():
                try:
                    label = labeled_image[row.local_row, row.local_col]
                    if label != 0:
                        gene_dict[label][row.Gene] = gene_dict[label].get(row.Gene, 0) + 1 
                except:
                    pass

            this_border = self.getTileBorder()
            # then get genes of the tile to the left
            if left_tile is not None:
                left_border = left_tile.getTileBorder()
                left_matches = this_border.matchTileBorders(left_border, self.grid)

                for this_label, left_label in left_matches.items():
                    left_gene_dict = left_tile.getGenesOfLabel(left_label)
                    concatGeneDicts(gene_dict, left_gene_dict, this_label, left_label)


            if top_tile is not None:
                top_border = top_tile.getTileBorder()
                top_matches = this_border.matchTileBorders(top_border, self.grid)

                for this_label, top_label in top_matches.items():
                    top_gene_dict = top_tile.getGenesOfLabel(top_label)
                    concatGeneDicts(gene_dict, top_gene_dict, this_label, top_label)

            return gene_dict

        gene_dict = assignGenesToSpots(self.labeled_image, self.detected_genes_df)

        gene_dict_list = [gene_dict[i] for i in sorted(gene_dict.keys())]

        keys = set().union(*gene_dict_list)
        final = {k: [d.get(k, 0) for d in gene_dict_list] for k in keys}

        count_matrix = pd.DataFrame(final)

        return count_matrix


    def getTileBorder(self):
        return tileBorder(self.labeled_image, self.tile_nr)


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
        # axs[i-1].set_title(f"with {i+1}: {list(tile2_matches.keys())} ; with {i + tile_grid.coldiv}: {list(tile3_matches.keys())}")
        axs[i-1].set_xticks([])
        axs[i-1].set_yticks([])

    plt.show()

if __name__ == '__main__':
    grid = tileGrid(4,4,2048,2048)
    # grid.getNeighbouringTiles(2)

    plotLabeledImageOverlap(glob.glob("../out_dir/labeled1_MERFISH_nuclei_tile*.tif"), grid)

