import numpy as np
from skimage import io
from PIL import Image
from matplotlib import pyplot as plt
import re
from typing import Tuple, List
import cv2

def calculateTileGridStatistics(tile_grid_shape, tile_size_x: int, tile_size_y: int, form="row_by_row", direction="left_to_right") -> Tuple[int]:
    """Calculates all necessary grid statistics based on tile shape and size

    Parameters
    ----------
    tile_grid_shape : Tuple[int, int]
        Tuple of int representing the number of tiles that fit in the x and y dimensions of the original image respectively.
    tile_size_x : int
        Number of pixels that in the x dimension of the original image
    tile_size_y : int
        Number of pixels that in the y dimension of the original image
    form : str
        string representing how the tiles are layed out, choose either 'row_by_row' or 'col_by_col'
    direction : str
        direction in which the tiling occurs, choose either 'left_to_right' or 'right_to_left'

    Returns
    -------
    [int] total_n_tiles
       Total number of tiles in the original image
    [ndarray] tile_grid_array
        An array representing the layout of the tiles
    [int] original_x
        length of the original image in the x dimension
    [int] original_y
        length of the original image in the y dimension
       
    """
    total_n_tiles = tile_grid_shape[0]*tile_grid_shape[1]
    # Create range list of the tiles
    total_tiles_list = list(range(1,total_n_tiles+1))
    if direction == "right_to_left":
        total_tiles_list = list(range(total_n_tiles,0, -1))

    # Reshape the range list into the tile grid of the original image.
    # We swap the elements of the grid because the rest of the pipeline sees x and y as horizontal vs vertical, but numpy sees it as an array, where x = vertical movement
    swapped_grid = (tile_grid_shape[1],tile_grid_shape[0])
    if form=="row_by_row":
        tile_grid_array = np.reshape(total_tiles_list, swapped_grid)
    if form=="col_by_col":
        tile_grid_array = np.reshape(total_tiles_list, swapped_grid)
        tile_grid_array = np.transpose(tile_grid_array)

    # Creating an empty array the size of an original image
    original_x = tile_grid_array.shape[1] * tile_size_x
    original_y = tile_grid_array.shape[0] * tile_size_y
    return total_n_tiles, tile_grid_array, original_x, original_y


def stitchImageList(image_path_list: List[str], tile_grid_shape: Tuple[int], tile_size_x: int, tile_size_y: int , imtype: str = "grey" ) -> np.array:
    """Stitch a list of images back together based on a stats describing its tiling grid

    Parameters
    ----------
    image_path_list : str
        list of paths that point to the tiles. The list should be ordered along the subsequent tiles.
    tile_grid_shape : tuple[int]
        tuple indicating the shape of the tiling grid: e.g.: (x-dim, y-dim) 
    tile_size_x : int
        size of the tiles in the x-dimension
    tile_size_y : int
        size of the tiles in the y-dimension
    imtype : str
        string depicting the type of image. Choose either grey or RGB. Default = grey

    Returns
    -------
    np.array

    """
    #  Now we create a list of the original path images, but sorted by tile number:
    # By first creating a dict where the key is the tile number, and the value is the path
    # Tile number is extracted by re.findalling on "tiled_d" and then extracting the number from that, i cast it to int to make the sorting work like it should
    tile_image_dict = {int(re.findall(r"\d+", re.findall(r"tiled_\d+", image_path)[0])[0]): image_path for image_path in image_path_list}
    sorted_tile_images = [value for (key, value) in sorted(tile_image_dict.items(), key=lambda x:x[0])]

    total_n_tiles, tile_grid_array, original_x, original_y = calculateTileGridStatistics(tile_grid_shape, tile_size_x, tile_size_y, form="col_by_col", direction="left_to_right")
    print(tile_grid_array)

    if imtype=="grey":
        new_im = Image.new('I;16', (original_x, original_y ))
    elif imtype=="RGB":
        new_im = Image.new('RGB', (original_x, original_y ))

    for index, image_path in enumerate(sorted_tile_images, 1):
        im = Image.open(image_path)
        row_location, col_location = np.where(tile_grid_array==index) # this returns rows and columns, NOT X and Y, which is the opposite
        x_offset = int(col_location)*im.size[0]
        y_offset = int(row_location)*im.size[1]
        new_im.paste(im, (x_offset,y_offset))

    numpy_image = np.asarray(new_im)
    return numpy_image
