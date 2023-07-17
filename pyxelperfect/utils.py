import math
import re
import numpy as np
from pathlib import Path
from skimage import io
import pandas as pd
from scipy.ndimage import distance_transform_edt
from glob import glob


def getPatch(arr, idx, radius=3, fill=None):
    """
    Gets surrounding elements from a numpy array

    Parameters:
    arr (ndarray of rank N): Input array
    idx (N-Dimensional Index): The index at which to get surrounding elements. If None is specified for a particular axis,
        the entire axis is returned.
    radius (array-like of rank N or scalar): The radius across each axis. If None is specified for a particular axis,
        the entire axis is returned.
    fill (scalar or None): The value to fill the array for indices that are out-of-bounds.
        If value is None, only the surrounding indices that are within the original array are returned.

    Returns:
    ndarray: The surrounding elements at the specified index
    """

    assert len(idx) == len(arr.shape)

    if np.isscalar(radius):
        radius = tuple([radius for i in range(len(arr.shape))])

    slices = []
    paddings = []
    for axis in range(len(arr.shape)):
        if idx[axis] is None or radius[axis] is None:
            slices.append(slice(0, arr.shape[axis]))
            paddings.append((0, 0))
            continue

        r = radius[axis]
        l = idx[axis] - r
        r = idx[axis] + r

        pl = 0 if l > 0 else abs(l)
        pr = 0 if r < arr.shape[axis] else r - arr.shape[axis] + 1

        slices.append(slice(max(0, l), min(arr.shape[axis], r+1)))
        paddings.append((pl, pr))

    if fill is None:
        return arr[tuple(slices)]
    return np.pad(arr[tuple(slices)], paddings, 'constant', constant_values=fill)

def getLastDimensions(array, nr_dims: int = 2):
    """Extracts last X dimensions from an array

    Parameters
    ----------
    array :
        input array to be extracted
    nr_dims : int
        nr of dimensions to extract. Default = 2 
    """
    if array.ndim <= nr_dims:
        return array
    if nr_dims > nr_dims:
        raise ValueError(f"Requested nr of dimensions {nr_dims} higher than nr of dimensions present in array.")
    slc = [0] * (array.ndim - nr_dims)
    slc += [slice(None) for i in range(0, nr_dims)]
    return array[tuple(slc)]

#TODO check userfulness
def isolateSingleCellsFromTile(measured_df_basename: str , image_basename: str, grid: "tileGrid", tile_nr, out_dir: str or Path = None, image_prefix: str = "", labeled_image_basename = None, GEimage_basename = None):

    # Input parsing
    if image_prefix and not image_prefix.endswith("_"):
        image_prefix += "_"

    if not out_dir:
        out_dir = Path(".")
    else:
        out_dir = Path(out_dir)

    out_dir.mkdir(exist_ok=True)

    if GEimage_basename is not None:
        GE_out_dir = out_dir / "gene_expression_imgs/"
        GE_out_dir.mkdir(exist_ok=True)

    regular_dir = out_dir / "regular_bbox/"  
    regular_dir.mkdir(exist_ok=True)

    # first we build the image
    middle_tile = io.imread(f"{image_basename}{tile_nr}.tif")
    measured_df = pd.read_csv(f"{measured_basename}{tile_nr}.csv")

    neighbours = grid.getNeighbouringTiles(tile_nr)

    all_4_shape =(grid.tile_size_row * 3, grid.tile_size_col * 3) # If all neighbours exist
    empty = np.zeros(all_4_shape)

    # create list of neighbours such that looping is ordered 
    # Now fill in the images in the large empty image
    row_size = grid.tile_size_row
    col_size = grid.tile_size_col
    if neighbours["top"] is not None:
        top_neighbour =  io.imread(f"{image_basename}{neighbours['top']}.tif")
        top_measured_df = pd.read_csv(f"{measured_basename}{neighbours['top']}.csv")
        empty[0:row_size, col_size: 2*col_size] = top_neighbour

    if neighbours["left"] is not None:
        left_neighbour =  io.imread(f"{image_basename}{neighbours['left']}.tif")
        left_measured_df = pd.read_csv(f"{measured_basename}{neighbours['left']}.csv")
        empty[row_size : 2*row_size, 0:col_size] =left_neighbour

    if neighbours["right"] is not None:
        right_neighbour =  io.imread(f"{image_basename}{neighbours['right']}.tif")
        right_measured_df = pd.read_csv(f"{measured_basename}{neighbours['right']}.csv")
        empty[row_size : 2*row_size, 2*col_size : 3*col_size] =right_neighbour

    if neighbours["bot"] is not None:
        bot_neighbour =  io.imread(f"{image_basename}{neighbours['bot']}.tif")
        bot_measured_df = pd.read_csv(f"{measured_basename}{neighbours['bot']}.csv")
        empty[2*row_size : 3*row_size, col_size : 2*col_size] = bot_neighbour

    # Now add the middle tile as well
    empty[row_size : 2*row_size, col_size : 2*col_size] = middle_tile


    # then we adapt the center locations in the context of our custom 3x3 grid 
    def adaptCoordsOfMiddleTile(row, col, grid):
        new_row = row + grid.tile_size_row
        new_col = col + grid.tile_size_col

        return new_row, new_col
    def adaptCoordsOfBottomTile(row, col, grid):
        new_row = row + 2*grid.tile_size_row
        new_col = col + grid.tile_size_col

        return new_row, new_col
    def adaptCoordsOfRightTile(row, col, grid):
        new_row = row + grid.tile_size_row
        new_col = col + 2*grid.tile_size_col

        return new_row, new_col

    plt.imshow(empty)
    # plt.show()
    # Check which labels shouldn't be included in this iteration of isolation (so those bot and right)
    middle_border = tileBorder(middle_tile, tile_nr)

    ## Take the list of the labels of objects that are located on the top and left border 
    labels_to_leave_out = (*middle_border.orientation_label_centers["top"].values(), *middle_border.orientation_label_centers["left"].values())
    labels_on_the_border =  (*middle_border.orientation_label_centers["bot"].values(), *middle_border.orientation_label_centers["right"].values())

    #TODO last step: align centers for new_row, new_col, it doesn't yet because the cells on the border's center of gravity is incorrect when measured on just the tile
    # Maybe here we take the corresponding label and just calculate the middle of the 2
    # tile = tileBorder(img, i)
    # try:
    #     img2 =  io.imread(labeled_image_paths[i])
    #     tile2 = tileBorder(img2, i+1)
    #     tile2_matches = tile.matchTileBorders(tile2, tile_grid)
    def calcMidpointByArea(point1, point2, point1_ratio, point2_ratio):
        assert point1_ratio + point2_ratio == 1.0
        return ((point1[0] + point2[0])/2, (point1[1] + point2[1])/2)
    def weighted_midpoint(center1, area1, center2, area2):
        weighted_center1 = tuple(x * area1 for x in center1)
        weighted_center2 = tuple(x * area2 for x in center2)
        total_area = area1 + area2
        weighted_midpoint = tuple((x + y) / total_area for x, y in zip(weighted_center1, weighted_center2))
        return weighted_midpoint

     ## # then we can loop over the df and extract the images from the enlarged image
    for i, cell_row in enumerate(measured_df.itertuples()):
        if cell_row.image_label not in labels_to_leave_out:
            if cell_row.image_label in labels_on_the_border:
                row = cell_row.center_Y
                col = cell_row.center_X
                new_row, new_col = adaptCoordsOfMiddleTile(row, col, grid)
                # If this cell is on the border, it's center is wrong, so we find the corresponding label on the neighbouring tile and calculate the mean distance between the two as new center of the mosaic
                if cell_row.image_label in middle_border.orientation_label_centers["bot"].values(): # if the label is on the bottom we need to look at bottom label
                    # first create border of neighbour and find matching label
                    bot_border = tileBorder(bot_neighbour, neighbours["bot"])
                    bot_matches = middle_border.matchTileBorders(bot_border, grid)
                    matching_label = bot_matches[cell_row.image_label]
                    print(bot_measured_df.loc[bot_measured_df['image_label'] == matching_label]['center_X'])
                    matching_label_center_row, matching_label_center_col = bot_measured_df.loc[bot_measured_df['image_label'] == matching_label]['center_Y'][0],  bot_measured_df.loc[bot_measured_df['image_label'] == matching_label]['center_X'][0]
                    # then adapt that label's coordinates to the current 3x3 coordinate system
                    new_matching_label_center_row, new_matching_label_center_col  = adaptCoordsOfBottomTile(matching_label_center_row, matching_label_center_col, grid)

                    # Then define a new point in between this and matching center, weighted by their areas
                    current_point_area = cell_row.Area
                    matching_point_area = bot_measured_df.loc[bot_measured_df['image_label'] == matching_label]['Area'][0]
                    midpoint_row, midpoint_col = weighted_midpoint((new_matching_label_center_row, new_matching_label_center_col), matching_point_area, (new_row,new_col), current_point_area) 
                    plt.scatter((new_col, new_matching_label_center_col, midpoint_col), (new_row, new_matching_label_center_row, midpoint_row), color=('red', 'green', 'orange'))
                    # plt.show()

            else:
                pass
                # row = cell_row.center_Y
                # col = cell_row.center_X
                # new_row, new_col = adaptCoordsOfMiddleTile(row, col, grid)
                # bb = getPatch(empty, (new_row,new_col), radius = (149,149), fill=0)
                # io.imsave(regular_dir / f"{image_prefix}cell{i}.tif", bb)
                # if labeled_image is not None:
                #     labeled_bb = getPatch(labeled_image, (Y,X), radius = (74,74), fill=0)
                #     bb[labeled_bb != cell_row.image_label] = 0 
                # io.imsave(out_dir / f"{image_prefix}cell{i}.tif", bb)

                # if GEimage is not None:
                #     gene_labeled_bb = getPatch(GEimage, (Y,X), radius = (74,74), fill=0)
                #     gene_labeled_bb[labeled_bb != cell_row.image_label] = 0
                #     io.imsave(GE_out_dir / f"{image_prefix}gene_labeled{i}.tif", gene_labeled_bb)
                # plt.imshow(bb)
                # plt.title(f"tile {tile_nr}, label: {cell_row.image_label}")
            # plt.gca().invert_yaxis()
            # plt.scatter((col, new_col), (row, new_row), color=('red', 'green'))
            # plt.show()
    # plt.show()

def sortByRegex(target_list, regex):
    r = re.compile(rf"{regex}")
    def key_func(m):
        return int(r.search(m).group(1))

    target_list.sort(key=key_func)
    return target_list

def findCenter(x_shape):
    middle = x_shape[0] / 2, x_shape[1] / 2
    return middle

def exceptGlob(full_glob: str, except_glob: str):
    return list(set(glob(full_glob)) - set(glob(except_glob)))

def expand_labels_manual(label_image, distance=1):
    distances, nearest_label_coords = distance_transform_edt(
        label_image == 0, return_indices=True
    )
    labels_out = np.zeros_like(label_image)
    dilate_mask = distances <= distance

    masked_nearest_label_coords = [
        dimension_indices[dilate_mask]
        for dimension_indices in nearest_label_coords
    ]
    nearest_labels = label_image[tuple(masked_nearest_label_coords)]
    labels_out[dilate_mask] = nearest_labels
    return labels_out, distances, nearest_label_coords

def expand_labels_on_something_else(label_image, distances, nearest_label_coords, distance=1):
    labels_out = np.zeros_like(label_image)
    dilate_mask = distances <= distance
    masked_nearest_label_coords = [
        dimension_indices[dilate_mask]
        for dimension_indices in nearest_label_coords
    ]
    nearest_labels = label_image[tuple(masked_nearest_label_coords)]
    labels_out[dilate_mask] = nearest_labels
    return labels_out

if __name__ == '__main__':
    import pickle
    import matplotlib.pyplot as plt
    from tiling import tileGrid, tileBorder

    image_basename = "../out_dir/labeled1_MERFISH_nuclei_tile"
    measured_basename = "../out_dir/labeled1_MERFISH_nuclei_measured_tile"
    with open("../out_dir/tile_grid.pickle", 'rb') as f:
        grid = pickle.load(f)
    # grid = tileGrid(4,4,2048, 2048)
    for i in range(1,17):
        print(f"{i} iteration")
        isolateSingleCellsFromTile(measured_basename, image_basename, grid, i, out_dir = "../out_dir/isolated_images/")
        
