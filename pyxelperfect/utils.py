import numpy as np
from pathlib import Path
from skimage import io
import pandas as pd

#tmp
import matplotlib.pyplot as plt
from tiling import tileBorder

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

    all_4_res =(grid.tile_size_row * 3, grid.tile_size_col * 3) # If all neighbours exist
    empty = np.zeros(all_4_res)

    # create list of neighbours such that looping is ordered 
    # Now fill in the images in the large empty image
    row_size = grid.tile_size_row
    col_size = grid.tile_size_col
    if neighbours["top"] is not None:
        empty[0:row_size, col_size: 2*col_size] = io.imread(f"{image_basename}{neighbours['top']}.tif")

    if neighbours["left"] is not None:
        empty[row_size : 2*row_size, 0:col_size] =  io.imread(f"{image_basename}{neighbours['left']}.tif")

    if neighbours["right"] is not None:
        empty[row_size : 2*row_size, 2*col_size : 3*col_size] =  io.imread(f"{image_basename}{neighbours['right']}.tif")

    if neighbours["bot"] is not None:
        empty[2*row_size : 3*row_size, col_size : 2*col_size] =  io.imread(f"{image_basename}{neighbours['bot']}.tif")

    # Now add the middle tile as well
    empty[row_size : 2*row_size, col_size : 2*col_size] = middle_tile


    # then we adapt the center locations 
    def adaptCoordsOfMiddleTile(row, col):
        new_row = row + grid.tile_size_row
        new_col = col + grid.tile_size_col

        return new_row, new_col

    # Check which labels shouldn't be included in this iteration of isolation (so those left and top)
    #TODO
    middle_border = tileBorder(middle_tile, tile_nr)
    print(middle_border.orientation_label_centers)


    # plt.imshow(middle_tile)
    # # then we can loop over the df and extract the images from the enlarged image
    # for i,row in enumerate(measured_df.itertuples()):
    #     X = row.center_X
    #     Y = row.center_Y
    #     new_row, new_col = adaptCoordsOfMiddleTile(Y, X)
    #     bb = getPatch(empty, (new_row,new_col), radius = (149,149), fill=0)
        # io.imsave(regular_dir / f"{image_prefix}cell{i}.tif", bb)
    #     if labeled_image is not None:
    #         labeled_bb = getPatch(labeled_image, (Y,X), radius = (74,74), fill=0)
    #         bb[labeled_bb != row.image_label] = 0 
    #     io.imsave(out_dir / f"{image_prefix}cell{i}.tif", bb)

    #     if GEimage is not None:
    #         gene_labeled_bb = getPatch(GEimage, (Y,X), radius = (74,74), fill=0)
    #         gene_labeled_bb[labeled_bb != row.image_label] = 0
    #         io.imsave(GE_out_dir / f"{image_prefix}gene_labeled{i}.tif", gene_labeled_bb)
        # plt.imshow(bb)
        # plt.show()

if __name__ == '__main__':
    # import pickle
    from tiling import tileGrid
    image_basename = "../out_dir/labeled1_MERFISH_nuclei_tile"
    measured_basename = "../out_dir/labeled1_MERFISH_nuclei_measured_tile"
    # with open("../out_dir/tile_grid.pickle", 'rb') as f:
    #     grid = pickle.load(f)
    grid = tileGrid(4,4,2048, 2048)
    isolateSingleCellsFromTile(measured_basename, image_basename, grid, 14, out_dir = "../out_dir/isolated_images/")
