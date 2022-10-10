import pandas as pd
from skimage import io
import pickle
import matplotlib.pyplot as plt
import numpy as np

from pyxelperfect.tiling import tile, tileGrid, tileBorder
from pyxelperfect.display import plotSpatial
from pyxelperfect.decorators import measureTime
# from pyxelperfect.segment import cellPoseSegment
from pyxelperfect.measure import measureLabeledImage


with open("./out_dir/tile_grid.pickle", 'rb') as f:
    grid = pickle.load(f)

grid.addDataCoordinates(pd.read_csv("./out_dir/merfish_decoded_genes.csv"))


class Tile:
    def __init__(self, labeled_image, detected_genes_df, measured_df, grid, tile_nr):
        #TODO implement a way to assign genes to cells and create a count matrix off of it
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

    
    def createCountMatrix(self, right_tile: tileBorder = None, bot_tile: tileBorder = None):

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
            # then get genes of the tile to the right
            if right_tile is not None:
                right_border = right_tile.getTileBorder()
                right_matches = this_border.matchTileBorders(right_border, self.grid)
                #Boilerplate is correct, but #TODO: It's going form i to i+1, while we should be counting backwards, since we're extracting counts from i-1

            return gene_dict


        gene_dict = assignGenesToSpots(self.labeled_image, self.detected_genes_df)

        # gene_dict_list = [gene_dict[i] for i in sorted(gene_dict.keys())]

        # keys = set().union(*gene_dict_list)
        # final = {k: [d.get(k, 0) for d in gene_dict_list] for k in keys}

        # count_matrix = pd.DataFrame(final)

        # return count_matrix

    def getTileBorder(self):
        return tileBorder(self.labeled_image, self.tile_nr)




for i in range(14,grid.n_tiles+1):
    # grid.getTileDataCoordinates(i, rowname="y", colname="x").to_csv(f"./out_dir/merfish_decoded_genes_tile{i}.csv")
    with open("./out_dir/tile_grid.pickle", 'rb') as f:
        grid = pickle.load(f)
    img = io.imread(f"./out_dir/labeled1_MERFISH_nuclei_tile{i}.tif")
    df = pd.read_csv(f"./out_dir/merfish_decoded_genes_tile{i}.csv")
    measured_df = pd.read_csv(f"./out_dir/labeled1_MERFISH_nuclei_measured_tile{i}.csv")
    tile = Tile(img, df, measured_df, grid, i)

    try:
        img1 = io.imread(f"./out_dir/labeled1_MERFISH_nuclei_tile{i+1}.tif")
        df1 = pd.read_csv(f"./out_dir/merfish_decoded_genes_tile{i+1}.csv")
        measured_df1 = pd.read_csv(f"./out_dir/labeled1_MERFISH_nuclei_measured_tile{i+1}.csv")
        tile1 = Tile(img1, df1, measured_df, grid, i+1)
    except:
        tile1=None

    try:
        img2 = io.imread(f"./out_dir/labeled1_MERFISH_nuclei_tile{i + grid.coldiv}.tif")
        df2 = pd.read_csv(f"./out_dir/merfish_decoded_genes_tile{i + grid.coldiv}.csv")
        measured_df2 = pd.read_csv(f"./out_dir/labeled1_MERFISH_nuclei_measured_tile{i + grid.coldiv}.csv")
        tile2 = Tile(img2, df2, measured_df, grid, i+grid.coldiv)
    except:
        tile2 = None


    tile.createCountMatrix(tile1, tile2)
    break
    



