import pandas as pd
from skimage import io
import pickle
import matplotlib.pyplot as plt
import numpy as np

from .display import plotSpatial
from .decorators import measureTime
from .measure import measureLabeledImage

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

    
    def createCountMatrix(self, left_tile: tileBorder = None, top_tile: tileBorder = None):

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


if __name__ == '__main__':
    for i in range(1,grid.n_tiles+1):
        # grid.getTileDataCoordinates(i, rowname="y", colname="x").to_csv(f"./out_dir/merfish_decoded_genes_tile{i}.csv")
        with open("./out_dir/tile_grid.pickle", 'rb') as f:
            grid = pickle.load(f)
        img = io.imread(f"./out_dir/labeled1_MERFISH_nuclei_tile{i}.tif")
        df = pd.read_csv(f"./out_dir/merfish_decoded_genes_tile{i}.csv")
        measured_df = pd.read_csv(f"./out_dir/labeled1_MERFISH_nuclei_measured_tile{i}.csv")
        tile = Tile(img, df, measured_df, grid, i)

        try:
            img1 = io.imread(f"./out_dir/labeled1_MERFISH_nuclei_tile{i-1}.tif")
            df1 = pd.read_csv(f"./out_dir/merfish_decoded_genes_tile{i-1}.csv")
            measured_df1 = pd.read_csv(f"./out_dir/labeled1_MERFISH_nuclei_measured_tile{i-1}.csv")
            tile1 = Tile(img1, df1, measured_df, grid, i-1)
        except:
            tile1=None

        try:
            img2 = io.imread(f"./out_dir/labeled1_MERFISH_nuclei_tile{i - grid.coldiv}.tif")
            df2 = pd.read_csv(f"./out_dir/merfish_decoded_genes_tile{i - grid.coldiv}.csv")
            measured_df2 = pd.read_csv(f"./out_dir/labeled1_MERFISH_nuclei_measured_tile{i - grid.coldiv}.csv")
            tile2 = Tile(img2, df2, measured_df, grid, i-grid.coldiv)
        except:
            tile2 = None

        count_matrix = tile.createCountMatrix(tile1, tile2)
        print(count_matrix)
        



