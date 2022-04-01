from diagnose import getImageStats, compareImageStats
from skimage import io
import pandas as pd

# good_image_list = [io.imread(f"/home/david/.config/nnn/mounts/nacho@10.38.76.144/tool/enteric_neurones/slidescanner_examples/Good/processed_Slide2-2-2_Region0000_Channel647,555,488_Seq0017/Slide2-2-2_Region0000_Channel647,555,488_Seq0017_c1_z0_tile{i}.tif") for i in range(1, 25)]

# bad_image_list =  [io.imread(f"/home/david/.config/nnn/mounts/nacho@10.38.76.144/tool/enteric_neurones/slidescanner_examples/Bad/Slide2-1-2-Channel647,555,488_Seq00008/Slide2-1-2_Channel647,555,488_Seq0008_555_tile{i}.tif") for i in range(1, 3)]
# df_list = [getImageStats(bad_image_list[i], out_print = False, result_prefix = i).T for i in range(0,2)]
# merged_df = pd.concat(df_list)
# print(merged_df)

good_df  = compareImageStats(glob_pattern = "/home/david/.config/nnn/mounts/nacho@10.38.76.144/tool/enteric_neurones/slidescanner_examples/tiles/good_tiles/*", add_props = {"mode": "Good"}, result_prefix = "good_stats")
bad_df = compareImageStats(glob_pattern = "/home/david/.config/nnn/mounts/nacho@10.38.76.144/tool/enteric_neurones/slidescanner_examples/tiles/bad_tiles/*.tif", add_props = {"mode": "Bad"}, result_prefix = "bad_stats")





