from skimage import io
import numpy as np
import pandas as pd
from skimage.filters import laplace
from skimage.util import dtype_limits
from segment import otsuThreshold



def getImageStats(image: np.ndarray, save: bool = False,out_print: bool = False) -> pd.DataFrame:
    attribute_col = []
    value_col = []

    attribute_col.append("Shape")
    value_col.append(image.shape)

    attribute_col.append("Dtype")
    value_col.append(image.dtype)

    attribute_col.append("Dtype range")
    value_col.append(dtype_limits(image))

    attribute_col.append("Mean")
    value_col.append(np.mean(image))

    attribute_col.append("Median")
    value_col.append(np.median(image))

    attribute_col.append("Laplace std")
    value_col.append(np.std(laplace(image)))

    otsu_threshold = otsuThreshold(image)
    attribute_col.append("Otsu threshold")
    value_col.append(otsu_threshold)

    above_otsu = np.sum(np.where(image > otsu_threshold, 1, 0))
    below_otsu =(image.shape[0] * image.shape[1]) - above_otsu
    perc_above = above_otsu / (image.shape[0] * image.shape[1]) * 100

    attribute_col.append("Pixels above otsu")
    value_col.append(above_otsu)

    attribute_col.append("Pixels below otsu")
    value_col.append(below_otsu)

    attribute_col.append("Percentage above otsu")
    value_col.append(perc_above)

    zipped = list(zip(attribute_col, value_col))
    result_df = pd.DataFrame(zipped, columns=['Attribute', 'Value'], index=None)
    if out_print:
        print(result_df.to_markdown())

if __name__ == '__main__':
    image = io.imread("/home/david/Documents/vsn-pipelines/starfish_data/iss/Round2/Round2_c1.TIF")
    getImageStats(image, out_print = True)


    
