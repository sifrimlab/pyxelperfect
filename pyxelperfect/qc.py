import numpy as np
from skimage import io
from skimage.transform import resize
from skimage.feature import canny
from skimage.filters import sobel
from skimage.transform import hough_circle, hough_circle_peaks, hough_line, hough_line_peaks
from skimage.util import img_as_ubyte
from glob import glob
from .diagnose import getImageStats
import re
import math

def hasExactlyOneLine(img: np.ndarray, sigma: int = 0, low_threshold:float = None, high_threshold: float=None):
    """Detects the presence of a vertical or horizontal edge in an image. 
    To be used when filtering out edge-tiles from a dataset.

    Parameters
    ----------
    img : np.ndarray
        img
    sigma : int
        sigma to be used for gaussian blurring by canny edge filter. Default = 0
        For more documentation, see https://scikit-image.org/docs/dev/api/skimage.feature.html
    low_threshold : float
        Low threshold to be used by hysteresis thresholding. 
        For more documentation, see https://scikit-image.org/docs/dev/api/skimage.feature.html
    high_threshold : float
        High threshold to be used by hysteresis thresholding. 
        For more documentation, see https://scikit-image.org/docs/dev/api/skimage.feature.html
    """

    edge_image = canny(img, sigma=0, low_threshold = low_threshold, high_threshold = high_threshold)
    tested_angles = np.array([np.pi, np.pi / 2])
    h, theta, d = hough_line(edge_image, theta=tested_angles)
    accums, angles, dists =  hough_line_peaks(h, theta, d)
    return len(accums) == 1

def hasExactlyOneCircle(img: np.ndarray, min_ydistance:int = 50, min_xdistance:int = 50):
    """Checks whether the input image contains exactly one circle-like object
    Note: For this, it is important that the input img is 16bit.

    Parameters
    ----------
    img : np.ndarray
        img
    min_ydistance : int
        Min ydistance that two circles to be detected can be apart.
    min_xdistance : int
        Min xdistance that two circles to be detected can be apart.
    """

    sobel_image = sobel(img)
    sobel_image = img_as_ubyte(sobel_image)
    hough_radii = np.arange(25, 75, 2)
    hough_res = hough_circle(sobel_image, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, min_ydistance = 50, min_xdistance= 50 )
    return len(radii) == 1


def isGoodQuality(image: np.ndarray):
    df = getImageStats(image, out_print=False)
    if df["Actual range"][0] == (0,0):
        return False
    elif df["Actual range"][0][0] != 0:
        return False
    elif df["Otsu threshold"][0] < 0:
        return False
    elif float(df["Laplace std"][0]) < 0.001:
        return False
    elif df["Otsu threshold"][0] < 100:
        return False 
    elif df["Otsu threshold"][0] > 1500:
        return False 
    else:
        return True
if __name__ == "__main__":
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt     
    full_image_list = glob("/home/david/.config/nnn/mounts/nacho@10.38.76.144/tool/enteric_neurones/slidescanner_examples/tiles/*/*.tif")
    y_labs = ["good" in file for file in full_image_list ]
    y_preds =  [isGoodQuality(io.imread(file)) for file in full_image_list]
    correct_array = [pred == label for pred, label in zip(y_preds, y_labs)]
    wrong_images = [b for a, b in zip(correct_array, full_image_list) if not a]
    right_images = [b for a, b in zip(correct_array, full_image_list) if a]
    right_bad_images = [b for a, b in zip(correct_array, full_image_list) if a and "bad" in b]
    right_good_images = [b for a, b in zip(correct_array, full_image_list) if a and "good" in b]
    wrong_bad_images = [b for a, b in zip(correct_array, full_image_list) if not a and "bad" in b]
    wrong_good_images = [b for a, b in zip(correct_array, full_image_list) if not a and "good" in b]

    to_plot = wrong_good_images
    title = "wrong_good_images"
    # Testing dynamic plotting
    tot = len(to_plot)
    cols = round(math.sqrt(tot))
    rows = tot // cols 
    rows += tot % cols

    position = range(1, tot+1)

    fig = plt.figure(1)
    st = fig.suptitle(title, fontsize="x-large")
    for k in range(tot):
        ax = fig.add_subplot(rows, cols, position[k])
        image = io.imread(to_plot[k])
        ax.imshow(resize(image, (image.shape[0] // 5, image.shape[1] // 5),anti_aliasing=True))
        ax.set_title(re.findall(r'Slide\d+-\d+-\d+', to_plot[k])[0]  + "-"  + re.findall(r'tile\d+', to_plot[k])[0])
        ax.axis("off")
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)
    plt.show()

    # Confusion matrix
    # cm = confusion_matrix(y_labs, y_preds, labels=[True, False])
    # ax= plt.subplot()
    # sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

    # # labels, title and ticks
    # ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    # ax.set_title('Confusion Matrix'); 
    # ax.xaxis.set_ticklabels(['Good', 'Bad']); ax.yaxis.set_ticklabels(['Good', 'Bad']);

    # plt.show()
