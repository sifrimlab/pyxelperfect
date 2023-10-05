import cv2
import numpy as np
import pandas as pd
from skimage import io
from skimage import color
import matplotlib.pyplot as plt
from skimage.color import label2rgb
import matplotlib.patches as mpatches
from skimage.util import img_as_ubyte
from skimage.segmentation import find_boundaries
from .manipulate import automaticBrightnessAndContrast

def makeHist(data, mn, mx, interval):
    n, bins, patches = plt.hist(x=data, bins=np.arange(mn, mx, interval)-interval/2, color='navy',
                            alpha=0.7, rwidth=0.85)

    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Nr mitf spots')
    plt.xticks(np.arange(mn,mx,interval))
    plt.ylabel('Frequency')
    maxfreq = n.max()
    plt.show()

def showSegmentation(labeled_image: np.array, original_image: np.array = None, save=False, plot=True, overlay=False):
    colored_image = color.label2rgb(labeled_image, bg_label=0)
    if original_image is not None:
        original_image = img_as_ubyte(original_image)

    colored_image_on_DAPI = color.label2rgb(labeled_image, original_image, bg_label=0)
    if save:
        io.imsave("labeled_image.tif", colored_image_on_DAPI)

    if plot:
        if original_image is not None:
            if not overlay:
                fig, axs = plt.subplots(1,2, sharex=True, sharey=True)
                axs[0].imshow(original_image)
                axs[1].imshow(colored_image_on_DAPI)
                plt.axis("off")
                plt.show()
            elif overlay:
                plotBoundaries(labeled_image, original_image)

        else:
            plt.imshow(colored_image_on_DAPI)
            plt.axis("off")
            plt.show()



def showImage(image: np.array):
    plt.imshow(image)
    plt.axis("off")
    plt.show()

def createOverlay(labeled_image, image):
    max_val = np.amax(image) 
    tmp_image = image.copy()
    tmp_image[find_boundaries(labeled_image)] = max_val
    return tmp_image

def plotBoundaries(labeled_image,image):
    plt.imshow(createOverlay(labeled_image, image))
    plt.axis("off")
    plt.show()


def createComposite(img1_path, img2_path):
    A = cv2.cv2.imread(img1_path, cv2.cv2.IMREAD_ANYDEPTH)
    B = cv2.cv2.imread(img2_path, cv2.cv2.IMREAD_ANYDEPTH)
    bitDepth = A.dtype
    # print(A.dtype, B.dtype)
    zeros = np.zeros(A.shape[:2], dtype=bitDepth)

    ##Don't forget, cv2.cv2 works with a different order sometimes, so it's not RGB it's BGR
    merged = cv2.cv2.merge((A,B,zeros))
    merged = img_as_ubyte(merged)
    return merged

def evaluateRegistration(ref_image, target_image, registered_target, identifier: str = "", plot=True):
    original_situation = createComposite(ref_image, target_image)
    original_situation = img_as_ubyte(original_situation)
    registered_situation = createComposite(ref_image, registered_target)
    registered_situation = img_as_ubyte(registered_situation)

    if identifier:
        io.imsave(f"overlay_registered_{identifier}.tif", registered_situation)
    if plot:
        fig, axs = plt.subplots(1,2)
        axs[0].imshow(original_situation)
        axs[0].set_title("Before Registration")
        red_patch = mpatches.Patch(color='red', label='MS ref')
        green_patch = mpatches.Patch(color='green', label='ST target')
        axs[0].legend(handles=[red_patch, green_patch])
        axs[1].imshow(registered_situation)
        axs[1].set_title("After Registration")
        green_patch_2 = mpatches.Patch(color='green', label='ST moved')
        axs[1].legend(handles=[red_patch, green_patch_2])
        plt.show()

def plotSpatial(image, dataframe, rowname = "row", colname="col", dotsize=5, color="red", save=False, plot=True, ax=None, colormap="tab20"):

    def get_cmap(n, name='tab20'):
    ##Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    ##RGB color; the keyword argument name must be a standard mpl colormap name.
        return plt.cm.get_cmap(name, n)


    if ax is None:
        fig, ax = plt.subplots(1,1)
    ax.imshow(image, cmap="gray")
    if color in dataframe.columns:
        unique_els = dataframe[color].unique()

        cmap = get_cmap(len(unique_els), name=colormap)
        patches = []
        for i, c in enumerate(unique_els):
            tmp_df = dataframe[dataframe[color] == c]
            if isinstance(dotsize, str):
                tmp_dotsize = np.array(tmp_df[dotsize].astype(float))
            else:
                tmp_dotsize = dotsize
            curr_col = cmap(i)
            ax.scatter(tmp_df[colname], tmp_df[rowname], color=curr_col, s=tmp_dotsize)
            patches.append(mpatches.Patch(color=curr_col, label=c))
    else:
        if isinstance(dotsize, str):
            tmp_dotsize = np.array(tmp_df[dotsize].astype(float))
        else:
            tmp_dotsize = dotsize
        ax.scatter(dataframe[colname], dataframe[rowname], color=color, s=tmp_dotsize)

    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

    if save:
        plt.savefig("spatial_plot.png")
    if plot:
        plt.show()

    return ax


def plotObjectPerAreaBin(labeled_image, measured_df, mn, mx, interval):
    bins = list(range(mn, mx, interval))
    bins.append(float("inf"))

    str_labels = []
    for i, el in enumerate(bins):
        if not i == len(bins)-2:
            str_labels.append(f"{el}-{bins[i+1]}")
        else:
            str_labels.append(f"{bins[-2]}+")
            break
    print(str_labels)

    def get_cmap(n, name='tab10'):
        ##Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
        ##RGB color; the keyword argument name must be a standard mpl colormap name.
        return plt.cm.get_cmap(name, n)


    color_labels= []
    global_cmap = get_cmap(len(bins))

    for i in range(len(bins)-1):
        color_labels.append(global_cmap(i))

    area_interval_colors = pd.cut(measured_df["Area"], bins=bins, labels=color_labels)
    area_interval_strings = pd.cut(measured_df["Area"], bins=bins, labels=str_labels)
    measured_df["area_interval_colors"] = area_interval_colors
    measured_df["area_interval_strings"] = area_interval_strings

    plt.imshow(label2rgb(labeled_image, colors=measured_df["area_interval_colors"]))
    colors = color_labels
    # create a patch (proxy artist) for every color 
    patches = [ mpatches.Patch(color=colors[i], label=str_labels[i]) for i in range(len(str_labels)) ]
    plt.axis("off")
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )




