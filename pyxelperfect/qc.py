import numpy as np
from skimage import io
from skimage.transform import resize
from glob import glob
from .diagnose import getImageStats
import re
import math

def isEdgeImage(image: np.ndarray):
    return np.amin(image)[0] == 0

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
