import numpy as np
from skimage import io
from glob import glob
from diagnose import getImageStats

def isGoodQuality(image: np.ndarray):
    df = getImageStats(image, out_print=False)

    if df["Actual range"][0] == (0,0):
        return False
    elif df["Laplace std"][0] < 0.001:
        return False
    elif df["Actual range"][0][0] == 0:
        return False
    elif df["Otsu threshold"][0] < 0:
        return False
    else:
        return True
if __name__ == "__main__":
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt     
    full_image_list = glob("/home/david/.config/nnn/mounts/nacho@10.38.76.144/amenra/single_nuclei/LiverSampleQC/*/*")
    y_labs = [True if "good" in file else False for file in full_image_list ]
    y_preds =  [isGoodQuality(io.imread(file)) for file in full_image_list ]
    correct_array = [pred == label for pred, label in zip(y_preds, y_labs)]
    wrong_images = [b for a, b in zip(correct_array, full_image_list) if not a]
    right_images = [b for a, b in zip(correct_array, full_image_list) if a]

    fig, axs = plt.subplots(1,len(right_images))
    for i,path in enumerate(right_images):
        axs[i].imshow(io.imread(path))
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
