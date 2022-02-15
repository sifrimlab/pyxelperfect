import pandas as pd
import numpy as np
from skimage import measure

def measureLabeledImage(labeled_image: np.array, original_DAPI_image: np.array) -> pd.DataFrame:
    regions = measure.regionprops(labeled_image, intensity_image=original_DAPI_image)

    propList = ['Area',
                'bbox']
    rows_list=[]
    for region_props in regions:
        attribute_dict = {}
        center_y, center_x= region_props['centroid']
        attribute_dict['image_label'] =region_props['Label']
        attribute_dict['cell_label'] = f"X{int(center_x)}_Y{int(center_y)}_{region_props['Label']}"
        attribute_dict['center_X'] = int(center_x)
        attribute_dict['center_Y'] = int(center_y)
        for i,prop in enumerate(propList):
            attribute_dict[prop] = region_props[prop]
        rows_list.append(attribute_dict)
    attribute_df = pd.DataFrame(rows_list)

    return attribute_df

if __name__ == "__main__":
    image_path = "/media/gojira/MERFISH_datasets/download.brainimagelibrary.org/02/26/02265ddb0dae51de/mouse1_sample2_raw/extracted_aligned_images/aligned_images_tile101_DAPI.tiff"
    image = io.imread(image_path)
    label_image, attribute_df = otsuSegment(image)
    plt.imshow(label_image)
    plt.show()
    # print(attribute_df)

