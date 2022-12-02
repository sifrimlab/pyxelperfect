import pandas as pd
import numpy as np
from skimage import measure

def measureLabeledImage(labeled_image: np.array, original_image: np.array = None, pixels_to_um:int = 0) -> pd.DataFrame:
    regions = measure.regionprops(labeled_image, intensity_image=original_image)

    propList = ['Area',
                'bbox',
                'equivalent_diameter', 
                'orientation', 
                'MajorAxisLength',
                'MinorAxisLength',
                'Perimeter',
                'MinIntensity',
                'MeanIntensity',
                'MaxIntensity']    

    rows_list=[]
    for region_props in regions:
        attribute_dict = {}
        center_row, center_col= region_props['centroid']
        attribute_dict['image_label'] =region_props['Label']
        attribute_dict['cell_label'] = f"row{int(center_row)}_col{int(center_col)}_{region_props['Label']}"
        attribute_dict['center_row'] = int(center_row)
        attribute_dict['center_col'] = int(center_col)
        for i,prop in enumerate(propList):
            if(prop == 'Area') and pixels_to_um != 0: 
                attribute_dict['real_area'] = region_props[prop]*pixels_to_um**2
            elif (prop.find('Intensity') < 0) and pixels_to_um != 0:          # Any prop without Intensity in its name
                attribute_dict[prop] = region_props[prop]*pixels_to_um
            elif (prop.find('Intensity') < 0):
                attribute_dict[prop] = region_props[prop]
            else: 
                if original_image is not None:
                    attribute_dict[prop] = region_props[prop]

        rows_list.append(attribute_dict)
    attribute_df = pd.DataFrame(rows_list)
    return attribute_df

