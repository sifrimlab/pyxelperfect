from skimage import io
import pandas as pd
from skimage.draw import disk, rectangle
from skimage.transform import rotate, resize, warp
from pyxelperfect.decorators import  measureTime
import SimpleITK as sitk
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import ast

"""
This module assumes 'point': len(tuple) == 2 with row and col being the fir and second element respectively.
All functions are structured in a way that they take only 2 arguments, point and value, where value is everything the function needs to do is thing.
This is to make sure that registerPoint can handle an arbitrary amount of different instructions, at the expense of readability.
"""

def loadRegisterKwargsFromCsv(path_to_csv):
    ## Ordered processing steps will be a list of single element dicts, where key = step and value = value. registerPoint will have to be adapted with a mapping that maps step to their respective functions
    def readFlip(value):
        return ast.literal_eval(value)
    def readBbox(value):
        return ast.literal_eval(value)
    # usually when transforming a point it's for transforming a point belonging to the target coordinate axis, so we assume we want an inverse transform
    def readTransform(value):
        if "mha" in value:
            return sitk.DisplacementFieldTransform(sitk.ReadImage(value))
        else:
            return sitk.ReadTransform(value).GetInverse()
    def readRotation(value):
        rotation = ast.literal_eval(value)
        center = rotation[0]
        angle = rotation[1]
        if len(rotation) > 2:
            translation = rotation[2]
        else:
            translation = (0,0)
        return {"center": center, "angle": angle, "translation": translation}
    def readZoomFactors(value):
        return ast.literal_eval(value)

    def readTranslation(value):
        return ast.literal_eval(value)

    functions = {"flip": readFlip, "bbox": readBbox, "rotation":readRotation, "zoom_factors": readZoomFactors, "transform": readTransform, "translation": readFlip}


    df = pd.read_csv(path_to_csv)
    ordered_processing_steps = []
    for row in df.itertuples():
        # The value of the single element dict is already the parsed argument, as to not load in transforms multiple times when registering multiple points
        ordered_processing_steps.append({row.step: functions[row.step](row.value)})

    return ordered_processing_steps


def registerPoint(point, args_list):
    args_dict = {"flip": flipPoint, "rotation": rotatePointDict, "bbox": transformPointsWithBbox, "zoom_factors": resizePointWithTransform, "transform": sitkTransformPoint, "translation": translatePoint}
    for arg in args_list:
        for key, value in arg.items():
            # arg = dict and will only have one item
            point = args_dict[key](point, value)
            if isinstance(point, type(None)):
                return (0,0)
    return point

def findCenter(x_shape):
    middle = x_shape[0] / 2, x_shape[1] / 2
    return middle

def rotatePointDict(point, arg_dict: dict):
    # Duplicate of rotatePoint but with takes input as a dict. This just to facilitate the function dict used in registerPoint

    origin, angle= arg_dict["center"], arg_dict["angle"]
    try:
        translation = arg_dict["translation"]
    except:
        translation = (0,0)

    point = (point[0] - translation[0], point[1] - translation[1])

    if isinstance(angle, int):
        angle = math.radians(angle)
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return (qx, qy)

def rotatePoint(point, origin, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    If angle is an integer, it is assumed to be degrees and will be converted to radians
    """

    if isinstance(angle, int):
        angle = math.radians(angle)

    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return (qx, qy)

def cutBboxFromImage(bbox, image: np.ndarray):
    min_row, min_col, max_row, max_col = bbox

    cut_image = image[min_row:max_row, min_col:max_col]
    return cut_image

def getShapeFromBbox(bbox):
    min_row, min_col, max_row, max_col = bbox
    return (max_row, max_col)

def point2int(point):
    return tuple([round(el) for el in point])

def transformPointsWithBbox(point, bbox, force = False):
    min_row, min_col, max_row, max_col = bbox
    if min_row <= point[0] < max_row and min_col <= point[1] < max_col:
        new_row = point[0] - min_row
        new_col = point[1] - min_col
        return (new_row, new_col)
    else:
        if not force:
            return None
        else:
            # If it doesn't fit, we find the closest point to the bbox that does fit
            closest_row = min(max(min_row, point[0]), max_row)
            closest_col = min(max(min_col, point[1]), max_col)
        
            return closest_row - min_row, closest_col - min_col

def calculateResizeZoomFactors(original_shape, target_shape):
    """
    Calculates the zoom factors necessary to go from original_shape to target shape
    """
    tform = [] 
    for i in range(len(original_shape)):
        tform.append(target_shape[i] / original_shape[i])
    zoom_factors = np.array(tform)
    return zoom_factors

def resizePointWithTransform(point, zoom_factors):
    """resizePointWithTransform.
    Parameters
    ----------
    point :
        2D point in form of a tuple
    zoom_factors :
        transform in the form of the zoom factors used in the ndi.zoom function by scipy
    """

    point = np.array([point]) # add a second artificial dimension to it
    new_point = np.array(zoom_factors) * point
    new_point = (float(new_point[0,0]), float(new_point[0,1]))
    return new_point

def reflectPoint180degrees(point, origin):
    new_point = 2*origin[0] - point[0], 2*origin[1] - point[1]
    return new_point

def flipPoint(point, shape):
    """
    Flips a point along a specific axis. 
    shape = [n_rows, n_cols, axis],
    where if axis  == 0: no flip
    axis == 1: flip along the rows | vertically | mirror over a horizontal line 
    axis == 2: flip along the cols | horizonatlly | mirror over a vertical line 
    axis == 3: flip along both rows and cols | vertically  and horizontally | mirror over both a horizontal and vertical line 
    """
    img_shape, axis = shape[:2], shape[2]
    # img_shape = shape[:2]
    
    if axis == 0: 
        return point
    elif axis == 1:
        new_row = (img_shape[0] - point[0]) - 1
        new_col = point[1]
        new_point = (new_row, new_col)

    elif axis == 2:
        new_col = (img_shape[1] - point[1]) -1
        new_row = point[0]
        new_point = (new_row, new_col)

    elif axis == 3:
        new_row = (img_shape[0] - point[0]) - 1
        new_col = (img_shape[1] - point[1]) -1
        new_point = (new_row, new_col)
    return new_point

def sitkTransformPoint(point, sitk_transform):
    """sitkTransformPoint.
    -   sitk.transform.TransformPoint assumes (col, row) under the hood, so we invert them beforehand and invert them back after transform
    Parameters
    ----------
    point :
        tuple: (row, col)
    sitk_transform :
        sitk_transform
    """
    if isinstance(sitk_transform, str):
        sitk_transform = sitk.ReadTransform(sitk_transform)
    inverted_point = (point[1], point[0])
    transformed_point = sitk_transform.TransformPoint(inverted_point)
    return_point = (transformed_point[1], transformed_point[0])
    return return_point

def translatePoint(point, offset):
    new_point =(point[0] + offset[0] , point[1] + offset[1])
    return new_point
