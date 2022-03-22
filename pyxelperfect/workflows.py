import os
import SimpleITK as sitk
import pandas as pd
import cv2 as cv
import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.util import invert,img_as_ubyte, img_as_uint
from skimage.transform import rotate
import matplotlib.pyplot as plt
from .manipulate import equalizeImageSize
from .register import performRegistration
from .measure import measureLabeledImage
from .decorators import measureTime


def expandBbox(bbox, size: int):
    min_row, min_col, max_row, max_col = bbox
    min_row -= size
    max_row += size
    min_col -= size
    max_col += size
    bbox = min_row, min_col, max_row, max_col
    return bbox

def cutBboxFromImage(bbox, image: np.ndarray):
        min_row, min_col, max_row, max_col = bbox

        cut_image = image[min_row:max_row, min_col:max_col]
        #io.imsave(filename, cut_image)
        return cut_image

def isolateForeground(input_image: np.array, kernel_size:int = 100, bbox_expansion:int= 1000):
    # create labeled image
    # gray_correct = np.array(255 * (input_image / 255) ** 1.2 , dtype='uint8')
    gray_correct = img_as_ubyte(input_image)
    # Local adaptative threshold

    thresh = cv.adaptiveThreshold(gray_correct, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 255, 19)
    thresh = cv.bitwise_not(thresh)

    # # Dilatation et erosion
    kernel = np.ones((kernel_size,kernel_size), np.uint8)
    img_dilation = cv.dilate(thresh, kernel, iterations=1)
    img_erode = cv.erode(img_dilation,kernel, iterations=1)
    # # clean all noise after dilatation and erosion
    img_erode = cv.medianBlur(img_erode, 7)

    ret, labeled_image = cv.connectedComponents(img_erode)


    # io.imsave("tmp_labeledimg.tif", labeled_image)
    # Meaure imageprops
    df = measureLabeledImage(labeled_image)

    # get max area row
    max_area_label_number = df.iloc[df['Area'].idxmax()]['image_label']

    # check if the object is horizontal or vertical, horizontal if orientation < 0
    bool_rotate = df.iloc[df['Area'].idxmax()]['orientation']

    # Set everything that isnt max label to zero
    labeled_image[labeled_image!=max_area_label_number] = 0
    image_bbox =  df.iloc[df['Area'].idxmax()]['bbox']
    image_bbox = expandBbox(image_bbox, bbox_expansion)

    foreground_image = cutBboxFromImage(image_bbox, input_image)

    return foreground_image, bool_rotate

def ST_to_MSI_registration(fixed_path: str, moving_path:str):
    """ST_to_MSI_registration.

    BE WARNED. This function is CPU and Memory intensive!

    """

    # fixed_image = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
    # moving_image = sitk.ReadImage(moving_path, sitk.sitkFloat32)

    out_file_base = os.path.splitext(moving_path)[0]

    ref_image = io.imread(fixed_path)
    target_image = io.imread(moving_path)

    ref_image = invert(ref_image)

    target_image = invert(target_image)


    # Crop foreground from the weird Spatial H&E stain
    target_image, bool_rotate = isolateForeground(target_image, bbox_expansion= 500)

    if bool_rotate:
        target_image = rotate(target_image, -90, resize=True)

    # Downscale
    target_image = equalizeImageSize(ref_image, target_image)

    # Equalize ref image dtype, cause target image comes out as uint
    ref_image = img_as_uint(ref_image)

    fixed_image = sitk.GetImageFromArray(ref_image)

    moving_image = sitk.GetImageFromArray(target_image)

    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
    
    initial_transform = sitk.CenteredTransformInitializer(
    fixed_image,
    moving_image,
    sitk.AffineTransform(2),
    sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    moving_resampled = sitk.Resample(
        moving_image,
        fixed_image,
        initial_transform,
        sitk.sitkLinear,
        0.0,
        moving_image.GetPixelID(),
    )
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.10)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=50000,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform)

    intermediate_transform = registration_method.Execute(
        sitk.Cast(fixed_image, sitk.sitkFloat32), sitk.Cast(moving_image, sitk.sitkFloat32)
    )
    print(f"Affine Final metric value: {registration_method.GetMetricValue()}")
    print(f"Affine Optimizer's stopping condition, {registration_method.GetOptimizerStopConditionDescription()}")


    bspline_registration_method = sitk.ImageRegistrationMethod()

    transformDomainMeshSize = [8] * moving_resampled.GetDimension()
    tx = sitk.BSplineTransformInitializer(fixed_image, transformDomainMeshSize)


    bspline_registration_method.SetMetricAsCorrelation()

    bspline_registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5,
                               numberOfIterations=100,
                               maximumNumberOfCorrections=5,
                               maximumNumberOfFunctionEvaluations=1000,
                               costFunctionConvergenceFactor=1e+7)
    bspline_registration_method.SetInitialTransform(tx, inPlace=False)
    bspline_registration_method.SetInterpolator(sitk.sitkLinear)

    #bspline_registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    #bspline_registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    #bspline_registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    #bspline_registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()


    final_transform = bspline_registration_method.Execute(
        sitk.Cast(fixed_image, sitk.sitkFloat32), sitk.Cast(moving_resampled, sitk.sitkFloat32)
    )
    print(f"Bspline Final metric value: {bspline_registration_method.GetMetricValue()}")
    print(f"Bspline Optimizer's stopping condition, {bspline_registration_method.GetOptimizerStopConditionDescription()}")

    moving_bspline_resampled = sitk.Resample(
        moving_resampled,
        fixed_image,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving_image.GetPixelID(),
    )
    sitk.WriteImage(
        moving_bspline_resampled, os.path.join(f"{out_file_base}_registered.tif")
    )
    sitk.WriteTransform(
        final_transform, os.path.join(f"{out_file_base}_transform.txt")
    )

    return moving_bspline_resampled, final_transform


