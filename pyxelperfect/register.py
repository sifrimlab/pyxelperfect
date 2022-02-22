import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import io
from icecream import ic

def calculateTransform(fixed, moving):
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()
    R.SetOptimizerAsRegularStepGradientDescent(4.0, .01, 200)
    R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
    R.SetInterpolator(sitk.sitkLinear)
    outTx = R.Execute(fixed, moving)
    return outTx

def calculateBsplineTransform(fixed, moving):
    transformDomainMeshSize = [8] * moving.GetDimension()
    tx = sitk.BSplineTransformInitializer(fixed,
                                          transformDomainMeshSize)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()

    R.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5,
                           numberOfIterations=100,
                           maximumNumberOfCorrections=5,
                           maximumNumberOfFunctionEvaluations=1000,
                           costFunctionConvergenceFactor=1e+7)
    R.SetInitialTransform(tx, True)
    R.SetInterpolator(sitk.sitkLinear)

    outTx = R.Execute(fixed, moving)

    return outTx

def warpImage(image, transform: str or sitk.Transform) -> sitk.Image :
    if isinstance(transform, str):
        transform = sitk.ReadTransform(transform)
    resampled = sitk.Resample(image, transform, sitk.sitkLinear, 0.0, sitk.sitkUInt16)
    return resampled

def rigidRegister(ref_image: str or sitk.Image, target_image: str or sitk.Image, method="rigid",out_name: str = "") -> np.array:

    if isinstance(ref_image, str):
        ref = sitk.ReadImage(ref_image, sitk.sitkFloat32)
    else:
        ref = ref_image
    if isinstance(target_image, str):
        target = sitk.ReadImage(target_image, sitk.sitkFloat32)
    else:
        target = target_image

    transform = affineRegister(ref, target)

    reformed_target = warpImage(target, transform)

    # if theres no out name but also no path of the target, we can't make an outname, so ti will not be saved
    if not isinstance(target_image, str) and not out_name:
        print("No out_name was given but target image is an image, so no output filepath can be generated. Registered image is not written to file.")
        reformed_target = sitk.GetArrayFromImage(reformed_target)
        return reformed_target
    # if the target was a string, but no outname was given, we have to derive the out_name first
    elif not out_name:
        out_name = os.path.splitext(target_image)[0]+ "_registered.tif"

    # finally we write the image and return it
    sitk.WriteImage(reformed_target, out_name)

    reformed_target = sitk.GetArrayFromImage(reformed_target)
    return reformed_target

def affineRegister(fixed, moving):
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()
    R.SetOptimizerAsRegularStepGradientDescent(4.0, .01, 200)
    R.SetInitialTransform(sitk.AffineTransform(fixed.GetDimension()))
    R.SetInterpolator(sitk.sitkLinear)
    outTx = R.Execute(fixed, moving)
    return outTx

if __name__ == '__main__':
    registered_image = rigidRegister("/media/sdc1/prostate_cancer/PWB929_cancer_H&E_+cmc_padded_big.tif", "/media/sdc1/prostate_cancer/PWB_929_DLC2_High_Res_horizontal.tif",out_name="/media/sdc1/prostate_cancer/PWB929_cancer_H&E_+cmc_padded_big_bspline_registered.tif")
