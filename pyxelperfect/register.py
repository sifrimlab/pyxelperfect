import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import io
from icecream import ic
from display import evaluateRegistration, createComposite

def translationTranform(fixed, moving):
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()
    R.SetOptimizerAsRegularStepGradientDescent(4.0, .01, 200)
    R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
    R.SetInterpolator(sitk.sitkLinear)
    outTx = R.Execute(fixed, moving)
    return outTx

def bSplineTransform(fixed, moving):
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

def performRegistration(ref_image: str or sitk.Image or np.array, target_image: str or sitk.Image or np.array, method="rigid",out_name: str = "") -> np.array:
    transform_dict = {"translation": translationTranform, "rigid": rigidTransform, "affine": affineTransform, "bspline": bSplineTransform}

    transformFunc = transform_dict[method]

    if isinstance(ref_image, str):
        ref = sitk.ReadImage(ref_image, sitk.sitkFloat32)
    elif isinstance(ref_image, np.array):
        ref = sitk.GetImageFromArray(ref_image)
    else:
        ref = ref_image

    if isinstance(target_image, str):
        target = sitk.ReadImage(target_image, sitk.sitkFloat32)
    elif isinstance(target_image, np.array):
        target = sitk.GetImageFromArray(target_image)
    else:
        target = target_image


    transform = transformFunc(ref, target)

    reformed_target = warpImage(target, transform)

    # if theres no out name but also no path of the target, we can't make an outname, so ti will not be saved
    if not isinstance(target_image, str) and not out_name:
        print("No out_name was given but target image is an image, so no output filepath can be generated. Registered image is not written to file.")
        reformed_target = sitk.GetArrayFromImage(reformed_target)
        return reformed_target

    # if the target was a string, but no outname was given, we have to derive the out_name first
    elif not out_name:
        out_name = os.path.splitext(target_image)[0]+ f"_{method}registered.tif"

    # finally we write the image and return it
    sitk.WriteImage(reformed_target, out_name)

    reformed_target = sitk.GetArrayFromImage(reformed_target)
    return reformed_target

def affineTransform(fixed, moving):
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()
    R.SetOptimizerAsRegularStepGradientDescent(4.0, .01, 200)
    R.SetInitialTransform(sitk.AffineTransform(fixed.GetDimension()))
    R.SetInterpolator(sitk.sitkLinear)
    outTx = R.Execute(fixed, moving)
    return outTx

def rigidTransform(fixed, moving):
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()
    R.SetOptimizerAsRegularStepGradientDescent(4.0, .01, 200)
    R.SetInitialTransform(sitk.Euler2DTransform())
    R.SetInterpolator(sitk.sitkLinear)
    outTx = R.Execute(fixed, moving)
    return outTx

if __name__ == '__main__':
    original_image ="/home/david/Documents/prostate_cancer/PWB929_normal_HE_minus_cmc_10X_grey.tif"
    target_image ="/home/david/Documents/prostate_cancer/PWB929_DLC1_grey_cropped_resized.tif"
    # registered_image =  performRegistration("/home/david/Documents/prostate_cancer/PWB929_normal_HE_minus_cmc_10X_grey.tif","/home/david/Documents/prostate_cancer/PWB929_normal_HE_minus_cmc_10X_grey_translated.tif", method="bspline")
    registered_image =  performRegistration(original_image,target_image, method="bspline")
    # io.imsave("original_sit.tif", createComposite(original_image, target_image))
    # io.imsave("registered_sit.png", createComposite(original_image,  "/home/david/Documents/prostate_cancer/PWB929_normal_HE_minus_cmc_10X_grey_translated_registered.tif"))
    # evaluateRegistration(original_image, target_image, "/home/david/Documents/prostate_cancer/PWB929_normal_HE_minus_cmc_10X_grey_translated_registered.tif", plot=False, identifier="bspline")


