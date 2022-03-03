import os
import numpy as np
import SimpleITK as sitk
from skimage import io

def translationTranform(fixed, moving):
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()
    R.SetOptimizerAsRegularStepGradientDescent(4.0, .01, 200)
    R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
    R.SetInterpolator(sitk.sitkLinear)
    outTx = R.Execute(fixed, moving)
    return outTx

def rigidTransform(fixed, moving):
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()
    R.SetOptimizerAsRegularStepGradientDescent(4.0, .01, 200)
    initial_transform = sitk.CenteredTransformInitializer(
                        fixed,
                        moving,
                        sitk.Euler2DTransform(),
                        sitk.CenteredTransformInitializerFilter.GEOMETRY,
                    )
    R.SetInitialTransform(initial_transform)
    R.SetInterpolator(sitk.sitkLinear)
    outTx = R.Execute(fixed, moving)
    return outTx

def scaleTransform(fixed, moving):
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()
    R.SetOptimizerAsRegularStepGradientDescent(4.0, .01, 200)
    initial_transform = sitk.CenteredTransformInitializer(
                        fixed,
                        moving,
                        sitk.ScaleTransform(fixed.GetDimension()),
                        sitk.CenteredTransformInitializerFilter.GEOMETRY,
                    )
    R.SetInitialTransform(initial_transform)
    R.SetInterpolator(sitk.sitkLinear)
    outTx = R.Execute(fixed, moving)
    return outTx

def multScaleTransform(fixed, moving, initial_transform):
    # This is the registration configuration which we use in all cases. The only parameter that we vary
# is the initial_transform.
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.01)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        estimateLearningRate=R.Once,
    )
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetInitialTransform(initial_transform)
    R.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    R.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    outTx = R.Execute(fixed, moving)
    return outTx


def affineTransform(fixed, moving):
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()
    R.SetOptimizerAsRegularStepGradientDescent(4.0, .01, 200)
    initial_transform = sitk.CenteredTransformInitializer(
                        fixed,
                        moving,
                        sitk.AffineTransform(fixed.GetDimension()),
                        sitk.CenteredTransformInitializerFilter.GEOMETRY,
                    )
    R.SetInitialTransform(initial_transform)
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

def performRegistration(ref_image: str or sitk.Image or np.ndarray, target_image: str or sitk.Image or np.ndarray, method="rigid",out_name: str = "") -> np.array:
    transform_dict = {"translation": translationTranform, "rigid": rigidTransform, "affine": affineTransform, "bspline": bSplineTransform, "scale": scaleTransform}

    transformFunc = transform_dict[method]

    if isinstance(ref_image, str):
        ref = sitk.ReadImage(ref_image, sitk.sitkFloat32)
    elif isinstance(ref_image, np.ndarray):
        ref = sitk.GetImageFromArray(ref_image)
        ref = sitk.Cast(ref, sitk.sitkFloat32)

    else:
        ref = ref_image

    if isinstance(target_image, str):
        target = sitk.ReadImage(target_image, sitk.sitkFloat32)
    elif isinstance(target_image, np.ndarray):
        target = sitk.GetImageFromArray(target_image)
        target = sitk.Cast(target, sitk.sitkFloat32)
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
    return reformed_target, out_name
