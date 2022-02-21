import os
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

def rigidRegister(ref_image_path: str, target_image_path: str, out_name: str = ""):
    ref = sitk.ReadImage(ref_image_path, sitk.sitkFloat32)
    target = sitk.ReadImage(target_image_path, sitk.sitkFloat32)


    transform = calculateTransform(ref, target)

    reformed_target = warpImage(target, transform)

    if not out_name:
        out_name = os.path.splitext(target_image_path)[0]+ "_registered.tif"
    sitk.WriteImage(reformed_target, out_name)

    return reformed_target



if __name__ == '__main__':
    path_image1 = "/media/sdc1/prostate_cancer/PWB929_normal_H&E_+cmc.tif"
    path_image2 = "/media/sdc1/prostate_cancer/PWB929_cancer_H&E_+cmc.tif"

    image1 = io.imread(path_image1)
    image2 = io.imread(path_image2)

    # registered_image = rigidRegister(path_image1, path_image2)
    # ic(image1.shape, image2.shape, registered_image.shape)
    
    registered_image = io.imread("/media/sdc1/prostate_cancer/PWB929_cancer_H&E_+cmc_registered.tif")
    fig, axs = plt.subplots(1,3)
    axs[0] = plt.imshow(image1)
    axs[1] = plt.imshow(image2)
    axs[2] = plt.imshow(registered_image)
    plt.show()

