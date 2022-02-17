from pathlib import Path
import SimpleITK as sitk
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

def readIntoSITK(file_path: str) -> sitk.Image:
    image = sitk.ReadImage(file_path, sitk.sitkFloat32)
    return image

def writeOutOfSITK(image: sitk.Image, out_path: str):
    sitk.WriteImage(image, out_path)

def rigidRegister(ref_image_path: str, target_image_path: str):
    ref = readIntoSITK(ref_image_path)
    target = readIntoSITK(target_image_path)


    transform = calculateTransform(ref, target)

    reformed_target = warpImage(target, transform)

    out_name = Path(target_image_path).basename + "_registered.tiff"
    writeOutOfSITK(reformed_target, )



    



if __name__ == '__main__':
    image1 =sitk.ReadImage("/home/david/Documents/communISS/data/merfish/merfish_4.tif", sitk.sitkFloat32)
    image2 =sitk.ReadImage("/home/david/Documents/communISS/data/merfish/merfish_5.tif", sitk.sitkFloat32)

    rigidRegister(image1, image2)

