import SimpleITK as sitk
import napari
import numpy as np
import Nacteni

#Nacteni pomoci Nacteni.py
data_base = Nacteni.data_load(r'C:\Users\andre\OneDrive\Plocha\School\BTB_3\Bakalářská práce\subjekty\pokus_1\manifest-data1407430404196\QIN Breast DCE-MRI\QIN-Breast-DCE-MRI-BC01\05-15-1996-NA-Breast CE-49782\07.000000-twist20sdynTRA h20 ex B17TT49.6s-18302')
data_chang = Nacteni.data_load(r'C:\Users\andre\OneDrive\Plocha\School\BTB_3\Bakalářská práce\subjekty\pokus_1\manifest-data1407430404196\QIN Breast DCE-MRI\QIN-Breast-DCE-MRI-BC01\05-15-1996-NA-Breast CE-49782\35.000000-twist20sdynTRA h20 ex B17TT331.8s-61791')
fixed_image = sitk.GetImageFromArray(data_base.astype(np.float32)) #zisk 3D obrazu z numpy array
moving_image = sitk.GetImageFromArray(data_chang.astype(np.float32))

#Rigidni registrace
initial_transformation = sitk.CenteredTransformInitializer( #Pocateni zarovnání, střed objemu je na sobe, vycentrovan
    fixed_image, #referenční obraz
    moving_image, #zarovnávaný obraz
    sitk.Euler3DTransform(), #rigidni transformace
    sitk.CenteredTransformInitializerFilter.GEOMETRY #určení středu obrazů
)

registration_method = sitk.ImageRegistrationMethod() #vytvoření objektu registrace, následně to budu nastovovat
registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)#Rozdělí hodnoty intenzity do 50 histogramových binů, metrika podobnosti
registration_method.SetInterpolator(sitk.sitkLinear) #lineární dopočet hodnot mezi voxely
registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=200, estimateLearningRate=sitk.ImageRegistrationMethod.Never) #Mění parametry transformace, aby se metrika zlepšovala
registration_method.SetOptimizerScalesFromPhysicalShift() #Normalizace, nastavení vah parametru transfromace na základě vlivu na obraz, mm ne voxely u descentu
registration_method.SetShrinkFactorsPerLevel(shrinkFactors= [4,2,1]) #Pyramidová registrace, od hrube po jemnou
registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0]) #Jak moc se na jednotlivých úrovních pyramidy obraz rozmaže
registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn() #Určuje že jednotky jsou v mm ne pixelech, u pyramidy
registration_method.SetInitialTransform(initial_transformation, inPlace=False) #Zvolím začátek registrace - initial_transformation

#Metrika
initial_metric = registration_method.MetricEvaluate(fixed_image, moving_image) #Získá hodnotu podobnosti
print('Prvotni hodnota:', initial_metric)
final_transform = registration_method.Execute(fixed_image, moving_image)
print('Final metric value: {0}'.format(registration_method.GetMetricValue()))

moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
moving_array = sitk.GetArrayFromImage(moving_resampled)
#Zisk masky
fixed_maska = (data_base > 25).astype(np.uint8) #větší jak 25 = 1 - ať se nepočíta z pozadí
moving_maska = (moving_array > 25).astype(np.uint8)

fixed_maska_image = sitk.GetImageFromArray(fixed_maska)

moving_maska_image = sitk.GetImageFromArray(moving_maska)

#Bspline registrace non-rigid
print('Bspline')
mesh_size = [4, 4, 4]
bspline_transformation = sitk.BSplineTransformInitializer(fixed_image, mesh_size)
bspline_reg = sitk.ImageRegistrationMethod()
bspline_reg.SetMetricAsJointHistogramMutualInformation(numberOfHistogramBins=20)
bspline_reg.SetMetricSamplingStrategy(bspline_reg.RANDOM) #Vybere 10% - kvůli náročnosti
bspline_reg.SetMetricSamplingPercentage(0.1)
bspline_reg.SetInterpolator(sitk.sitkLinear)
bspline_reg.SetShrinkFactorsPerLevel(shrinkFactors=[8,4,2,1])
bspline_reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[3,2, 1, 0])
bspline_reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
bspline_reg.SetInitialTransform(bspline_transformation, inPlace=False)
bspline_reg.SetOptimizerAsLBFGS2(numberOfIterations=100, solutionAccuracy=1e-2, deltaConvergenceTolerance=0.01)
bspline_reg.SetMetricMovingMask(moving_maska_image)
bspline_reg.SetMetricFixedMask(fixed_maska_image)
bspline_final_transformation = bspline_reg.Execute(fixed_image, moving_resampled)
print('Bspline metric:', bspline_reg.GetMetricValue())

moving_bspline_resampled = sitk.Resample(moving_resampled, fixed_image, bspline_final_transformation, sitk.sitkBSpline, 0.0, moving_resampled.GetPixelID())

#prevod na numpy, kvůli zobrazení
fixed = sitk.GetArrayFromImage(fixed_image)
before = sitk.GetArrayFromImage(moving_image)
rigid = sitk.GetArrayFromImage(moving_resampled)
bspline = sitk.GetArrayFromImage(moving_bspline_resampled)


viewer = napari.Viewer()
viewer.add_image(fixed, name='Fixed', colormap='gray', blending='additive')
viewer.add_image(before, name='Before', colormap='gray', blending='additive')
viewer.add_image(rigid, name='Rigid', colormap='green', blending='additive', opacity=0.5)
viewer.add_image(bspline, name='bspline', colormap='blue', blending='additive', opacity=0.5)
viewer.add_image(bspline, name='bspline, sede', colormap='gray', blending='additive')
viewer.add_labels(fixed_maska, name='Maska fixed', opacity=0.5)
viewer.add_labels(moving_maska, name='Maska moving', opacity=0.5)
napari.run()