import numpy as np
import SimpleITK as sitk
import Nacteni
import napari

#Nacteni pomoci Nacteni.py
data_base = Nacteni.data_load(r'C:\Users\andre\OneDrive\Plocha\School\BTB_3\Bakalářská práce\subjekty\pokus_1\manifest-data1407430404196\QIN Breast DCE-MRI\QIN-Breast-DCE-MRI-BC01\05-15-1996-NA-Breast CE-49782\07.000000-twist20sdynTRA h20 ex B17TT49.6s-18302')
data_chang = Nacteni.data_load(r'C:\Users\andre\OneDrive\Plocha\School\BTB_3\Bakalářská práce\subjekty\pokus_1\manifest-data1407430404196\QIN Breast DCE-MRI\QIN-Breast-DCE-MRI-BC01\05-15-1996-NA-Breast CE-49782\49.000000-twist20sdynTRA h20 ex B17TT473.0s-83569')

fixed_image = sitk.GetImageFromArray(data_base.astype(np.float32)) #zisk 3D obrazu z numpy array
moving_image = sitk.GetImageFromArray(data_chang.astype(np.float32))

initial_transformation = sitk.CenteredTransformInitializer( #Pocateni zarovnání, střed objemu je na sobe, vycentrovan
    fixed_image, #referenční obraz
    moving_image, #zarovnávaný obraz
    sitk.Euler3DTransform(), #rigidni transformace
    sitk.CenteredTransformInitializerFilter.GEOMETRY #určení středu obrazů
)

registration_method = sitk.ImageRegistrationMethod() #vytvoření objektu registrace, následně to budu nastovovat

registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)#Rozdělí hodnoty intenzity do 50 histogramových binů, metrika podobnosti
#registration_method.SetMetricSamplingStrategy(registration_method.RANDOM) #Náhodně vybere voxely z obrazu, urychlení
#registration_method.SetMetricSamplingPercentage(0.05) #Vybere 1% voxelu z obrazu

registration_method.SetInterpolator(sitk.sitkLinear) #lineární dopočet hodnot mezi voxely

registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=200, estimateLearningRate=sitk.ImageRegistrationMethod.Never) #Mění parametry transformace, aby se metrika zlepšovala
registration_method.SetOptimizerScalesFromPhysicalShift() #Normalizace, nastavení vah parametru transfromace na základě vlivu na obraz

registration_method.SetShrinkFactorsPerLevel(shrinkFactors= [4,2,1]) #Pyramidová registrace, od hrube po jemnou
registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0]) #Jak moc se na jednotlivých úrovních pyramidy obray rozmaže
registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn() #Určuje že jednotky jsou v mm ne pixelech

#optimized_transform = sitk.Euler3DTransform()
#registration_method.SetMovingInitialTransform(initial_transformation)
registration_method.SetInitialTransform(initial_transformation, inPlace=False) #Zvolím začátek registrace - initial_transformation


initial_metric = registration_method.MetricEvaluate(fixed_image, moving_image) #Získá hodnotu podobnosti
print('Prvotni hodnota:', initial_metric)
#final_transform = sitk.CompositeTransform([registration_method.Execute(fixed_image, moving_image), initial_transformation]) #Spuštění registrace
final_transform = registration_method.Execute(fixed_image, moving_image)
print('Final metric value: {0}'.format(registration_method.GetMetricValue()))


moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

fixed_np = sitk.GetArrayFromImage(fixed_image)
moving_np_before = sitk.GetArrayFromImage(moving_image)
moving_np_after = sitk.GetArrayFromImage(moving_resampled)

viewer = napari.Viewer()
viewer.add_image(fixed_np, name='fixed', colormap='gray', contrast_limits=[np.min(fixed_np), np.max(fixed_np)])
viewer.add_image(moving_np_before, name='moving_before', colormap='red', opacity=0.4, blending='additive',
                 contrast_limits=[np.min(fixed_np), np.max(fixed_np)])
viewer.add_image(moving_np_after, name='moving_after', colormap='green', opacity=0.4, blending='additive',
                 contrast_limits=[np.min(fixed_np), np.max(fixed_np)])
napari.run()