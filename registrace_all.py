import os
import numpy as np
import SimpleITK as sitk
import Nacteni

#Registrace vsech casovych okamziku, ulozeni do .nii.gz jako 4D
def registration_full(read_path, save_path):
    time_folders = [] #Zde se ulozi slozky s rezy v danem case
    registrovane = []
    for folder in os.listdir(read_path):
        if os.path.isdir(os.path.join(read_path, folder)):
            cesta = os.path.join(read_path, folder)
            time_folders.append(cesta)

    #Nacteni fixed
    fixed_array = Nacteni.data_load(time_folders[0])
    fixed_image = sitk.GetImageFromArray(fixed_array.astype(np.float32))

    #Maska fixed
    fixed_maska_array = (fixed_array > 25).astype(np.uint8)
    fixed_maska_image = sitk.GetImageFromArray(fixed_maska_array)

    registrovane.append(fixed_array)

    #Prochazeni jednotlivych casovych slozek krome 1. (ta je nas fixed)
    for i in range(1, len(time_folders)):
        moving_array = Nacteni.data_load(os.path.join(read_path, time_folders[i]))
        moving_image = sitk.GetImageFromArray(moving_array.astype(np.float32))

        #Nejdrive rigidni transformace
        initial_transformation = sitk.CenteredTransformInitializer(
            # Pocateni zarovnání, střed objemu je na sobe, vycentrovan
            fixed_image,  # referenční obraz
            moving_image,  # zarovnávaný obraz
            sitk.Euler3DTransform(),  # rigidni transformace
            sitk.CenteredTransformInitializerFilter.GEOMETRY  # určení středu obrazů
        )

        # vytvoření objektu registrace, následně to budu nastovovat
        registration_method = sitk.ImageRegistrationMethod()
        #Nastavovani
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)  # Rozdělí hodnoty intenzity do 50 histogramových binů, metrika podobnosti
        registration_method.SetInterpolator(sitk.sitkLinear)  # lineární dopočet hodnot mezi voxely
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=200, estimateLearningRate=sitk.ImageRegistrationMethod.Never)  # Mění parametry transformace, aby se metrika zlepšovala
        registration_method.SetOptimizerScalesFromPhysicalShift()  # Normalizace, nastavení vah parametru transfromace na základě vlivu na obraz, mm ne voxely u descentu
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])  # Pyramidová registrace, od hrube po jemnou
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])  # Jak moc se na jednotlivých úrovních pyramidy obraz rozmaže
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()  # Určuje že jednotky jsou v mm ne pixelech, u pyramidy
        registration_method.SetInitialTransform(initial_transformation, inPlace=False)  # Zvolím začátek registrace - initial_transformation
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM) #Náhodně vybere voxely z obrazu, urychlení
        registration_method.SetMetricSamplingPercentage(0.05) #Vybere 5% voxelu z obrazu

        final_rigid_transformation = registration_method.Execute(fixed_image, moving_image)

        #Vysledny moving zregistrovany obraz po rigidni registraci
        moving_resampled = sitk.Resample(moving_image, fixed_image, final_rigid_transformation, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

        #Maska moving
        moving_resampled_array = sitk.GetArrayFromImage(moving_resampled)
        moving_maska_array = (moving_resampled_array > 25).astype(np.uint8)
        moving_maska_image = sitk.GetImageFromArray(moving_maska_array)

        #Bspline
        mesh_size = [4, 4, 4]
        bspline_transformation = sitk.BSplineTransformInitializer(fixed_image, mesh_size)
        bspline_reg = sitk.ImageRegistrationMethod()
        bspline_reg.SetMetricAsJointHistogramMutualInformation(numberOfHistogramBins=20)
        bspline_reg.SetInterpolator(sitk.sitkBSpline)
        bspline_reg.SetShrinkFactorsPerLevel(shrinkFactors=[8, 4, 2, 1])
        bspline_reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[3, 2, 1, 0])
        bspline_reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        bspline_reg.SetInitialTransform(bspline_transformation, inPlace=False)
        bspline_reg.SetOptimizerAsLBFGS2(numberOfIterations=100, solutionAccuracy=1e-2, deltaConvergenceTolerance=0.01)
        bspline_reg.SetMetricSamplingStrategy(registration_method.RANDOM) #Náhodně vybere voxely z obrazu, urychlení
        bspline_reg.SetMetricSamplingPercentage(0.05) #Vybere 5% voxelu z obrazu
        bspline_reg.SetMetricFixedMask(fixed_maska_image)
        bspline_reg.SetMetricMovingMask(moving_maska_image)
        bspline_final_transformation = bspline_reg.Execute(fixed_image, moving_resampled)
        #moving po bspline registaci
        moving_bspline_resampled = sitk.Resample(moving_resampled, fixed_image, bspline_final_transformation,
                                                 sitk.sitkBSpline, 0.0, moving_resampled.GetPixelID())

        #Pridani do seznamu vsech arrays
        resampled_array = sitk.GetArrayFromImage(moving_bspline_resampled)
        registrovane.append(resampled_array)

        print(f'Moving cislo {i} byl ulozen')

    array_3D = []

    #Prevod arrays, aby se mohl pouzit joinseries
    for arrrr in registrovane:
        img_3d = sitk.GetImageFromArray(arrrr.astype(np.float32))
        array_3D.append(img_3d)

    print('Spojovani do 4D')
    img_all = sitk.JoinSeries(array_3D)

    os.makedirs(save_path, exist_ok=True)
    save_path_niftii = os.path.join(save_path, 'registered_4D_2.nii.gz') #nazev souboru, potencionalne jako vstupni argument
    sitk.WriteImage(img_all, save_path_niftii)


