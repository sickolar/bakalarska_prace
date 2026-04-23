import os
import SimpleITK as sitk
import Nacteni_image
import numpy as np
import csv
import time
#Registrace vsech casovych okamziku, ulozeni do .nii.gz jako 4D - rigid
def registration_full_rigid(read_path, save_path):
    time_folders = [] #Zde se ulozi slozky s rezy v danem case
    registrovane = []
    mi_scores = []
    mse_scores = []
    time_stamps = []
    for folder in sorted(os.listdir(read_path)):
        if os.path.isdir(os.path.join(read_path, folder)):
            cesta = os.path.join(read_path, folder)
            time_folders.append(cesta)

    #Nacteni fixed
    fixed_image = Nacteni_image.data_load(time_folders[0])
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)

    registrovane.append(fixed_image)

    fixed_maska_array = (sitk.GetArrayFromImage(fixed_image) > 25).astype(np.uint8)
    height = fixed_maska_array.shape[1]
    substracting = int(height * 0.55)
    fixed_maska_array[:, substracting:, :] = 0
    fixed_maska_image = sitk.GetImageFromArray(fixed_maska_array)
    fixed_maska_image.CopyInformation(fixed_image)

    evaluate_mi = sitk.ImageRegistrationMethod()
    evaluate_mi.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    evaluate_mi.SetMetricFixedMask(fixed_maska_image)

    evaluate_mse = sitk.ImageRegistrationMethod()
    evaluate_mse.SetMetricAsMeanSquares()
    evaluate_mse.SetMetricFixedMask(fixed_maska_image)

    #Nulty cas / fixed image
    fixed_float = sitk.Cast(fixed_image, sitk.sitkFloat32)
    mi_scores.append(evaluate_mi.MetricEvaluate(fixed_float, fixed_float))
    mse_scores.append(evaluate_mse.MetricEvaluate(fixed_float, fixed_float))
    time_stamps.append('T_00')
    #Prochazeni jednotlivych casovych slozek krome 1. (ta je nas fixed)
    for i in range(1, len(time_folders)):
        time_i = f'T_{i:02d}'

        moving_image = Nacteni_image.data_load(os.path.join(read_path, time_folders[i]))
        moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

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


        #Pridani do seznamu jako ITK obrazy
        registrovane.append(moving_resampled)
        print(f'Moving cislo {i} byl ulozen')

        #Vypocet metrik
        resampled_float = sitk.Cast(moving_resampled, sitk.sitkFloat32)

        mi_i = evaluate_mi.MetricEvaluate(fixed_float, resampled_float)
        mse_i = evaluate_mse.MetricEvaluate(fixed_float, resampled_float)

        mi_scores.append(mi_i)
        mse_scores.append(mse_i)
        time_stamps.append(time_i)

    print('Spojovani do 4D')
    img_all = sitk.JoinSeries(registrovane)

    os.makedirs(save_path, exist_ok=True)
    save_path_niftii = os.path.join(save_path, 'registered_4D_rigid_59716.nii.gz') #nazev souboru, potencionalne jako vstupni argument
    sitk.WriteImage(img_all, save_path_niftii)

    #Ulozeni metrik co csv
    filename = os.path.join(save_path, 'vysledky_metrik_rigid_59716.csv')
    with open(filename, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['Časový okamžik', 'MI_skóre', 'MSE_skóre'])
        for timee, mi, mse in zip(time_stamps, mi_scores, mse_scores):
            #Ulozeni, pozdeji napr v excelu upravit do abs
            writer.writerow([timee, mi, mse])

if __name__ =='__main__':
    print('Zacatek vypoctu')
    start = time.time()
    dcm_folder = r'C:\Users\andre\OneDrive\Plocha\School\BTB_3\Bakalářská práce\subjekty\pokus_1\manifest-data1407430404196\QIN Breast DCE-MRI\QIN-Breast-DCE-MRI-BC05\03-23-1997-NA-Breast CE-93044'
    save_floder = r'C:\Bakalarka\Nifty_files'
    registration_full_rigid(dcm_folder, save_floder)
    minutes, seconds = divmod(time.time() - start, 60)
    hours, minutes = divmod(minutes, 60)

    result = f'Cas vypoctu: {int(hours)} hodin, {int(minutes)}, minut a {int(seconds)} sekund'
    print(result)

    with open('cas_niftii_rigid_93044.txt', 'w', encoding='utf-8') as f:
        f.write(result)



