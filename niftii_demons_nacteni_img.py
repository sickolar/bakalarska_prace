import csv
import os
import numpy as np
import SimpleITK as sitk
import Nacteni_image
import time
#Registrace vsech casovych okamziku, ulozeni do .nii.gz jako 4D - bspline
def registration_full_demons(read_path, save_path):
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
    fixed_float = sitk.Cast(fixed_image, sitk.sitkFloat32)  # prevod z int na float kvuli vypoctum.

    registrovane.append(fixed_image)

    fixed_maska_array = (sitk.GetArrayFromImage(fixed_image) > 25).astype(np.uint8)
    height = fixed_maska_array.shape[1]  # nacteni vysky snimku
    substracting = int(height * 0.55)  # Oříznutí spodních 45% snímku
    fixed_maska_array[:, substracting:, :] = 0
    fixed_maska_image = sitk.GetImageFromArray(fixed_maska_array)
    fixed_maska_image.CopyInformation(fixed_image)
    fixed_float_masked = sitk.Mask(fixed_float, fixed_maska_image)

    #Inicializace 4D metrik
    evaluate_mi = sitk.ImageRegistrationMethod()
    evaluate_mi.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    evaluate_mi.SetMetricFixedMask(fixed_maska_image)

    evaluate_mse = sitk.ImageRegistrationMethod()
    evaluate_mse.SetMetricAsMeanSquares()
    evaluate_mse.SetMetricFixedMask(fixed_maska_image)

    #Vypocet pro nulty cas (fixed image)
    fixed_float = sitk.Cast(fixed_image, sitk.sitkFloat32)
    mi_scores.append(evaluate_mi.MetricEvaluate(fixed_float,fixed_float))
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


        #Demons
        moving_float = sitk.Cast(moving_resampled, sitk.sitkFloat32)  # prevod z int na float kvuli vypoctum.

        #maska
        moving_maska_array = (sitk.GetArrayFromImage(moving_resampled) > 25).astype(np.uint8)


        moving_maska_array[:, substracting:, :] = 0

        moving_maska_image = sitk.GetImageFromArray(moving_maska_array)

        moving_maska_image.CopyInformation(moving_resampled)

        moving_float_masked = sitk.Mask(moving_float, moving_maska_image)

        # Snaha o matching rozložení barev mezi moving a fixed
        matcher = sitk.HistogramMatchingImageFilter()
        matcher.SetNumberOfHistogramLevels(1024)  # Rozdeleni na 1024 odstinu
        matcher.SetNumberOfMatchPoints(7)  # pocet bodu křivky, ktere se musi střetnout
        matcher.ThresholdAtMeanIntensityOn()  # Ignoruje černé pozadí - bere průměrný jas

        moving_matched = matcher.Execute(moving_float_masked, fixed_float_masked)

        demons = sitk.DiffeomorphicDemonsRegistrationFilter()  # Ošetření toho aby se nepřeložila tkáň přes sebe

        demons.SetNumberOfIterations(680)  # Počet iterací
        demons.SetStandardDeviations(1.0002878393478)  # Vyhlazuje deformační pole
        demons.SetUpdateFieldStandardDeviations(1.10165553633308)  # Aby nešly sousedním pixely opačným směrem

        demons_field = demons.Execute(fixed_float_masked, moving_matched)  # Výpočet vektorového pole pohybu
        final_demons = sitk.DisplacementFieldTransform(demons_field)  # Převedení na objekt final demons
        demons_resampled = sitk.Resample(moving_resampled, fixed_image, final_demons, sitk.sitkLinear, 0.0,
                                         moving_image.GetPixelID())  # Aplikace transformace
        #Pridani do seznamu vsech arrays
        registrovane.append(demons_resampled)
        print(f'Moving cislo {i} byl ulozen')

        #Vypocet metrik
        resampled_float = sitk.Cast(demons_resampled, sitk.sitkFloat32)

        mi_i = evaluate_mi.MetricEvaluate(fixed_float, resampled_float)
        mse_i =evaluate_mse.MetricEvaluate(fixed_float, resampled_float)

        mi_scores.append(mi_i)
        mse_scores.append(mse_i)
        time_stamps.append(time_i)
    print('Spojovani do 4D')
    img_all = sitk.JoinSeries(registrovane)

    #Ulozeni do niftii
    os.makedirs(save_path, exist_ok=True)
    save_path_niftii = os.path.join(save_path, 'registered_4D_demons_59716.nii.gz') #nazev souboru, potencionalne jako vstupni argument
    sitk.WriteImage(img_all, save_path_niftii)

    #Ulozeni metrik co csv
    filename = os.path.join(save_path, 'vysledky_metrik_demons_59716.csv')
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
    registration_full_demons(dcm_folder, save_floder)
    minutes, seconds = divmod(time.time() - start, 60)
    hours, minutes = divmod(minutes, 60)

    result = f'Cas vypoctu: {int(hours)} hodin, {int(minutes)}, minut a {int(seconds)} sekund'
    print(result)

    with open('cas_niftii_demons_93044.txt', 'w', encoding='utf-8') as f:
        f.write(result)
