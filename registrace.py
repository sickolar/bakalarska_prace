import os
import SimpleITK as sitk
import Nacteni_image
import numpy as np
import csv

def registration_full_rigid(read_path, save_path, pacient_id):
    """
    Provede rigidní registraci 4D DCE-MRI obrazů (všech časových okamžiků vůči prvnímu).

    Funkce načte sérii DICOM snímků a iterativně zarovná všechny pohyblivé (moving) obrazy
    vůči referenčnímu (fixed) obrazu v čase T_00 pomocí metody Euler3DTransform. Toto
    počáteční rigidní zarovnání probíhá na celých snímcích (bez použití masky). Dále je v kódu
    vytvořena maska pro eliminaci hrudního koše a srdce, která se však aplikuje výhradně
    až při průběžném výpočtu hodnot Mutual Information (MI) a Mean Squares (MSE), aby
    vysoce kontrastní oblasti mimo prs nezkreslovaly výsledné hodnocení kvality registrace.

    Parametry:
    ----------
    read_path : str
        Cesta ke složce obsahující DICOM data pacienta (bere v potaz abecední pořadí)
    save_path : str
        Cesta ke složce, kam se uloží výsledný NIfTI objem a CSV tabulka s metrikami.
    pacient_id : str nebo int
        Identifikátor pacienta, který se přidá na konec názvů výstupních souborů
        (např. '10' nebo 'Pacient_3').

    Výstupy:
    --------
    Funkce nevrací žádné proměnné, ale ukládá na disk:
    1. 'registered_4D_rigid_{pacient_id}.nii.gz' - Zaregistrovaný 4D objem obrazů.
    2. 'vysledky_metrik_rigid_{pacient_id}.csv' - Tabulka vývoje metrik MI a MSE v čase.
    """

    time_folders = []
    registrovane = []
    mi_scores = []
    mse_scores = []
    time_stamps = []

    #Načtení cesty, ke všem časovým složkám, je potřeba je abecedně seřadit (01, 02..)
    for folder in sorted(os.listdir(read_path)):
        if os.path.isdir(os.path.join(read_path, folder)):
            cesta = os.path.join(read_path, folder)
            time_folders.append(cesta)

    #Načtení prvního časového okamžiku, ten slouží jako referenční pro všechny ostatní
    fixed_image = Nacteni_image.data_load(time_folders[0])
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)

    registrovane.append(fixed_image)

    #Tvorba masky pomocí prahování. Nastavení minimální intenzity pixelu na >25.
    #Následně je spodních 55% obrazu vynulovaných, snaží se eliminovat srdce z výpočtu.
    fixed_maska_array = (sitk.GetArrayFromImage(fixed_image) > 25).astype(np.uint8)
    height = fixed_maska_array.shape[1]
    substracting = int(height * 0.55)
    fixed_maska_array[:, substracting:, :] = 0

    fixed_maska_image = sitk.GetImageFromArray(fixed_maska_array)
    fixed_maska_image.CopyInformation(fixed_image)

    #Inicializace evaluace pro průběžný výpočet podobnosti (statistika)
    evaluate_mi = sitk.ImageRegistrationMethod()
    evaluate_mi.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    evaluate_mi.SetMetricFixedMask(fixed_maska_image)

    evaluate_mse = sitk.ImageRegistrationMethod()
    evaluate_mse.SetMetricAsMeanSquares()
    evaluate_mse.SetMetricFixedMask(fixed_maska_image)

    #Výpočet statistiky na prvním (fixním) obrazu
    fixed_float = sitk.Cast(fixed_image, sitk.sitkFloat32)
    mi_scores.append(evaluate_mi.MetricEvaluate(fixed_float, fixed_float))
    mse_scores.append(evaluate_mse.MetricEvaluate(fixed_float, fixed_float))
    time_stamps.append('T_00')

    #Iterativní registrace pohyblivých obrazů
    for i in range(1, len(time_folders)):
        time_i = f'T_{i:02d}'

        #Načtení moving obrazu, který je v dané iteraci registrován
        moving_image = Nacteni_image.data_load(os.path.join(read_path, time_folders[i]))
        moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

        #Počáteční zarovnání podle geometrických středů obou obrazů (fixed a daný moving)
        initial_transformation = sitk.CenteredTransformInitializer(
            fixed_image,
            moving_image,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )

        #Inicializace objektu registrace
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetInitialTransform(initial_transformation, inPlace=False)


        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=200, estimateLearningRate=sitk.ImageRegistrationMethod.Never)
        registration_method.SetOptimizerScalesFromPhysicalShift()

        #Nastevní pyramidového přístupu, pro zrychlení registrace (od hrubého rozlišení po jemné)
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        #Výpočet podobnostní metriky z náhodných 5% voxelů (pro urychlení registrace)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.05)

        #Výpočet a aplikace transformace
        final_rigid_transformation = registration_method.Execute(fixed_image, moving_image)
        moving_resampled = sitk.Resample(moving_image, fixed_image, final_rigid_transformation, sitk.sitkLinear, 0.0, moving_image.GetPixelID())


        #Pridani do seznamu jako ITK obrazy
        registrovane.append(moving_resampled)
        print(f'Moving cislo {i} byl ulozen')

        #Vypocet metrik (statistika)
        resampled_float = sitk.Cast(moving_resampled, sitk.sitkFloat32)

        mi_i = evaluate_mi.MetricEvaluate(fixed_float, resampled_float)
        mse_i = evaluate_mse.MetricEvaluate(fixed_float, resampled_float)

        mi_scores.append(mi_i)
        mse_scores.append(mse_i)
        time_stamps.append(time_i)

    #Spojení do 4D objemu objektu simpleITK
    print('Spojování do 4D')
    img_all = sitk.JoinSeries(registrovane)

    #Vytvoření složky, pokud neexstiuje.
    os.makedirs(save_path, exist_ok=True)

    #Uložení 4D objemu do formátu NIfFTI
    save_path_niftii = os.path.join(save_path, f'registered_4D_rigid_{pacient_id}.nii.gz')
    sitk.WriteImage(img_all, save_path_niftii)

    #Uložení metrik do .csv
    filename = os.path.join(save_path, f'vysledky_metrik_rigid_{pacient_id}.csv')
    with open(filename, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['Časový okamžik', 'MI_skóre', 'MSE_skóre'])
        for timee, mi, mse in zip(time_stamps, mi_scores, mse_scores):
            writer.writerow([timee, mi, mse])


def registration_full_bspline_mse(read_path, save_path, pacient_id):
    """
    Provede nelineární B-spline registraci (MSE) 4D DCE-MRI obrazů a vypočítá metriky
    podobnosti (MI a MSE).

    Funkce nejprve provede globální rigidní zarovnání celých snímků vůči času T_00.
    Po tomto hrubém zarovnání se aplikuje maskování (vynulování oblasti hrudníku
    a srdce) a spustí senelineární B-spline registrace zaměřená na tkáň prsu.
    Jako metrika podobnosti pro B-spline je použita střední kvadratická chyba (MSE).
    Pro každý zregistrovaný snímek průběžně počítá hodnoty Mutual Information (MI)
    a Mean Squares (MSE).

    Parametry:
    ----------
    read_path : str
        Cesta ke složce obsahující DICOM data pacienta (bere v potaz abecední pořadí).
    save_path : str
        Cesta ke složce, kam se uloží výsledný NIfTI objem a CSV tabulka s metrikami.
    pacient_id : str nebo int
        Identifikátor pacienta, který se přidá na konec názvů výstupních souborů
        (např. '10' nebo 'Pacient_3').

    Výstupy:
    --------
    Funkce nevrací žádné proměnné, ale ukládá na disk:
    1. 'registered_4D_mse_{pacient_id}.nii.gz' - Zaregistrovaný 4D objem obrazů.
    2. 'vysledky_metrik_mse_{pacient_id}.csv' - Tabulka vývoje metrik MI a MSE v čase.
    """

    time_folders = []
    registrovane = []
    mi_scores = []
    mse_scores = []
    time_stamps = []

    #Načtení cesty, ke všem časovým složkám, je potřeba je abecedně seřadit (01, 02..)
    for folder in sorted(os.listdir(read_path)):
        if os.path.isdir(os.path.join(read_path, folder)):
            cesta = os.path.join(read_path, folder)
            time_folders.append(cesta)

    #Načtení prvního časového okamžiku, ten slouží jako referenční pro všechny ostatní
    fixed_image = Nacteni_image.data_load(time_folders[0])
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)

    registrovane.append(fixed_image)

    #Tvorba masky pomocí prahování. Nastavení minimální intenzity pixelu na >25.
    #Následně je spodních 55% obrazu vynulovaných, snaží se eliminovat srdce z výpočtu.
    fixed_maska_array = (sitk.GetArrayFromImage(fixed_image) > 25).astype(np.uint8)
    height = fixed_maska_array.shape[1]
    substracting = int(height * 0.55) #odrezani spodich 45% - tedy hrudnich kosti a srdce
    fixed_maska_array[:, substracting:, :] = 0

    fixed_maska_image = sitk.GetImageFromArray(fixed_maska_array)
    fixed_maska_image.CopyInformation(fixed_image)



    #Inicializace evaluace pro průběžný výpočet podobnosti (statistika)
    evaluate_mi = sitk.ImageRegistrationMethod()
    evaluate_mi.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    evaluate_mi.SetMetricFixedMask(fixed_maska_image)

    evaluate_mse = sitk.ImageRegistrationMethod()
    evaluate_mse.SetMetricAsMeanSquares()
    evaluate_mse.SetMetricFixedMask(fixed_maska_image)

    #Výpočet statistiky na prvním (fixním) obrazu
    fixed_float = sitk.Cast(fixed_image, sitk.sitkFloat32)
    mi_scores.append(evaluate_mi.MetricEvaluate(fixed_float, fixed_float))
    mse_scores.append(evaluate_mse.MetricEvaluate(fixed_float, fixed_float))
    time_stamps.append('T_00')

    #Iterativní registrace pohyblivých obrazů
    for i in range(1, len(time_folders)):
        time_i = f'T_{i:02d}'

        #Načtení moving obrazu, který je v dané iteraci registrován
        moving_image = Nacteni_image.data_load(os.path.join(read_path, time_folders[i]))
        moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
        moving_image.CopyInformation(fixed_image)

        #Počáteční zarovnání podle geometrických středů obou obrazů (fixed a daný moving)
        initial_transformation = sitk.CenteredTransformInitializer(
            fixed_image,
            moving_image,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )

        #Inicializace objektu registrace
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetInitialTransform(initial_transformation, inPlace=False)

        #Nastavení parametrů rigidní registrace
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=200, estimateLearningRate=sitk.ImageRegistrationMethod.Never)
        registration_method.SetOptimizerScalesFromPhysicalShift()

        #Nastevní pyramidového přístupu, pro zrychlení registrace (od hrubého rozlišení po jemné)
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        #Výpočet podobnostní metriky z náhodných 5% voxelů (pro urychlení registrace)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.05)

        #Výpočet a aplikace transformace
        final_rigid_transformation = registration_method.Execute(fixed_image, moving_image)
        moving_resampled = sitk.Resample(moving_image, fixed_image, final_rigid_transformation, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

        #Vytvoření ořezané masky i pro pohyblivý (moving) obraz
        moving_resampled_array = sitk.GetArrayFromImage(moving_resampled)
        moving_maska_array = (moving_resampled_array > 25).astype(np.uint8)
        moving_maska_array[:, substracting:, :] = 0
        moving_maska_image = sitk.GetImageFromArray(moving_maska_array)
        moving_maska_image.CopyInformation(moving_resampled)

        #Inicializace mřížky pro deformaci
        mesh_size = [10, 10, 10]

        #Inicializace objektu B-spline registrace
        bspline_transformation = sitk.BSplineTransformInitializer(fixed_image, mesh_size)
        bspline_reg = sitk.ImageRegistrationMethod()

        #Nastavení podobnostní metriky a interpolace
        bspline_reg.SetMetricAsMeanSquares()
        bspline_reg.SetInterpolator(sitk.sitkBSpline)
        bspline_reg.SetInitialTransform(bspline_transformation, inPlace=False)

        #Nastevní pyramidového přístupu, pro zrychlení registrace (od hrubého rozlišení po jemné)
        bspline_reg.SetShrinkFactorsPerLevel(shrinkFactors=[8, 4, 2, 1])
        bspline_reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[3, 2, 1, 0])
        bspline_reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        #Nastevní optimalizátoru
        bspline_reg.SetOptimizerAsLBFGS2(numberOfIterations=740, solutionAccuracy=8.65013878786101e-07, deltaConvergenceTolerance=4.903164219581583e-07)

        #Výpočet podobnostní metriky z náhodných 5% voxelů (pro urychlení registrace)
        bspline_reg.SetMetricSamplingStrategy(bspline_reg.RANDOM)
        bspline_reg.SetMetricSamplingPercentage(0.05)

        #Aplikace masky nastavené masky, pro výpočet podobnostní metriky, na moving a fixed image
        bspline_reg.SetMetricFixedMask(fixed_maska_image)
        bspline_reg.SetMetricMovingMask(moving_maska_image)

        #Výpočet a aplikace nastavené B-spline transformace
        bspline_final_transformation = bspline_reg.Execute(fixed_image, moving_resampled)
        moving_bspline_resampled = sitk.Resample(moving_resampled, fixed_image, bspline_final_transformation,
                                                 sitk.sitkBSpline, 0.0, moving_resampled.GetPixelID())

        #Pridani do seznamu jako ITK obrazy
        registrovane.append(moving_bspline_resampled)
        print(f'Moving cislo {i} byl ulozen')

        #Vypocet metrik (statistika)
        resampled_float = sitk.Cast(moving_bspline_resampled, sitk.sitkFloat32)

        mi_i = evaluate_mi.MetricEvaluate(fixed_float, resampled_float)
        mse_i = evaluate_mse.MetricEvaluate(fixed_float, resampled_float)

        mi_scores.append(mi_i)
        mse_scores.append(mse_i)
        time_stamps.append(time_i)

    #Spojení do 4D objemu objektu simpleITK
    print('Spojovani do 4D')
    img_all = sitk.JoinSeries(registrovane)

    #Vytvoření složky, pokud neexstiuje.
    os.makedirs(save_path, exist_ok=True)

    #Uložení 4D objemu do formátu NIfFTI
    save_path_niftii = os.path.join(save_path, f'registered_4D_mse_{pacient_id}.nii.gz') #nazev souboru, potencionalne jako vstupni argument
    sitk.WriteImage(img_all, save_path_niftii)

    #Ulozeni metrik co .csv
    filename = os.path.join(save_path, f'vysledky_metrik_mse_{pacient_id}.csv')
    with open(filename, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['Časový okamžik', 'MI_skóre', 'MSE_skóre'])
        for timee, mi, mse in zip(time_stamps, mi_scores, mse_scores):
            writer.writerow([timee, mi, mse])

def registration_full_bspline_mi(read_path, save_path, pacient_id):
    """
    Provede nelineární B-spline registraci (MI) 4D DCE-MRI obrazů a vypočítá metriky
    podobnosti (MI a MSE).

    Funkce nejprve provede globální rigidní zarovnání celých snímků vůči času T_00.
    Po tomto hrubém zarovnání se aplikuje maskování (vynulování oblasti hrudníku
    a srdce) a spustí senelineární B-spline registrace zaměřená na tkáň prsu.
    Jako metrika podobnosti pro B-spline je použita metrika vzájemné informace (MI).
    Pro každý zregistrovaný snímek průběžně počítá hodnoty Mutual Information (MI)
    a Mean Squares (MSE).

    Parametry:
    ----------
    read_path : str
        Cesta ke složce obsahující DICOM data pacienta (bere v potaz abecední pořadí).
    save_path : str
        Cesta ke složce, kam se uloží výsledný NIfTI objem a CSV tabulka s metrikami.
    pacient_id : str nebo int
        Identifikátor pacienta, který se přidá na konec názvů výstupních souborů
        (např. '10' nebo 'Pacient_3').

    Výstupy:
    --------
    Funkce nevrací žádné proměnné, ale ukládá na disk:
    1. 'registered_4D_mi_{pacient_id}.nii.gz' - Zaregistrovaný 4D objem obrazů.
    2. 'vysledky_metrik_mi_{pacient_id}.csv' - Tabulka vývoje metrik MI a MSE v čase.
    """
    time_folders = []
    registrovane = []
    mi_scores = []
    mse_scores = []
    time_stamps = []

    #Načtení cesty, ke všem časovým složkám, je potřeba je abecedně seřadit (01, 02..)
    for folder in sorted(os.listdir(read_path)):
        if os.path.isdir(os.path.join(read_path, folder)):
            cesta = os.path.join(read_path, folder)
            time_folders.append(cesta)

    #Načtení prvního časového okamžiku, ten slouží jako referenční pro všechny ostatní
    fixed_image = Nacteni_image.data_load(time_folders[0])
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)

    registrovane.append(fixed_image)

    #Tvorba masky pomocí prahování. Nastavení minimální intenzity pixelu na >25.
    #Následně je spodních 55% obrazu vynulovaných, snaží se eliminovat srdce z výpočtu.
    fixed_maska_array = (sitk.GetArrayFromImage(fixed_image) > 25).astype(np.uint8)
    height = fixed_maska_array.shape[1]
    substracting = int(height * 0.55)
    fixed_maska_array[:, substracting:, :] = 0

    fixed_maska_image = sitk.GetImageFromArray(fixed_maska_array)
    fixed_maska_image.CopyInformation(fixed_image)




    #Inicializace evaluace pro průběžný výpočet podobnosti (statistika)
    evaluate_mi = sitk.ImageRegistrationMethod()
    evaluate_mi.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    evaluate_mi.SetMetricFixedMask(fixed_maska_image)

    evaluate_mse = sitk.ImageRegistrationMethod()
    evaluate_mse.SetMetricAsMeanSquares()
    evaluate_mse.SetMetricFixedMask(fixed_maska_image)

    #Výpočet statistiky na prvním (fixním) obrazu
    fixed_float = sitk.Cast(fixed_image, sitk.sitkFloat32)
    mi_scores.append(evaluate_mi.MetricEvaluate(fixed_float, fixed_float))
    mse_scores.append(evaluate_mse.MetricEvaluate(fixed_float, fixed_float))
    time_stamps.append('T_00')

    #Iterativní registrace pohyblivých obrazů
    for i in range(1, len(time_folders)):
        time_i = f'T_{i:02d}'

        #Načtení moving obrazu, který je v dané iteraci registrován
        moving_image = Nacteni_image.data_load(os.path.join(read_path, time_folders[i]))
        moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

        #Počáteční zarovnání podle geometrických středů obou obrazů (fixed a daný moving)
        initial_transformation = sitk.CenteredTransformInitializer(
            fixed_image,
            moving_image,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )

        #Inicializace objektu registrace
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetInitialTransform(initial_transformation, inPlace=False)

        #Nastavení parametrů rigidní registrace
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=200, estimateLearningRate=sitk.ImageRegistrationMethod.Never)
        registration_method.SetOptimizerScalesFromPhysicalShift()

        #Nastevní pyramidového přístupu, pro zrychlení registrace (od hrubého rozlišení po jemné)
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        #Výpočet podobnostní metriky z náhodných 5% voxelů (pro urychlení registrace)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.05)

        #Výpočet a aplikace transformace
        final_rigid_transformation = registration_method.Execute(fixed_image, moving_image)
        moving_resampled = sitk.Resample(moving_image, fixed_image, final_rigid_transformation, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

        #Vytvoření ořezané masky i pro pohyblivý (moving) obraz
        moving_resampled_array = sitk.GetArrayFromImage(moving_resampled)
        moving_maska_array = (moving_resampled_array > 25).astype(np.uint8)
        moving_maska_array[:, substracting:, :] = 0
        moving_maska_image = sitk.GetImageFromArray(moving_maska_array)
        moving_maska_image.CopyInformation(moving_resampled)

        #Inicializace mřížky pro deformaci
        mesh_size = [5, 5, 5]

        #Inicializace objektu B-spline registrace
        bspline_transformation = sitk.BSplineTransformInitializer(fixed_image, mesh_size)
        bspline_reg = sitk.ImageRegistrationMethod()
        bspline_reg.SetInitialTransform(bspline_transformation, inPlace=False)

        #Nastavení podobnostní metriky a interpolace
        bspline_reg.SetMetricAsJointHistogramMutualInformation(numberOfHistogramBins=110)
        bspline_reg.SetInterpolator(sitk.sitkBSpline)

        #Nastevní pyramidového přístupu, pro zrychlení registrace (od hrubého rozlišení po jemné)
        bspline_reg.SetShrinkFactorsPerLevel(shrinkFactors=[8, 4, 2, 1])
        bspline_reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[3, 2, 1, 0])
        bspline_reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        #Nastevní optimalizátoru
        bspline_reg.SetOptimizerAsLBFGS2(numberOfIterations=140, solutionAccuracy=1.873476907101121e-08, deltaConvergenceTolerance=1.156746444484934e-05)

        #Výpočet podobnostní metriky z náhodných 5% voxelů (pro urychlení registrace)
        bspline_reg.SetMetricSamplingStrategy(bspline_reg.RANDOM)
        bspline_reg.SetMetricSamplingPercentage(0.05)

        #Aplikace masky nastavené masky, pro výpočet podobnostní metriky, na moving a fixed image
        bspline_reg.SetMetricFixedMask(fixed_maska_image)
        bspline_reg.SetMetricMovingMask(moving_maska_image)

        #Výpočet a aplikace nastavené B-spline transformace
        bspline_final_transformation = bspline_reg.Execute(fixed_image, moving_resampled)
        moving_bspline_resampled = sitk.Resample(moving_resampled, fixed_image, bspline_final_transformation,
                                                 sitk.sitkBSpline, 0.0, moving_resampled.GetPixelID())

        #Pridani do seznamu jako ITK obrazy
        registrovane.append(moving_bspline_resampled)
        print(f'Moving cislo {i} byl ulozen')

        #Vypocet metrik (statistika)
        resampled_float = sitk.Cast(moving_bspline_resampled, sitk.sitkFloat32)

        mi_i = evaluate_mi.MetricEvaluate(fixed_float, resampled_float)
        mse_i = evaluate_mse.MetricEvaluate(fixed_float, resampled_float)

        mi_scores.append(mi_i)
        mse_scores.append(mse_i)
        time_stamps.append(time_i)

    #Spojení do 4D objemu objektu simpleITK
    print('Spojovani do 4D')
    img_all = sitk.JoinSeries(registrovane)

    #Vytvoření složky, pokud neexstiuje.
    os.makedirs(save_path, exist_ok=True)

    #Uložení 4D objemu do formátu NIfFTI
    save_path_niftii = os.path.join(save_path, f'registered_4D_mi_{pacient_id}.nii.gz') #nazev souboru, potencionalne jako vstupni argument
    sitk.WriteImage(img_all, save_path_niftii)

    #Ulozeni metrik co .csv
    filename = os.path.join(save_path, f'vysledky_metrik_mi_{pacient_id}.csv')
    with open(filename, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['Časový okamžik', 'MI_skóre', 'MSE_skóre'])
        for timee, mi, mse in zip(time_stamps, mi_scores, mse_scores):
            writer.writerow([timee, mi, mse])

def registration_full_demons(read_path, save_path, pacient_id):
    """
        Provede registraci 4D DCE-MRI obrazů pomocí algoritmu Diffeomorphic Demons
        a vypočítá metriky podobnosti (MI a MSE).

        Proces začíná hrubým rigidním zarovnáním. Následně je na referenční i pohyblivý
        obraz aplikována maska (eliminace hrudníku a srdce) a provede se Histogram Matching
        pro sjednocení rozložení intenzit. Nakonec je vypočítáno deformační vektorové pole.
        Algoritmus využívá nastavení pro elastické a viskózní vyhlazování pole.
        Pro každý zregistrovaný snímek průběžně počítá hodnoty Mutual Information (MI)
        a Mean Squares (MSE).

        Parametry:
        ----------
        read_path : str
            Cesta ke složce obsahující DICOM data pacienta (bere v potaz abecední pořadí).
        save_path : str
            Cesta ke složce, kam se uloží výsledný NIfTI objem a CSV tabulka s metrikami.
        pacient_id : str nebo int
            Identifikátor pacienta, který se přidá na konec názvů výstupních souborů
            (např. '10' nebo 'Pacient_3').

        Výstupy:
        --------
        Funkce nevrací žádné proměnné, ale ukládá na disk:
        1. 'registered_4D_demons_{pacient_id}.nii.gz' - Zaregistrovaný 4D objem obrazů.
        2. 'vysledky_metrik_demons_{pacient_id}.csv' - Tabulka vývoje metrik MI a MSE v čase.
        """

    time_folders = []
    registrovane = []
    mi_scores = []
    mse_scores = []
    time_stamps = []

    #Načtení cesty, ke všem časovým složkám, je potřeba je abecedně seřadit (01, 02..)
    for folder in sorted(os.listdir(read_path)):
        if os.path.isdir(os.path.join(read_path, folder)):
            cesta = os.path.join(read_path, folder)
            time_folders.append(cesta)

    #Načtení prvního časového okamžiku, ten slouží jako referenční pro všechny ostatní
    fixed_image = Nacteni_image.data_load(time_folders[0])
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    fixed_float = sitk.Cast(fixed_image, sitk.sitkFloat32)

    registrovane.append(fixed_image)

    #Tvorba masky pomocí prahování. Nastavení minimální intenzity pixelu na >25.
    #Následně je spodních 55% obrazu vynulovaných, snaží se eliminovat srdce z výpočtu.
    fixed_maska_array = (sitk.GetArrayFromImage(fixed_image) > 25).astype(np.uint8)
    height = fixed_maska_array.shape[1]
    substracting = int(height * 0.55)
    fixed_maska_array[:, substracting:, :] = 0

    fixed_maska_image = sitk.GetImageFromArray(fixed_maska_array)
    fixed_maska_image.CopyInformation(fixed_image)
    fixed_float_masked = sitk.Mask(fixed_float, fixed_maska_image)

    #Inicializace evaluace pro průběžný výpočet podobnosti (statistika)
    evaluate_mi = sitk.ImageRegistrationMethod()
    evaluate_mi.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    evaluate_mi.SetMetricFixedMask(fixed_maska_image)

    evaluate_mse = sitk.ImageRegistrationMethod()
    evaluate_mse.SetMetricAsMeanSquares()
    evaluate_mse.SetMetricFixedMask(fixed_maska_image)

    #Výpočet statistiky na prvním (fixním) obrazu
    fixed_float = sitk.Cast(fixed_image, sitk.sitkFloat32)
    mi_scores.append(evaluate_mi.MetricEvaluate(fixed_float,fixed_float))
    mse_scores.append(evaluate_mse.MetricEvaluate(fixed_float, fixed_float))
    time_stamps.append('T_00')

    #Iterativní registrace pohyblivých obrazů
    for i in range(1, len(time_folders)):
        time_i = f'T_{i:02d}'

        #Načtení moving obrazu, který je v dané iteraci registrován
        moving_image = Nacteni_image.data_load(os.path.join(read_path, time_folders[i]))
        moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

        #Počáteční zarovnání podle geometrických středů obou obrazů (fixed a daný moving)
        initial_transformation = sitk.CenteredTransformInitializer(
            # Pocateni zarovnání, střed objemu je na sobe, vycentrovan
            fixed_image,  # referenční obraz
            moving_image,  # zarovnávaný obraz
            sitk.Euler3DTransform(),  # rigidni transformace
            sitk.CenteredTransformInitializerFilter.GEOMETRY  # určení středu obrazů
        )

        #Inicializace objektu registrace
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetInitialTransform(initial_transformation, inPlace=False)

        #Nastavení parametrů rigidní registrace
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=200, estimateLearningRate=sitk.ImageRegistrationMethod.Never)
        registration_method.SetOptimizerScalesFromPhysicalShift()

        #Nastevní pyramidového přístupu, pro zrychlení registrace (od hrubého rozlišení po jemné)
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        #Výpočet podobnostní metriky z náhodných 5% voxelů (pro urychlení registrace)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.05)

        #Výpočet a aplikace transformace
        final_rigid_transformation = registration_method.Execute(fixed_image, moving_image)
        moving_resampled = sitk.Resample(moving_image, fixed_image, final_rigid_transformation, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

        #Vytvoření ořezané masky i pro pohyblivý (moving) obraz
        moving_float = sitk.Cast(moving_resampled, sitk.sitkFloat32)
        moving_maska_array = (sitk.GetArrayFromImage(moving_resampled) > 25).astype(np.uint8)
        moving_maska_array[:, substracting:, :] = 0
        moving_maska_image = sitk.GetImageFromArray(moving_maska_array)
        moving_maska_image.CopyInformation(moving_resampled)

        #Aplikace masky na moving obraz
        moving_float_masked = sitk.Mask(moving_float, moving_maska_image)

        #Úprava rozložení intenzit pohyblivého obrazu tak, aby odpovídala referenčnímu obrazu.
        matcher = sitk.HistogramMatchingImageFilter()
        matcher.SetNumberOfHistogramLevels(1024)
        matcher.SetNumberOfMatchPoints(7)
        matcher.ThresholdAtMeanIntensityOn()

        moving_matched = matcher.Execute(moving_float_masked, fixed_float_masked)

        #Inicializace Diffeomorphic Demons registrace
        demons = sitk.DiffeomorphicDemonsRegistrationFilter()

        #Hodnoty vyhlazování (získané optimalizací) řídí charakter deformačního pole:
        demons.SetNumberOfIterations(660)
        demons.SetStandardDeviations(1.0000138310824247)
        demons.SetUpdateFieldStandardDeviations(2.6910533404994568)

        #Výpočet vektorového pole pohybu a převedení na objekt final_demons
        demons_field = demons.Execute(fixed_float_masked, moving_matched)
        final_demons = sitk.DisplacementFieldTransform(demons_field)

        #Aplikace transformace na původní moving image (zachování původní intenzity obrazu)
        demons_resampled = sitk.Resample(moving_resampled, fixed_image, final_demons, sitk.sitkLinear, 0.0,
                                         moving_image.GetPixelID())

        #Pridani do seznamu jako ITK obrazy
        registrovane.append(demons_resampled)
        print(f'Moving cislo {i} byl ulozen')

        #Vypocet metrik (statistika)
        resampled_float = sitk.Cast(demons_resampled, sitk.sitkFloat32)

        mi_i = evaluate_mi.MetricEvaluate(fixed_float, resampled_float)
        mse_i =evaluate_mse.MetricEvaluate(fixed_float, resampled_float)

        mi_scores.append(mi_i)
        mse_scores.append(mse_i)
        time_stamps.append(time_i)

    #Spojení do 4D objemu objektu simpleITK
    print('Spojovani do 4D')
    img_all = sitk.JoinSeries(registrovane)

    #Vytvoření složky, pokud neexstiuje.
    os.makedirs(save_path, exist_ok=True)

    #Uložení 4D objemu do formátu NIfFTI
    save_path_niftii = os.path.join(save_path, f'registered_4D_demons_{pacient_id}.nii.gz') #nazev souboru, potencionalne jako vstupni argument
    sitk.WriteImage(img_all, save_path_niftii)

    #Ulozeni metrik co .csv
    filename = os.path.join(save_path, f'vysledky_metrik_demons_{pacient_id}.csv')
    with open(filename, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['Časový okamžik', 'MI_skóre', 'MSE_skóre'])
        for timee, mi, mse in zip(time_stamps, mi_scores, mse_scores):
            writer.writerow([timee, mi, mse])
