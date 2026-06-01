import optuna
import SimpleITK as sitk
import numpy as np
import Nacteni_image
import os

def optuna_mse(fixed_dir, moving_dir, save_path, pacient_id, n_trials=200):
    """
    Optimalizuje hyperparametry pro nelineární B-spline registraci
    (podobnostní metrika MSE) pomocí frameworku Optuna.


    Funkce načte referenční (fixed) a pohyblivý (moving) obraz a provede jejich
    počáteční rigidní zarovnání. Následně definuje optimalizační prostor pro
    B-spline registraci (hustota mřížky, počet iterací a toleranční parametry
    optimalizátoru LBFGS2). Optuna iterativně hledá nejlepší kombinaci těchto
    parametrů tak, aby minimalizovala hodnotu metriky Mutual Information (MI),
    která se počítá na tkáni prsu (s aplikovanou maskou). Nejkvalitnější
    nalezená transformace a průběžné výsledky se ukládají na disk.

    Parametry:
    ----------
    fixed_dir : str
        Cesta ke složce s DICOM daty referenčního snímku (bere v potaz abecední pořadí).
    moving_dir : str
        Cesta ke složce s DICOM daty zarovnávaného snímku (bere v potaz abecední pořadí).
    save_path : str
        Cesta ke složce, kam se uloží databáze, transformace a textové logy.
    pacient_id : str nebo int
        Identifikátor pacienta pro dynamické pojmenování výstupních souborů.
    n_trials : int, volitelné
        Počet iterací pro hledání hyperparametrů (výchozí je 200).

    Výstupy:
    --------
    Funkce nevrací žádné proměnné, ale ukládá do zadané složky:
    1. 'rigid_transform_mse_{pacient_id}.tfm' - Počáteční rigidní transformace.
    2. 'best_transform_mse_{pacient_id}.tfm' - Nejlepší nalezená B-spline transformace.
    3. 'prubezne_ukladani_mse_{pacient_id}.txt' - Log s nejlepšími parametry.
    4. 'mse_optuna_{pacient_id}.db' - SQLite databáze s historií optimalizace.
    """
    #Nacteni zdrojových dat
    data_base = Nacteni_image.data_load(fixed_dir)
    data_chang = Nacteni_image.data_load(moving_dir)

    #Převod načtených numpy polí, na formát pro SimpleITK
    #Float pro přesné výpočty optimalizačních algoritmů
    fixed_image = sitk.Cast(data_base, sitk.sitkFloat32)
    moving_image = sitk.Cast(data_chang, sitk.sitkFloat32)

    # Počáteční zarovnání podle geometrických středů obou obrazů (fixed a daný moving)
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


    #Inicalizace metriky podobnosti
    initial_metric = registration_method.MetricEvaluate(fixed_image, moving_image)
    print('Prvotni hodnota:', initial_metric)
    #Výpočet transformace
    final_transform = registration_method.Execute(fixed_image, moving_image)
    print('Final metric value: {0}'.format(registration_method.GetMetricValue()))

    #Ulozeni rigidní transformace
    rigid_tfm_path = os.path.join(save_path, f'rigid_transform_mse_{pacient_id}.tfm')
    sitk.WriteTransform(final_transform, rigid_tfm_path)

    #Aplikace rigidní transformace
    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

    #Tvorba masky pomocí prahování. Nastavení minimální intenzity pixelu na >25.
    #Následně je spodních 55% obrazu vynulovaných, snaží se eliminovat srdce z výpočtu.
    fixed_maska_array = (sitk.GetArrayFromImage(fixed_image) > 25).astype(np.uint8)
    moving_maska_array = (sitk.GetArrayFromImage(moving_resampled) > 25).astype(np.uint8)
    height = fixed_maska_array.shape[1]  # nacteni vysky snimku
    substracting = int(height * 0.55)  # Oříznutí spodních 45% snímku
    fixed_maska_array[:, substracting:, :] = 0
    moving_maska_array[:, substracting:, :] = 0
    fixed_maska_image = sitk.GetImageFromArray(fixed_maska_array)
    moving_maska_image = sitk.GetImageFromArray(moving_maska_array)
    fixed_maska_image.CopyInformation(fixed_image)
    moving_maska_image.CopyInformation(moving_resampled)

    #Inicializace evaluace pro průběžný výpočet podobnosti (statistika)
    evaluator = sitk.ImageRegistrationMethod()
    evaluator.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    evaluator.SetMetricFixedMask(fixed_maska_image)

    #Udržení nelepšího skóre, hledáme minimum (záporné maximum), nastavení na plus nekonečno
    mi_checking = [float('inf')]

    #Účelová funkce pro Optunu, probíhá zde testování hyperparametrů
    def objective_mse(trial):

        #Optuna "navrhuje" hodnoty parametrů ze zadaných rozsahů.
        #Hledáme optimální hustotu mřížky, počet iterací a přesnost konvergence pro LBFGS2.
        update_grid = trial.suggest_int('MeshSize', 2, 10)
        update_iterations = trial.suggest_int('Iterations', 60, 800, step=20)
        update_sol_acc = trial.suggest_float('SolutionAccuracy', 1e-9, 1e-2, log=True)
        update_convergence = trial.suggest_float('ConvergenceTolerance', 1e-9, 1e-2, log = True)

        #Inicializace mřížky pro deformaci
        mesh_size = [update_grid, update_grid, update_grid]

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

        # Výpočet podobnostní metriky z náhodných 5% voxelů (pro urychlení registrace)
        bspline_reg.SetMetricSamplingStrategy(bspline_reg.RANDOM)
        bspline_reg.SetMetricSamplingPercentage(0.05)

        #Nastevní optimalizátoru
        bspline_reg.SetOptimizerAsLBFGS2(numberOfIterations=update_iterations, solutionAccuracy=update_sol_acc, deltaConvergenceTolerance=update_convergence)

        #Aplikace masky nastavené masky, pro výpočet podobnostní metriky, na moving a fixed image
        bspline_reg.SetMetricMovingMask(moving_maska_image)
        bspline_reg.SetMetricFixedMask(fixed_maska_image)

        #Výpočet a aplikace nastavené B-spline transformace
        bspline_final_transformation = bspline_reg.Execute(fixed_image, moving_resampled)
        print('Bspline metric:', bspline_reg.GetMetricValue())
        moving_bspline_resampled = sitk.Resample(moving_resampled, fixed_image, bspline_final_transformation,
                                             sitk.sitkBSpline, 0.0, moving_resampled.GetPixelID())

        #Zhodnocení kvality výsledku pomocí nezávislého evaluátoru (MI) s maskou.
        #Toto číslo se vrací Optuně a ta podle něj upravuje parametry pro další pokus.
        score = evaluator.MetricEvaluate(fixed_image, sitk.Cast(moving_bspline_resampled, sitk.sitkFloat32))

        #Pokud je aktuální skóre lepší (menší) než dosud nalezené, uložíme si transformaci
        if score < mi_checking[0]:
            mi_checking[0] = score
            best_tfm_path = os.path.join(save_path, f'best_transform_mse_{pacient_id}.tfm')
            sitk.WriteTransform(bspline_final_transformation, best_tfm_path)

        return score

    #Nastavení průběžného ukládání
    def saving(study, trial):

        #Spuštěna při každém pokus, ukládá nejlepší výsledek
        if study.best_trial.number == trial.number:
            txt_path = os.path.join(save_path, f'prubezne_ukladani_mse_{pacient_id}.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write('Pruběžné výsledky optimalizace pomocí mse \n')
                f.write(f'Číslo pokusu: {trial.number} \n')
                f.write(f'Skore (MI): {study.best_value}\n')
                f.write('Nejlepší parametry (so far) \n')
                for par, value in study.best_params.items():
                    f.write(f' {par}: {value}\n')

    #Nastavení cest pro ukládání databáze (umožňuje navázat po výpadku)
    db_path = os.path.join(save_path, f'mse_optuna_{pacient_id}.db')
    study_name = f'mse_optimize_{pacient_id}'

    #Inicializace studie s cílem 'minimize' (protože SimpleITK vrací MI jako záporné číslo)
    study = optuna.create_study(direction='minimize',
                                study_name = study_name,
                                storage = f'sqlite:///{db_path}',
                                load_if_exists = True)

    #Zde program začne zkoušet různé kombinace hodnot a po každém pokusu si poznamená, jak dopadl.
    study.optimize(objective_mse, n_trials=n_trials, callbacks=[saving])
    print('Hotovo, konec optimalizace')

def optuna_mi(fixed_dir, moving_dir, save_path, pacient_id, n_trials=200):
    """
    Optimalizuje hyperparametry pro nelineární B-spline registraci
    (podobnostní metrika MI) pomocí frameworku Optuna.


    Funkce načte referenční (fixed) a pohyblivý (moving) obraz a provede jejich
    počáteční rigidní zarovnání. Následně definuje optimalizační prostor pro
    B-spline registraci (hustota mřížky, počet histogramových košů,
    počet iterací a toleranční parametry optimalizátoru LBFGS2).
    Optuna iterativně hledá nejlepší kombinaci těchto parametrů tak, aby
    minimalizovala hodnotu metriky Mutual Information (MI), která
    se počítá na tkáni prsu (s aplikovanou maskou). Nejkvalitnější
    nalezená transformace a průběžné výsledky se ukládají na disk.

    Parametry:
    ----------
    fixed_dir : str
        Cesta ke složce s DICOM daty referenčního snímku (bere v potaz abecední pořadí).
    moving_dir : str
        Cesta ke složce s DICOM daty zarovnávaného snímku (bere v potaz abecední pořadí).
    save_path : str
        Cesta ke složce, kam se uloží databáze, transformace a textové logy.
    pacient_id : str nebo int
        Identifikátor pacienta pro dynamické pojmenování výstupních souborů.
    n_trials : int, volitelné
        Počet iterací pro hledání hyperparametrů (výchozí je 200).

    Výstupy:
    --------
    Funkce nevrací žádné proměnné, ale ukládá do zadané složky:
    1. 'rigid_transform_mse_{pacient_id}.tfm' - Počáteční rigidní transformace.
    2. 'best_transform_mse_{pacient_id}.tfm' - Nejlepší nalezená B-spline transformace.
    3. 'prubezne_ukladani_mse_{pacient_id}.txt' - Log s nejlepšími parametry.
    4. 'mse_optuna_{pacient_id}.db' - SQLite databáze s historií optimalizace.
    """
    #Nacteni zdrojových dat
    data_base = Nacteni_image.data_load(fixed_dir)
    data_chang = Nacteni_image.data_load(moving_dir)

    #Převod načtených numpy polí, na formát pro SimpleITK
    #Float pro přesné výpočty optimalizačních algoritmů
    fixed_image = sitk.Cast(data_base, sitk.sitkFloat32)
    moving_image = sitk.Cast(data_chang, sitk.sitkFloat32)

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
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=200,
                                                      estimateLearningRate=sitk.ImageRegistrationMethod.Never)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Nastevní pyramidového přístupu, pro zrychlení registrace (od hrubého rozlišení po jemné)
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    #Inicalizace metriky podobnosti
    initial_metric = registration_method.MetricEvaluate(fixed_image, moving_image)
    print('Prvotni hodnota:', initial_metric)
    #Výpočet transformace
    final_transform = registration_method.Execute(fixed_image, moving_image)
    print('Final metric value: {0}'.format(registration_method.GetMetricValue()))

    #Ulozeni rigidní transformace
    rigid_tfm_path = os.path.join(save_path, f'rigid_transform_mi_{pacient_id}.tfm')
    sitk.WriteTransform(final_transform, rigid_tfm_path)

    #Aplikace rigidní transformace
    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0,
                                     moving_image.GetPixelID())

    #Tvorba masky pomocí prahování. Nastavení minimální intenzity pixelu na >25.
    #Následně je spodních 55% obrazu vynulovaných, snaží se eliminovat srdce z výpočtu.
    fixed_maska_array = (sitk.GetArrayFromImage(fixed_image) > 25).astype(np.uint8)
    moving_maska_array = (sitk.GetArrayFromImage(moving_resampled) > 25).astype(np.uint8)
    height = fixed_maska_array.shape[1]  # nacteni vysky snimku
    substracting = int(height * 0.55)  # Oříznutí spodních 45% snímku
    fixed_maska_array[:, substracting:, :] = 0
    moving_maska_array[:, substracting:, :] = 0
    fixed_maska_image = sitk.GetImageFromArray(fixed_maska_array)
    moving_maska_image = sitk.GetImageFromArray(moving_maska_array)
    fixed_maska_image.CopyInformation(fixed_image)
    moving_maska_image.CopyInformation(moving_resampled)

    #Inicializace evaluace pro průběžný výpočet podobnosti (statistika)
    evaluator = sitk.ImageRegistrationMethod()
    evaluator.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    evaluator.SetMetricFixedMask(fixed_maska_image)

    #Udržení nelepšího skóre, hledáme minimum (záporné maximum), nastavení na plus nekonečno
    mi_checking = [float('inf')]

    #Účelová funkce pro Optunu, probíhá zde testování hyperparametrů
    def objective_mi(trial):

        #Optuna "navrhuje" hodnoty parametrů ze zadaných rozsahů.
        #Hledáme optimální hustotu mřížky, počet iterací, počet histogramových košů a přesnost konvergence pro LBFGS2.
        update_grid = trial.suggest_int('MeshSize', 2, 10)
        update_iterations = trial.suggest_int('Iterations', 60, 800, step=20)
        update_bins = trial.suggest_int('HistogramBins', 20, 130, step= 10)
        update_sol_acc = trial.suggest_float('SolutionAccuracy', 1e-9, 1e-2, log=True)
        update_convergence = trial.suggest_float('ConvergenceTolerance', 1e-9, 1e-2, log = True)

        #Inicializace mřížky pro deformaci
        mesh_size = [update_grid, update_grid, update_grid]

        #Inicializace objektu B-spline registrace
        bspline_transformation = sitk.BSplineTransformInitializer(fixed_image, mesh_size)
        bspline_reg = sitk.ImageRegistrationMethod()


        #Nastavení podobnostní metriky a interpolace
        bspline_reg.SetMetricAsJointHistogramMutualInformation(update_bins)
        bspline_reg.SetInterpolator(sitk.sitkBSpline)
        bspline_reg.SetInitialTransform(bspline_transformation, inPlace=False)

        #Nastevní pyramidového přístupu, pro zrychlení registrace (od hrubého rozlišení po jemné)
        bspline_reg.SetShrinkFactorsPerLevel(shrinkFactors=[8, 4, 2, 1])
        bspline_reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[3, 2, 1, 0])
        bspline_reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # Výpočet podobnostní metriky z náhodných 5% voxelů (pro urychlení registrace)
        bspline_reg.SetMetricSamplingStrategy(bspline_reg.RANDOM)
        bspline_reg.SetMetricSamplingPercentage(0.05)

        #Nastevní optimalizátoru
        bspline_reg.SetOptimizerAsLBFGS2(numberOfIterations=update_iterations,solutionAccuracy= update_sol_acc,deltaConvergenceTolerance= update_convergence)

        #Aplikace masky nastavené masky, pro výpočet podobnostní metriky, na moving a fixed image
        bspline_reg.SetMetricMovingMask(moving_maska_image)
        bspline_reg.SetMetricFixedMask(fixed_maska_image)

        #Výpočet a aplikace nastavené B-spline transformace
        bspline_final_transformation = bspline_reg.Execute(fixed_image, moving_resampled)
        print('Bspline metric:', bspline_reg.GetMetricValue())
        moving_bspline_resampled = sitk.Resample(moving_resampled, fixed_image, bspline_final_transformation,
                                             sitk.sitkBSpline, 0.0, moving_resampled.GetPixelID())

        #Zhodnocení kvality výsledku pomocí nezávislého evaluátoru (MI) s maskou.
        #Toto číslo se vrací Optuně a ta podle něj upravuje parametry pro další pokus.
        score = evaluator.MetricEvaluate(fixed_image, sitk.Cast(moving_bspline_resampled, sitk.sitkFloat32))

        #Pokud je aktuální skóre lepší (menší) než dosud nalezené, uložíme si transformaci
        if score < mi_checking[0]:
            mi_checking[0] = score
            best_tfm_path = os.path.join(save_path, f'best_transform_mi_{pacient_id}.tfm')
            sitk.WriteTransform(bspline_final_transformation, best_tfm_path)

        return score

    #Nastavení průběžného ukládání
    def saving(study, trial):

        #Spuštěna při každém pokus, ukládá nejlepší výsledek
        if study.best_trial.number == trial.number:
            txt_path = os.path.join(save_path, f'prubezne_ukladani_mi_{pacient_id}.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write('Pruběžné výsledky optimalizace pomocí mi \n')
                f.write(f'Číslo pokusu: {trial.number} \n')
                f.write(f'Skore (MI): {study.best_value}')
                f.write('Nejlepší parametry (so far) \n')
                for par, value in study.best_params.items():
                    f.write(f' {par}: {value}\n')

    #Nastavení cest pro ukládání databáze (umožňuje navázat po výpadku)
    db_path = os.path.join(save_path, f'mi_optuna_{pacient_id}.db')
    study_name = f'mi_optimize_{pacient_id}'

    #Inicializace studie s cílem 'minimize' (protože SimpleITK vrací MI jako záporné číslo)
    study = optuna.create_study(direction='minimize',
                                study_name = study_name,
                                storage = f'sqlite:///{db_path}',
                                load_if_exists = True)

    #Zde program začne zkoušet různé kombinace hodnot a po každém pokusu si poznamená, jak dopadl.
    study.optimize(objective_mi, n_trials=n_trials, callbacks=[saving])
    print('Hotovo, konec optimalizace')

def optuna_demons(fixed_dir, moving_dir, save_path, pacient_id, n_trials=200):
    """
    Optimalizuje hyperparametry pro registraci algoritmem Diffeomorphic Demons
    pomocí knihovny Optuna.

    Funkce nejprve provede hrubé rigidní zarovnání. Poté aplikuje na oba obrazy
    masku (odstranění srdce a hrudníku) a sjednotí jejich jasové spektrum pomocí
    Histogram Matchingu. Následně Optuna prozkoumává prostor hyperparametrů pro
    Demons algoritmus (počet iterací, elastické a viskózní vyhlazování vektorového
    pole). Jako hodnotící kritérium kvality deformace je použita metrika Mutual
    Information (MI). Průběžné výsledky a nejlepší deformační pole se ukládají.

    Parametry:
    ----------
    fixed_dir : str
        Cesta ke složce s DICOM daty referenčního snímku (bere v potaz abecední pořadí).
    moving_dir : str
        Cesta ke složce s DICOM daty zarovnávaného snímku (bere v potaz abecední pořadí).
    save_path : str
        Cesta ke složce, kam se uloží databáze, transformace a textové logy.
    pacient_id : str nebo int
        Identifikátor pacienta pro dynamické pojmenování výstupních souborů.
    n_trials : int, volitelné
        Počet iterací pro hledání hyperparametrů (výchozí je 200).

    Výstupy:
    --------
    Funkce nevrací žádné proměnné, ale ukládá do zadané složky:
    1. 'rigid_transform_demons_{pacient_id}.tfm' - Počáteční rigidní transformace.
    2. 'best_transform_demons_{pacient_id}.tfm' - Nejlepší nalezené deformační pole.
    3. 'prubezne_ukladani_demons_{pacient_id}.txt' - Log s nejlepšími parametry.
    4. 'demons_optuna_{pacient_id}.db' - SQLite databáze s historií optimalizace.
    """

    #Nacteni zdrojových dat
    data_base = Nacteni_image.data_load(fixed_dir)
    data_chang = Nacteni_image.data_load(moving_dir)

    #Převod načtených numpy polí, na formát pro SimpleITK
    #Float pro přesné výpočty optimalizačních algoritmů
    fixed_image = sitk.Cast(data_base, sitk.sitkFloat32)
    moving_image = sitk.Cast(data_chang, sitk.sitkFloat32)

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
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=200,
                                                      estimateLearningRate=sitk.ImageRegistrationMethod.Never)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Nastevní pyramidového přístupu, pro zrychlení registrace (od hrubého rozlišení po jemné)
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    #Inicalizace metriky podobnosti
    initial_metric = registration_method.MetricEvaluate(fixed_image, moving_image)
    print('Prvotni hodnota:', initial_metric)
    #Výpočet transformace
    final_transform = registration_method.Execute(fixed_image, moving_image)
    print('Final metric value: {0}'.format(registration_method.GetMetricValue()))

    #Ulozeni rigidní transformace
    rigid_tfm_path = os.path.join(save_path, f'rigid_transform_demons_{pacient_id}.tfm')
    sitk.WriteTransform(final_transform, rigid_tfm_path)

    #Aplikace rigidní transformace
    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0,
                                     moving_image.GetPixelID())

    #Přetypování pro následné maskování a Histogram Matching
    fixed_float = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_float = sitk.Cast(moving_resampled, sitk.sitkFloat32)

    #Tvorba masky pomocí prahování. Nastavení minimální intenzity pixelu na >25.
    #Následně je spodních 55% obrazu vynulovaných, snaží se eliminovat srdce z výpočtu.
    fixed_maska_array = (sitk.GetArrayFromImage(fixed_image) > 25).astype(np.uint8)
    moving_maska_array = (sitk.GetArrayFromImage(moving_resampled) > 25).astype(np.uint8)
    height = fixed_maska_array.shape[1]
    substracting = int(height * 0.55)
    fixed_maska_array[:, substracting:, :] = 0
    moving_maska_array[:, substracting:, :] = 0
    fixed_maska_image = sitk.GetImageFromArray(fixed_maska_array)
    moving_maska_image = sitk.GetImageFromArray(moving_maska_array)
    fixed_maska_image.CopyInformation(fixed_image)
    moving_maska_image.CopyInformation(moving_resampled)

    #Aplikujeme tedy masky přímo na samotná obrazová data (Demons to vyžaduje).
    fixed_float_masked = sitk.Mask(fixed_float, fixed_maska_image)
    moving_float_masked = sitk.Mask(moving_float, moving_maska_image)

    #Přemapování intenzity pohyblivého obrazu podle referenčního.
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(1024)  # Rozdeleni na 1024 odstinu
    matcher.SetNumberOfMatchPoints(7)  # pocet bodu křivky, ktere se musi střetnout
    matcher.ThresholdAtMeanIntensityOn()  # Ignoruje černé pozadí - bere průměrný jas

    moving_matched = matcher.Execute(moving_float_masked, fixed_float_masked)

    #Inicializace evaluace pro průběžný výpočet podobnosti (statistika)
    evaluator = sitk.ImageRegistrationMethod()
    evaluator.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    evaluator.SetMetricFixedMask(fixed_maska_image)

    #Udržení nelepšího skóre, hledáme minimum (záporné maximum), nastavení na plus nekonečno
    mi_checking = [float('inf')]

    #Účelová funkce pro Optunu, probíhá zde testování hyperparametrů
    def objective_demons(trial):

        #Optuna "navrhuje" hodnoty parametrů ze zadaných rozsahů.
        #Hledáme optimální počet iterací, hodnoty pro elastické a viskózní vyhlazování vektorového pole
        update_standard = trial.suggest_float('StandardDeviations', 1, 3.5)
        update_field_standard = trial.suggest_float('FieldStandardDeviations', 0.5, 3.5)
        update_iterations = trial.suggest_int('Iterations', 60, 800, step = 20)

        #Inicializace Diffeomorphic Demons registrace
        demons = sitk.DiffeomorphicDemonsRegistrationFilter()

        #Hodnoty vyhlazování řídí charakter deformačního pole:
        demons.SetNumberOfIterations(update_iterations)
        demons.SetStandardDeviations(update_standard)
        demons.SetUpdateFieldStandardDeviations(update_field_standard)

        #Výpočet vektorového pole pohybu a převedení na objekt final_demons
        demons_field = demons.Execute(fixed_float_masked, moving_matched)
        final_demons = sitk.DisplacementFieldTransform(demons_field)

        #Aplikace transformace na původní moving image (zachování původní intenzity obrazu)
        demons_resampled = sitk.Resample(moving_resampled, fixed_image, final_demons, sitk.sitkBSpline, 0.0,
                                         moving_image.GetPixelID())

        #Zhodnocení kvality výsledku pomocí nezávislého evaluátoru (MI) s maskou.
        #Toto číslo se vrací Optuně a ta podle něj upravuje parametry pro další pokus.
        score = evaluator.MetricEvaluate(fixed_float, sitk.Cast(demons_resampled, sitk.sitkFloat32))

        #Pokud je aktuální skóre lepší (menší) než dosud nalezené, uložíme si transformaci
        if score < mi_checking[0]:
            mi_checking[0] = score
            best_tfm_path = os.path.join(save_path, f'best_transform_demons_{pacient_id}.tfm')
            sitk.WriteTransform(final_demons, best_tfm_path)

        return score


    #Nastavení průběžného ukládání
    def saving(study, trial):

        #Spuštěna při každém pokus, ukládá nejlepší výsledek
        if study.best_trial.number == trial.number:
            txt_path = os.path.join(save_path, f'prubezne_ukladani_demons_{pacient_id}.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write('Pruběžné výsledky optimalizace pomocí demons \n')
                f.write(f'Číslo pokusu: {trial.number} \n')
                f.write(f'Skore (MI): {study.best_value}')
                f.write('Nejlepší parametry (so far) \n')
                for par, value in study.best_params.items():
                    f.write(f' {par}: {value}\n')

    #Nastavení cest pro ukládání databáze (umožňuje navázat po výpadku)
    db_path = os.path.join(save_path, f'demons_optuna_{pacient_id}.db')
    study_name = f'demons_optimize_{pacient_id}'

    #Inicializace studie s cílem 'minimize' (protože SimpleITK vrací MI jako záporné číslo)
    study = optuna.create_study(direction='minimize',
                                study_name = study_name,
                                storage = f'sqlite:///{db_path}',
                                load_if_exists = True)

    #Zde program začne zkoušet různé kombinace hodnot a po každém pokusu si poznamená, jak dopadl.
    study.optimize(objective_demons, n_trials=n_trials, callbacks=[saving])
    print('Hotovo, konec optimalizace')