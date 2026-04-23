import SimpleITK as sitk
import numpy as np
import Nacteni_image
import optuna
import time
def optuna_mi():
    # Nacteni pomoci Nacteni.py
    data_base = Nacteni_image.data_load(
        r'C:\Users\andre\OneDrive\Plocha\School\BTB_3\Bakalářská práce\subjekty\pokus_1\manifest-data1407430404196\QIN Breast DCE-MRI\QIN-Breast-DCE-MRI-BC05\03-23-1997-NA-Breast CE-93044\07.000000-twist20sdynTRA h20 ex B17TT44.8s-60631')
    data_chang = Nacteni_image.data_load(
        r'C:\Users\andre\OneDrive\Plocha\School\BTB_3\Bakalářská práce\subjekty\pokus_1\manifest-data1407430404196\QIN Breast DCE-MRI\QIN-Breast-DCE-MRI-BC05\03-23-1997-NA-Breast CE-93044\41.000000-twist20sdynTRA h20 ex B17TT353.7s-58006')
    fixed_image = sitk.Cast(data_base, sitk.sitkFloat32)
    moving_image = sitk.Cast(data_chang, sitk.sitkFloat32)

    # Rigidni registrace
    initial_transformation = sitk.CenteredTransformInitializer(
        # Pocateni zarovnání, střed objemu je na sobe, vycentrovan
        fixed_image,  # referenční obraz
        moving_image,  # zarovnávaný obraz
        sitk.Euler3DTransform(),  # rigidni transformace
        sitk.CenteredTransformInitializerFilter.GEOMETRY  # určení středu obrazů
    )

    registration_method = sitk.ImageRegistrationMethod()  # vytvoření objektu registrace, následně to budu nastovovat
    registration_method.SetMetricAsMattesMutualInformation(
        numberOfHistogramBins=50)  # Rozdělí hodnoty intenzity do 50 histogramových binů, metrika podobnosti
    registration_method.SetInterpolator(sitk.sitkLinear)  # lineární dopočet hodnot mezi voxely
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=200,
                                                      estimateLearningRate=sitk.ImageRegistrationMethod.Never)  # Mění parametry transformace, aby se metrika zlepšovala
    registration_method.SetOptimizerScalesFromPhysicalShift()  # Normalizace, nastavení vah parametru transfromace na základě vlivu na obraz, mm ne voxely u descentu
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])  # Pyramidová registrace, od hrube po jemnou
    registration_method.SetSmoothingSigmasPerLevel(
        smoothingSigmas=[2, 1, 0])  # Jak moc se na jednotlivých úrovních pyramidy obraz rozmaže
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()  # Určuje že jednotky jsou v mm ne pixelech, u pyramidy
    registration_method.SetInitialTransform(initial_transformation,
                                            inPlace=False)  # Zvolím začátek registrace - initial_transformation

    # Metrika
    initial_metric = registration_method.MetricEvaluate(fixed_image, moving_image)  # Získá hodnotu podobnosti
    print('Prvotni hodnota:', initial_metric)
    final_transform = registration_method.Execute(fixed_image, moving_image)
    print('Final metric value: {0}'.format(registration_method.GetMetricValue()))

    # Ulozeni rigidni transformace
    sitk.WriteTransform(final_transform, 'rigid_transform_mi_CE-93044.tfm')

    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0,
                                     moving_image.GetPixelID())


    # Tvorba masky - srdce/zebra maji vysoky kontrast..
    fixed_maska_array = (sitk.GetArrayFromImage(fixed_image) > 25).astype(np.uint8)
    moving_maska_array = (sitk.GetArrayFromImage(moving_resampled) > 25).astype(np.uint8)

    height = fixed_maska_array.shape[1]  # nacteni vysky snimku
    print(height)
    substracting = int(height * 0.55)  # Oříznutí spodních 45% snímku

    fixed_maska_array[:, substracting:, :] = 0
    moving_maska_array[:, substracting:, :] = 0

    fixed_maska_image = sitk.GetImageFromArray(fixed_maska_array)
    moving_maska_image = sitk.GetImageFromArray(moving_maska_array)

    fixed_maska_image.CopyInformation(fixed_image)
    moving_maska_image.CopyInformation(moving_resampled)

    evaluator = sitk.ImageRegistrationMethod()
    evaluator.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    evaluator.SetMetricFixedMask(fixed_maska_image)

    # Udrzeni nelepsiho skore, hledame minimuj (zaporne maximum)
    # Pote se ulozi transformacni funkce podle tohoto skore - osetreni random 5%
    mi_checking = [float('inf')]  # nastaveni na infinity
    def objective_mi(trial):

        update_grid = trial.suggest_int('MeshSize', 2, 10)
        update_iterations = trial.suggest_int('Iterations', 60, 800, step=20)
        update_bins = trial.suggest_int('HistogramBins', 20, 130, step= 10)
        update_sol_acc = trial.suggest_float('SolutionAccuracy', 1e-9, 1e-2, log=True)
        update_convergence = trial.suggest_float('ConvergenceTolerance', 1e-9, 1e-2, log = True)
        # Bspline registrace non-rigid
        print('Bspline')

        mesh_size = [update_grid, update_grid, update_grid]
        bspline_transformation = sitk.BSplineTransformInitializer(fixed_image, mesh_size)
        bspline_reg = sitk.ImageRegistrationMethod()
        bspline_reg.SetMetricAsJointHistogramMutualInformation(update_bins)
        bspline_reg.SetMetricSamplingStrategy(bspline_reg.RANDOM)  # Vybere 5% - kvůli náročnosti
        bspline_reg.SetMetricSamplingPercentage(0.05)
        bspline_reg.SetInterpolator(sitk.sitkBSpline)
        bspline_reg.SetShrinkFactorsPerLevel(shrinkFactors=[8, 4, 2, 1])
        bspline_reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[3, 2, 1, 0])
        bspline_reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        bspline_reg.SetInitialTransform(bspline_transformation, inPlace=False)
        bspline_reg.SetOptimizerAsLBFGS2(numberOfIterations=update_iterations,solutionAccuracy= update_sol_acc,deltaConvergenceTolerance= update_convergence)
        bspline_reg.SetMetricMovingMask(moving_maska_image)
        bspline_reg.SetMetricFixedMask(fixed_maska_image)
        bspline_final_transformation = bspline_reg.Execute(fixed_image, moving_resampled)
        print('Bspline metric:', bspline_reg.GetMetricValue())

        moving_bspline_resampled = sitk.Resample(moving_resampled, fixed_image, bspline_final_transformation,
                                             sitk.sitkBSpline, 0.0, moving_resampled.GetPixelID())

        score = evaluator.MetricEvaluate(fixed_image, sitk.Cast(moving_bspline_resampled, sitk.sitkFloat32))

        if score < mi_checking[0]:
            mi_checking[0] = score
            sitk.WriteTransform(bspline_final_transformation, 'best_transform_mi_CE-93044.tfm') #ulozi transformacni funkci,kdyz najde lespi - prepise
        return score

    #Nastavení průběžného ukládání
    def saving(study, trial):
        #Spuštěna při každém pokus, ukládá mejlepší výsledek
        if study.best_trial.number == trial.number:
            with open('prubezne_ukladani_mi_93044.txt', 'w', encoding='utf-8') as f:
                f.write('Pruběžné výsledky optimalizace pomocí mi \n')
                f.write(f'Číslo pokusu: {trial.number} \n')
                f.write(f'Skore (MI): {study.best_value}')
                f.write('Nejlepší parametry (so far) \n')
                for par, value in study.best_params.items():
                    f.write(f' {par}: {value}\n')

    #optuna si to uklada do databaze, pokud to spade - lze nacist posledni vysledky zpet
    study = optuna.create_study(direction='minimize',
                                study_name = 'mi_optimize_CE-93044',
                                storage = 'sqlite:///mi_optuna_CE-93044.db',
                                load_if_exists = True)
    study.optimize(objective_mi, n_trials=200, callbacks=[saving])
    print('Hotovo, konec optimalizace')

if __name__ == "__main__":
    print('Zacatek vypoctu')
    start = time.time()
    optuna_mi()
    minutes, seconds = divmod(time.time() - start, 60)
    hours, minutes = divmod(minutes, 60)

    result = f'Cas vypoctu: {int(hours)} hodin, {int(minutes)}, minut a {int(seconds)} sekund'
    print(result)

    with open('cas_vypoctu_optuny_mi_22159.txt', 'w', encoding='utf-8') as f:
        f.write(result)