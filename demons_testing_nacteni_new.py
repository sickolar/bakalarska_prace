import optuna
import SimpleITK as sitk
import numpy as np
import Nacteni_image
import time
def optuna_demons():
    # Nacteni pomoci Nacteni.py
    data_base = Nacteni_image.data_load(
        r'C:\Users\andre\OneDrive\Plocha\School\BTB_3\Bakalářská práce\subjekty\pokus_1\manifest-data1407430404196\QIN Breast DCE-MRI\QIN-Breast-DCE-MRI-BC01\04-24-1996-NA-Breast CE-59716\08.000000-twist20sdynTRA h20 ex B17TT69.7s-92875')
    data_chang = Nacteni_image.data_load(
        r'C:\Users\andre\OneDrive\Plocha\School\BTB_3\Bakalářská práce\subjekty\pokus_1\manifest-data1407430404196\QIN Breast DCE-MRI\QIN-Breast-DCE-MRI-BC01\04-24-1996-NA-Breast CE-59716\25.000000-twist20sdynTRA h20 ex B17TT231.0s-01775')
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
    #ulozeni rigidni registrace
    sitk.WriteTransform(final_transform, 'rigid_transform_demons_CE-59716.tfm')

    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0,
                                     moving_image.GetPixelID())

    fixed_float = sitk.Cast(fixed_image, sitk.sitkFloat32)  # prevod z int na float kvuli vypoctum.
    moving_float = sitk.Cast(moving_resampled, sitk.sitkFloat32)  # prevod z int na float kvuli vypoctum.

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

    fixed_float_masked = sitk.Mask(fixed_float, fixed_maska_image)
    moving_float_masked = sitk.Mask(moving_float, moving_maska_image)

    # Snaha o matching rozložení barev mezi moving a fixed
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(1024)  # Rozdeleni na 1024 odstinu
    matcher.SetNumberOfMatchPoints(7)  # pocet bodu křivky, ktere se musi střetnout
    matcher.ThresholdAtMeanIntensityOn()  # Ignoruje černé pozadí - bere průměrný jas

    moving_matched = matcher.Execute(moving_float_masked, fixed_float_masked)

    evaluator = sitk.ImageRegistrationMethod()
    evaluator.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    evaluator.SetMetricFixedMask(fixed_maska_image) #pocita metriku z te masky, fixed bo po zarovnani se podoba spis fixed

    #Udrzeni nelepsiho skore, hledame minimuj (zaporne maximum)
    #Pote se ulozi transformacni funkce podle tohoto skore - osetreni random 5%
    mi_checking = [float('inf')] #nastaveni na infinity
    def objective_demons(trial):
        update_standard = trial.suggest_float('StandardDeviations', 1, 3.5)
        update_field_standard = trial.suggest_float('FieldStandardDeviations', 0.5, 3.5)
        update_iterations = trial.suggest_int('Iterations', 60, 800, step = 20)
        demons = sitk.DiffeomorphicDemonsRegistrationFilter()  # Ošetření toho aby se nepřeložila tkáň přes sebe
        demons.SetNumberOfIterations(update_iterations)  # Počet iterací
        demons.SetStandardDeviations(update_standard)  # Vyhlazuje deformační pole tady na 1.5
        demons.SetUpdateFieldStandardDeviations(update_field_standard)  # Aby nešly sousedním pixely opačným směrem

        demons_field = demons.Execute(fixed_float_masked, moving_matched)  # Výpočet vektorového pole pohybu
        final_demons = sitk.DisplacementFieldTransform(demons_field)  # Převedení na objekt final demons
        demons_resampled = sitk.Resample(moving_resampled, fixed_image, final_demons, sitk.sitkBSpline, 0.0,
                                         moving_image.GetPixelID())  # Aplikace transformace

        score = evaluator.MetricEvaluate(fixed_float, sitk.Cast(demons_resampled, sitk.sitkFloat32))

        if score < mi_checking[0]:
            mi_checking[0] = score
            sitk.WriteTransform(final_demons, 'best_transform_demons_CE-59716.tfm') #ulozi transformacni funkci,kdyz najde lespi - prepise
        return score


    #Nastavení průběžného ukládání
    def saving(study, trial):
        #Spuštěna při každém pokus, ukládá mejlepší výsledek
        if study.best_trial.number == trial.number:
            with open('prubezne_ukladani_demons.txt', 'w', encoding='utf-8') as f:
                f.write('Pruběžné výsledky optimalizace pomocí demons \n')
                f.write(f'Číslo pokusu: {trial.number} \n')
                f.write(f'Skore (MI): {study.best_value}')
                f.write('Nejlepší parametry (so far) \n')
                for par, value in study.best_params.items():
                    f.write(f' {par}: {value}\n')

    #optuna si to uklada do databaze, pokud to spade - lze nacist posledni vysledky zpet
    study = optuna.create_study(direction='minimize',
                                study_name = 'demons_optimize_CE-59716',
                                storage = 'sqlite:///demons_optuna_CE-59716.db',
                                load_if_exists = True)
    study.optimize(objective_demons, n_trials=200, callbacks=[saving])
    print('Hotovo, konec optimalizace')

if __name__ == "__main__":
    print('Zacatek vypoctu')
    start = time.time()
    optuna_demons()

    minutes, seconds = divmod(time.time() - start, 60)
    hours, minutes = divmod(minutes, 60)

    result = f'Cas vypoctu: {int(hours)} hodin, {int(minutes)}, minut a {int(seconds)} sekund'
    print(result)

    with open('cas_vypoctu_optuny_demons_22159.txt', 'w', encoding='utf-8') as f:
        f.write(result)