import sys
import time

#Přidání složky src do systémové cesty (pro import funkcí)
sys.path.append('src')

import zobrazeni_4d
import registrace
import testovani_hyperparametru
import niftii_load
import grafy

if __name__ == '__main__':
    #Před spuštěním skriptu je nutné nastavit lokální cesty k testovacím datům.

    #data_dcm: Absolutní cesta ke složce pacienta, která obsahuje chronologicky řazené podsložky
    #data_save: Adresář, do kterého budou uloženy výsledné 4D NIfTI objemy (.nii.gz) a doprovodné CSV tabulky metrik.
    #pacient_id: Identifikátor pacienta vložený do názvů výstupních souborů.

    data_dcm = r'Vase_cesta_k_datum'
    data_save = r'Vase_cesta_k_adresari_pro_ulozeni'
    pacient_id = 'Testovaci_Pacient'

    #optuna_fixed: Absolutní cesta ke složce k T_00 složce pacienta, která obsahuje jednotlivé 2D řezy prvního času
    #optuna_moving: Absolutní cesta ke složce k T_N složce pacienta, která obsahuje jednotlivé 2D řezy n-tého času
    #Tento moving čas je doporučen zvolit v fázi největšího nasycení kontrastní látkou.
    #Tuto fázi lze emepricky vyhodnotit pomocí funkce zobrazeni_dcm_4D

    optuna_fixed = r'Vase_cesta_k_fixed_image_objemu'
    optuna_moving = r'Vase_cesta_k_moving_image_objemu'

    start = time.time()

    # ------------------------------------------------------------------------------------------

    #Vizualizace raw dicom dat v Napari
    #zobrazeni_4d.zobrazeni_dcm_4d(data_dcm)

    # ------------------------------------------------------------------------------------------

    #Hledání optimálních hyperparametrů pomocí frameworku Optuna, lze upravit počet iterací
    #testovani_hyperparametru.optuna_mse(optuna_fixed, optuna_moving, data_save, pacient_id, n_trials=200)
    #testovani_hyperparametru.optuna_mi(optuna_fixed, optuna_moving, data_save, pacient_id, n_trials=200)
    #testovani_hyperparametru.optuna_demons(optuna_fixed, optuna_moving, data_save, pacient_id, n_trials=200)

    # ------------------------------------------------------------------------------------------

    #Registrace do 4D objemu, který uloží jako .nii.gz. Také se uloží statistika pro průběžnou metriku MSE a MI jako .csv.
    #Hyperparamerty jsou defaultně nastaveny na univerzální hodnoty.
    #Pro aplikaci optimálních hyperparametrů je potřeba změnit manuálně hyperpary jednotlivých registračních funkcí!!

    #registrace.registration_full_rigid(data_dcm, data_save, pacient_id)
    #registrace.registration_full_bspline_mse(data_dcm, data_save, pacient_id)
    #registrace.registration_full_bspline_mi(data_dcm, data_save, pacient_id)
    #registrace.registration_full_demons(data_dcm, data_save, pacient_id)

    # ------------------------------------------------------------------------------------------

    #Vizualizace výsledků
    #niftii_load.niftii_view(rigid_file=fr'{data_save}\registered_4D_rigid_{pacient_id}.nii.gz', mse_file = fr'{data_save}\registered_4D_mse_{pacient_id}.nii.gz', mi_file = fr'{data_save}\registered_4D_mi_{pacient_id}.nii.gz', demons_file = fr'{data_save}\registered_4D_demons_{pacient_id}.nii.gz')

    # ------------------------------------------------------------------------------------------

    #Vykreslení grafů a výpočet statistiky

    #Optuna grafy - konvergence a důležitost hyperparametrů, změnit mse na [mi, demons] dle potřeby
    #grafy.graf_optuna(study_name = f'mse_optimize_{pacient_id}', db_name = fr'{data_save}\mse_optuna_{pacient_id}.db', suffix = 'MSE', pacient_id = pacient_id, save_path = data_save)

    #Vývoj MSE a MI metrik v čase z csv tabulek
    #Funguje pouze pro pacient_id = 'Testovaci_Pacient', pro přidání dalšího pacienta
    #je potřeba upravit slovník ve funkci.
    #grafy.zpracuj_a_vykresli_data(data_save, suffix=pacient_id)

    #Výpočet TRE a individuální boxploty (vyžaduje csv s body)
    #grafy.TRE_calculateion(path_refe = fr'{data_save}\body_refe.csv', path_pocatecni = fr'{data_save}\body_pocatecni.csv', path_rigid = fr'{data_save}\body_rigid.csv', path_mi = fr'{data_save}\body_mi.csv', path_mse = fr'{data_save}\body_mse.csv', path_demons = fr'{data_save}\body_demons.csv', pacient = pacient_id)

    #Sdružený TRE boxplot s přerušenou osou y
    #mapovani_souboru = {pacient_id: fr'{data_save}\TRE_vysledky_{pacient_id}.csv'}
    #grafy.vytvor_tre_boxplot(mapovani_souboru=mapovani_souboru, vystupni_soubor=fr'{data_save}\TRE_sdruzeny_boxplot.png')

    #Wilcoxonův párový test
    #grafy.wilcoxon_test(cesta_optuna=fr'{data_save}\TRE_vysledky_optuna.csv', cesta_uni=fr'{data_save}\TRE_vysledky_uni.csv')

    # ------------------------------------------------------------------------------------------
    #Vypis času
    minutes, seconds = divmod(time.time() - start, 60)
    hours, minutes = divmod(minutes, 60)
    result = f'Cas vypoctu: {int(hours)} hodin, {int(minutes)}, minut a {int(seconds)} sekund'
    print('result')
