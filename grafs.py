import numpy as np
import optuna
import optuna.visualization.matplotlib as vis_mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy
def graf_create(study_name, db_name, suffix):

    #Nacteni z databaze
    study = optuna.load_study(study_name=study_name, storage = f'sqlite:///{db_name}')

    #Graf konvergence
    vis_mpl.plot_optimization_history(study)
    #plt.title(f'Historie optimalizace - {study_name}')
    plt.tight_layout()
    plt.savefig(f'graf_konvergence_18265_{suffix}.png', dpi=300)

    print('neco 1')
    # Dulezitost hyperparametru
    vis_mpl.plot_param_importances(study)
    #plt.title(f'Důležitost hyperparametrů - {study_name}')
    plt.tight_layout()
    plt.savefig(f'graf_dulezitost_18265_{suffix}.png', dpi=300)

    print('neco 2')
    # Vztahy parametru
    vis_mpl.plot_parallel_coordinate(study)
    #plt.title(f'Vztahy parametrů - {study_name}')
    plt.tight_layout()
    plt.savefig(f'graf_paralelni_18265_{suffix}.png', dpi=300)

    print('neco 3')
    plt.close('all')

def TRE_calculateion(path_refe, path_rigid, path_mi, path_mse, path_demons):

    #Spacing, potrebny pro prevod z px na mm
    spacing_x = 1.0625
    spacing_y = 1.0625
    spacing_z = 1.4

    #Nacteni CSV souboru
    csv_refe = pd.read_csv(path_refe)
    csv_rigid = pd.read_csv(path_rigid)
    csv_mi = pd.read_csv(path_mi)
    csv_mse = pd.read_csv(path_mse)
    csv_demons = pd.read_csv(path_demons)

    #Nacteni referencnich souradnic - posledni 3 slouopce v csv
    x_refe = csv_refe.iloc[:, -1]
    y_refe = csv_refe.iloc[:, -2]
    z_refe = csv_refe.iloc[:, -3]

    #Priprava tabulky
    vysledky = pd.DataFrame({'Index bodu': range(1, len(csv_refe) + 1)})

    #Vytvoreni slovniku pro zpracovani ve smycce
    metody = {
        'Rigid': csv_rigid,
        'MI': csv_mi,
        'MSE': csv_mse,
        'Demons': csv_demons
    }

    #Vypocet TRE pro kazdou metodu
    for nazev, metod in metody.items():
        x_metric = metod.iloc[:,-1]
        y_metric = metod.iloc[:,-2]
        z_metric = metod.iloc[:,-3]

        #Vypocet rozdilu vuci referencimu a prevod na mm - pomoci spacingu
        dx = (x_refe - x_metric) * spacing_x
        dy = (y_refe - y_metric) * spacing_y
        dz = (z_refe - z_metric) * spacing_z

        #Vypocet euklidovske vzdalenosti - tre
        tre = np.sqrt(dx**2 + dy**2 + dz**2)
        vysledky[f'TRE_{nazev} (mm)'] = tre

        #vypis vysledku
        print(vysledky.to_string(index=False))

        vysledky.to_csv('testovaci_59716.csv', index=False, sep=';')

if __name__ == '__main__':
   # read_path = r'C:\Bakalarka\Nifty_files\databaze_optuna\demons_optuna_CE-18265.db'
    #graf_create(study_name= 'demons_optimize_CE-18265', db_name= read_path, suffix='demons')
    TRE_calculateion(r'C:\Bakalarka\Nifty_files\TRE_body\T0_59716.csv', r'C:\Bakalarka\Nifty_files\TRE_body\rigid_59716.csv',
                     r'C:\Bakalarka\Nifty_files\TRE_body\MI_59716.csv', r'C:\Bakalarka\Nifty_files\TRE_body\MSE_59716.csv', r'C:\Bakalarka\Nifty_files\TRE_body\Demons_59716.csv')