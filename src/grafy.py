import numpy as np
import optuna
import optuna.visualization.matplotlib as vis_mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wilcoxon
import os
import seaborn as sns
import matplotlib.ticker as ticker


def graf_optuna(study_name, db_name, suffix, pacient_id, save_path):
    """
    Načte SQLite databázi z Optuny a vygeneruje dva grafy:
    1. Konvergenční křivku (vývoj skóre napříč pokusy).
    2. Důležitost jednotlivých hyperparametrů.

    Parametry:
    ----------
    study_name : str
        Název studie uložené v databázi (např. 'mse_optimize_CE-86807').
    db_name : str
        Cesta k souboru databáze (bez 'sqlite:///').
    method_name : str
        Název metody pro popisek grafu (např. 'B-spline (MI)').
    pacient_id : str
        Identifikátor pacienta (použije se v titulku a názvu souboru).
    save_path : str
        Složka, kam se grafy uloží.
    """
    #Načtení databáze s historii pokusů
    study = optuna.load_study(study_name=study_name, storage=f'sqlite:///{db_name}')

    #Slovník pro přeložení názvů hyperparametrů v grafech
    param_translation = {
        'HistogramBins': 'Počet histogramových košů',
        'MeshSize': 'Velikost mřížky',
        'Iterations': 'Počet iterací',
        'SolutionAccuracy': 'Přesnost řešení',
        'ConvergenceTolerance': 'Konvergenční tolerance',
        'StandardDeviatons': 'Elastické vyhlazení',
        'FieldStandardDeviations': 'Viskózní vyhlazení'
    }

    #Vytvoření složky pokud neexistuje
    os.makedirs(save_path, exist_ok=True)

    #Vytvoření grafu konvergence
    ax1 = vis_mpl.plot_optimization_history(study)
    ax1.set_title('Historie optimalizace (Pacient 2) - B-spline (MI)')
    ax1.set_xlabel('Pokus')
    ax1.set_ylabel('Hodnota účelové funkce')

    #Přeložení legendy
    handles, labels = ax1.get_legend_handles_labels()
    new_labels = ['Hodnota účelové funkce' if l == 'Objective Value' else 'Nejlepší hodnota' if l == 'Best Value' else l
                  for l in labels]
    ax1.legend(handles, new_labels)

    plt.tight_layout()

    #Uložení a zavření grafu, aby se nepřekrývaly
    plt.savefig(os.path.join(save_path, f'graf_konvergence_{pacient_id}_{suffix}.png'), dpi=300)
    plt.close()

    #Vytvoření grafu důležitosti hyperparametrů
    ax2 = vis_mpl.plot_param_importances(study)
    ax2.set_xlabel('Důležitost hyperparametru')
    ax2.set_ylabel('Hyperparametr')

    #Překlad názvů hyperparametrů na ose Y
    yticklabels = ax2.get_yticklabels()
    new_yticklabels = [param_translation.get(label.get_text(), label.get_text()) for label in yticklabels]
    ax2.set_yticks(ax2.get_yticks())  # Fix pro novější verze matplotlib
    ax2.set_yticklabels(new_yticklabels)

    #Úprava legendy
    handles, labels = ax2.get_legend_handles_labels()
    if handles:
        new_labels = ['Hodnota účelové funkce' if l == 'Objective Value' else l for l in labels]
        ax2.legend(handles, new_labels)

    plt.tight_layout()

    #Uložení a zavření grafu
    plt.savefig(os.path.join(save_path, f'graf_dulezitost_{pacient_id}_{suffix}.png'), dpi=300)
    plt.close('all')


def TRE_calculateion(path_refe, path_pocatecni, path_rigid, path_mi, path_mse, path_demons, pacient):
    """
    Vypočítá Target Registration Error (TRE) pro všechny metody a vygeneruje statistiku.

    Funkce načte CSV soubory se souřadnicemi bodů, převede pixelové vzdálenosti na milimetry
    pomocí fyzického rozestupu voxelů (spacing) a spočítá euklidovskou vzdálenost (TRE).
    Vygeneruje textový report se základní statistikou, spojené CSV s výsledky a individuální boxploty.

    Parametry:
    ----------
    path_refe - path_demons : str
        Cesty k CSV souborům s vyznačenými body pro referenci a jednotlivé metody.
    pacient_id : str
        Název pacienta (např. 'Pacient_5') pro dynamické pojmenování výstupů.
    """

    tre_hodnoty = []
    nazvy_metod = []
    textovy_vystup = "-- SOUHRNNÁ STATISTIKA TRE --\n\n"

    #Spacing, potrebny pro prevod z px na mm
    spacing_x = 1.0625
    spacing_y = 1.0625
    spacing_z = 1.4

    #Načtení CSV souboru (přidáno načtení surových dat a rigidní registrace)
    csv_refe = pd.read_csv(path_refe)
    csv_pocatecni = pd.read_csv(path_pocatecni)
    csv_rigid = pd.read_csv(path_rigid)
    csv_mi = pd.read_csv(path_mi)
    csv_mse = pd.read_csv(path_mse)
    csv_demons = pd.read_csv(path_demons)

    #Načtení referenčních souřadnic - posledni 3 sloupce v csv
    x_refe = csv_refe.iloc[:, -1].values
    y_refe = csv_refe.iloc[:, -2].values
    z_refe = csv_refe.iloc[:, -3].values

    #Násobek pro vynásobení referenčních bodů na požadovaný počet
    nasobek = len(csv_mi) // len(csv_refe)
    x_refe_60 = np.repeat(x_refe, nasobek)
    y_refe_60 = np.repeat(y_refe, nasobek)
    z_refe_60 = np.repeat(z_refe, nasobek)

    #Příprava tabulky
    vysledky = pd.DataFrame({'Index bodu': range(1, len(csv_mi) + 1)})

    #Vytvoření slovníku pro zpracování ve smyčce (kompletních 5 metod)
    metody = {
        'Pocatecni': csv_pocatecni,
        'Rigid': csv_rigid,
        'MI': csv_mi,
        'MSE': csv_mse,
        'Demons': csv_demons
    }

    #Výpočet TRE pro každou metriku
    for nazev, metod in metody.items():
        x_metric = metod.iloc[:, -1].values
        y_metric = metod.iloc[:, -2].values
        z_metric = metod.iloc[:, -3].values

        #Výpočet rozdílu oproti referenčnímu a převod na mm, pomocí spacingu
        dx = (x_refe_60 - x_metric) * spacing_x
        dy = (y_refe_60 - y_metric) * spacing_y
        dz = (z_refe_60 - z_metric) * spacing_z

        #Výpočet euklidovské vzdálenosti - TRE
        tre = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        vysledky[f'TRE_{nazev} (mm)'] = tre

        tre_hodnoty.append(tre)
        nazvy_metod.append(nazev)

        #Výpis výsledků do konzole
        stat_blok = (
            f"Metoda: {nazev}\n"
            f"-----------------------------------\n"
            f"Průměrné TRE:        {np.mean(tre):.3f} mm\n"
            f"Rozptyl (Variance):  {np.var(tre):.3f} mm^2\n"
            f"Směrodatná odchylka: {np.std(tre):.3f} mm\n"
            f"Minimum:             {np.min(tre):.3f} mm\n"
            f"1. kvartil (Q1):     {np.percentile(tre, 25):.3f} mm\n"
            f"Medián (Q2):         {np.median(tre):.3f} mm\n"
            f"3. kvartil (Q3):     {np.percentile(tre, 75):.3f} mm\n"
            f"Maximum:             {np.max(tre):.3f} mm\n\n"
        )

        textovy_vystup = textovy_vystup + stat_blok
        print(stat_blok)

    #Uložení všech vypočítaných TRE do jednoho souboru pro daného pacienta
    #Oddělovač středník zajistí, že si s tím nový boxplot skript poradí
    nazev_csv = f"TRE_vysledky_{pacient}.csv"
    vysledky.to_csv(nazev_csv, sep=';', index=False)
    print(f"Výsledky úspěšně uloženy do souboru: {nazev_csv}")

    #Vykreslování individuálních boxplotů
    box_style = dict(patch_artist=True, boxprops=dict(facecolor='lightblue', color='black'),
                     medianprops=dict(color='red', linewidth=2),
                     flierprops=dict(marker='o', color='red', alpha=0.5))

    #Smyčka pro uložení boxplotů (individuálně)
    for i in range(len(nazvy_metod)):
        metoda = nazvy_metod[i]

        nazev_pro_graf = metoda
        if metoda == "Pocatecni": nazev_pro_graf = "Před registrací"
        if metoda == "Rigid": nazev_pro_graf = "Rigidní"
        if metoda == "MI": nazev_pro_graf = "B-spline MI"
        if metoda == "MSE": nazev_pro_graf = "B-spline MSE"

        data = tre_hodnoty[i]

        plt.figure(figsize=(6, 6))
        plt.boxplot([data], tick_labels=[nazev_pro_graf], **box_style)

        plt.title(f'Přesnost registrace: {nazev_pro_graf} ({pacient})', fontsize=14)
        plt.ylabel('Cílová chyba registrace - TRE (mm)', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        nazev_souboru = f'TRE_boxplot_{metoda}_{pacient}.png'
        plt.savefig(nazev_souboru, dpi=300)
        plt.close()

    vysledky.to_csv('TRE_pacient_5_final.csv', index=False, sep=';')
    with open("TRE_statistika_souhrn_pacient_5_final.txt", "w", encoding="utf-8") as f:
        f.write(textovy_vystup)


def vytvor_tre_boxplot(mapovani_souboru, vystupni_soubor="TRE_sdruzeny_boxplot.png"):
    """
    Vygeneruje sdružený boxplot porovnávající cílovou chybu registrace (TRE)
    napříč pacienty a zvolenými registračními metodami.

    Graf využívá techniku přerušené osy Y (broken axis), čímž je vizuálně
    rozdělen do dvou podgrafů. Horní část slouží k zobrazení extrémních
    odchylek (outlierů), zatímco dolní, zvětšená část ukazuje reálné rozložení
    většiny naměřených hodnot. Úpravou ax_top a ax_bottom lze změnit rozsahy os.

    Parametry:
    ----------
    mapovani_souboru : dict
        Slovník mapující název pacienta na cestu k příslušnému CSV souboru s výsledky TRE.
    vystupni_soubor : str, volitelné
        Název a cesta k výstupnímu obrázku (výchozí: "TRE_sdruzeny_boxplot.png").
    """
    sloupce_metod = ["TRE_Pocatecni (mm)", "TRE_Rigid (mm)", "TRE_MI (mm)", "TRE_MSE (mm)", "TRE_Demons (mm)"]
    hezke_nazvy = {
        "TRE_Pocatecni (mm)": "Před registrací",
        "TRE_Rigid (mm)": "Rigidní",
        "TRE_MI (mm)": "B-spline (MI)",
        "TRE_MSE (mm)": "B-spline (MSE)",
        "TRE_Demons (mm)": "Demons"
    }

    data_vsech_pacientu = []

    for pacient, soubor in mapovani_souboru.items():
        if not os.path.exists(soubor):
            continue

        df = pd.read_csv(soubor, sep=';')
        df_melted = df.melt(value_vars=sloupce_metod, var_name="Metoda", value_name="TRE")
        df_melted["Metoda"] = df_melted["Metoda"].map(hezke_nazvy)
        df_melted["Pacient"] = pacient
        data_vsech_pacientu.append(df_melted)

    df_komplet = pd.concat(data_vsech_pacientu, ignore_index=True)

    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 14})

    #Vytvoření dvou podgrafů pod sebou (horní pro extrémy, dolní pro detaily)
    #Parametr height_ratios určuje, že dolní část bude 3x vyšší než horní
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(12, 7), gridspec_kw={'height_ratios': [1, 3]}
    )

    fig.subplots_adjust(hspace=0.05) # Minimální mezera mezi grafy

    #Vykreslení úplně stejného grafu do obou podgrafů, lišit se bude jen přiblížení osy Y
    sns.boxplot(data=df_komplet, x="Pacient", y="TRE", hue="Metoda", palette="colorblind", width=0.7, fliersize=3,
    ax=ax_top)
    sns.boxplot(data=df_komplet, x="Pacient", y="TRE", hue="Metoda", palette="colorblind", width=0.7, fliersize=3,
    ax=ax_bottom)

    #Nastavení ořezu, horní část pro extrémy (4, 65), dolní pro standardní výsledky
    ax_top.set_ylim(4, 65)
    ax_bottom.set_ylim(-0.1, 3)

    #Odstranění přebytečných okrajů, aby to vypadalo jako jeden graf
    ax_top.spines['bottom'].set_visible(False)
    ax_bottom.spines['top'].set_visible(False)

    ax_top.xaxis.tick_top()

    #Skryje popisky osy X nahoře
    ax_top.tick_params(labeltop=False)
    ax_bottom.xaxis.tick_bottom()

    #Vykreslení diagonálních čárek (//) naznačujících přerušení osy
    #Velikost čárek
    d = .015
    kwargs = dict(transform=ax_top.transAxes, color='gray', clip_on=False, linewidth=1.5)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    # Společný název přes celou figuru
    fig.suptitle("Rozložení cílové chyby registrace (TRE) u jednotlivých pacientů", fontsize=16, y=0.95)

    #Odstranění automatických popisků z dílčích grafů
    ax_top.set_xlabel("")
    ax_top.set_ylabel("")
    ax_bottom.set_xlabel("Pacienti s optimalizovanými hyperparametry", labelpad=15)
    ax_bottom.set_ylabel("")

    #Vytvoření jednoho velkého společného popisku osy Y
    fig.text(0.06, 0.5, 'TRE [mm]', va='center', rotation='vertical', fontsize=14)

    #Legenda bude jen v horním grafu
    ax_bottom.get_legend().remove()
    ax_top.legend(title="Stav / Registrační metoda", frameon=True, loc="upper right")

    #Uložení a zobrazení
    plt.savefig(vystupni_soubor, dpi=300, bbox_inches="tight")
    print(f"Graf s přerušenou osou byl úspěšně uložen jako: {vystupni_soubor}")
    plt.show()
def vykresli_mrizku_pro_metriku(df, metrika, y_label, nazev_grafu, vystupni_soubor):
    """
    Pomocná funkce pro vykreslení 2x2 mřížky grafů pro jednu vybranou metriku
    (např. MI nebo MSE) napříč všemi pacienty.

    Parametry:
    ----------
    df : pandas.DataFrame
        Datový rámec (Long Format) obsahující sloupce 'Cas', 'Metoda', 'Pacient' a hodnocenou metriku.
    metrika : str
        Název sloupce s metrikou, která se bude vykreslovat (např. 'MI' nebo 'MSE').
    y_label : str
        Popisek osy Y (např. 'MI skóre').
    nazev_grafu : str
        Hlavní nadpis celé vizualizace.
    vystupni_soubor : str
        Cesta k výstupnímu obrázku.
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 12})

    #Vytvoření plátna: 2 řádky, 2 sloupce se stejnou osou x
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()  # Zploštění pole os pro snazší iteraci

    #Definice palety barev (použití palety tab10 pro lepší odlišení až 10 pacientů)
    paleta_pacientu = sns.color_palette("tab10", 7)
    metody = ['Rigidní registrace', 'B-spline (MI)', 'B-spline (MSE)', 'Algoritmus Demons']

    for i, metoda in enumerate(metody):
        data_metody = df[df['Metoda'] == metoda]

        # Vykreslení vývoje v čase pro konkrétní metodu
        sns.lineplot(
            data=data_metody, x='Cas', y=metrika, hue='Pacient',
            palette=paleta_pacientu, marker='o', linewidth=2, markersize=6,
            ax=axes[i]
        )
        axes[i].set_title(f"{metoda}", fontsize=14, fontweight='bold')
        axes[i].set_ylabel(y_label)

        #Skrytí automaticky generovaných legend u každého podgrafu
        if axes[i].get_legend() is not None:
            axes[i].get_legend().remove()

        #Přidání popisku osy X pouze na spodní řádek mřížky
        if i >= 2:
            axes[i].set_xlabel('Časový okamžik')
        else:
            axes[i].set_xlabel('')

    #Tvorba jedné společné legendy
    handles, labels = axes[0].get_legend_handles_labels()

    #Umístění společné legendy dolů pod grafy (zaloemní do řádků podle počtu pacientů)
    fig.legend(handles, labels, title='Pacienti', loc='lower center',
               ncol=7, bbox_to_anchor=(0.5, -0.05), fontsize=13, title_fontsize=14)

    fig.suptitle(nazev_grafu, fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig(vystupni_soubor, dpi=300, bbox_inches="tight")
    print(f"Graf byl úspěšně uložen do: {vystupni_soubor}")
    plt.close()  # Zavření plátna, aby se nepřekrývalo s dalším grafem


def zpracuj_a_vykresli_data(slozka_s_daty, suffix='Testovaci'):
    """
    Načte průběžné výsledky metrik (MI, MSE) pro zadané pacienty a metody,
    sloučí je a vygeneruje sdružené liniové grafy (mřížky 2x2) vývoje metrik v čase.

    Parametry:
    ----------
    slozka_s_daty : str
        Cesta ke složce obsahující vygenerovaná CSV s průběhem metrik.
    suffix : str, volitelné
        Přípona pro názvy výstupních obrázků (výchozí je "Testovaci").
    """
    #Definice mapování ID pacientů pomocí slovníku
    #Pro přidání pacienta je potřeba přidat 'Nazev pacienta': 'tak jak se zobrazí v grafu'
    pacienti_mapa = {
        'Testovaci_Pacient': 'Testovací pacient'
    }

    metody_kody = {
        'rigid': 'Rigidní registrace',
        'mi': 'B-spline (MI)',
        'mse': 'B-spline (MSE)',
        'demons': 'Algoritmus Demons'
    }

    data = []

    #Načtení dat z csv souborů
    for pac_id, pac_label in pacienti_mapa.items():
        for kod_metody, nazev_metody in metody_kody.items():
            soubor = f"vysledky_metrik_{kod_metody}_pacient_{pac_id}.csv"
            cesta = os.path.join(slozka_s_daty, soubor)

            if not os.path.exists(cesta):
                continue

            try:
                df_temp = pd.read_csv(cesta, sep=';')
                for _, row in df_temp.iterrows():
                    #Čištění textového formátu času na celočíselnou hodnotu (např. 'T_05' -> 5)
                    cas_str = str(row['Časový okamžik'])
                    cas_int = int(cas_str.replace('T_', ''))

                    data.append({
                        'Cas': cas_int,
                        'Pacient': pac_label,
                        'Metoda': nazev_metody,
                        'MI': float(row['MI_skóre']),
                        'MSE': float(row['MSE_skóre'])
                    })
            except Exception as e:
                print(f"Chyba u {soubor}: {e}")

    df = pd.DataFrame(data)

    if df.empty:
        print("Nepodařilo se načíst žádná data.")
        return

    #Vygenerování dvou samostatných obrázků
    vystup_mi = os.path.join(slozka_s_daty, f"Graf_Vyvoj_MI_2x2_{suffix}.png")
    vykresli_mrizku_pro_metriku(df, metrika='MI', y_label='MI skóre',
                                nazev_grafu='Vývoj metriky vzájemné informace (MI) v čase - univerzální hyperparametry',
                                vystupni_soubor=vystup_mi)

    vystup_mse = os.path.join(slozka_s_daty, f"Graf_Vyvoj_MSE_2x2_{suffix}.png")
    vykresli_mrizku_pro_metriku(df, metrika='MSE', y_label='MSE skóre',
                                nazev_grafu='Vývoj střední kvadratické chyby (MSE) v čase - univerzální hyperparametry',
                                vystupni_soubor=vystup_mse)


def wilcoxon_test(cesta_optuna, cesta_uni):
    """
        Načte dva CSV soubory s výsledky TRE a provede oboustranný párový Wilcoxonův test.

        Funkce statisticky porovnává hodnoty chyb registrace (TRE) získané pomocí
        optimalizovaných hyperparametrů s výsledky získanými při použití univerzálního
        nastavení. Výsledné p-hodnoty a zhodnocení statistické významnosti (pro hladinu
        významnosti alfa = 0.05) se vypisují do standardního výstupu.

        Parametry:
        ----------
        cesta_optuna : str
            Cesta k CSV souboru s TRE výsledky po optimalizaci.
        cesta_uni : str
            Cesta k CSV souboru s TRE výsledky při univerzálním nastavení.
        """
    #Rozdělení sloupců (tenhle formát musí být v .csv)
    metody = ['TRE_MI (mm)', 'TRE_MSE (mm)', 'TRE_Demons (mm)']

    #Načtení výsledků
    try:
        df_optuna = pd.read_csv(cesta_optuna, sep=';')
        df_uni = pd.read_csv(cesta_uni, sep=';')
    except FileNotFoundError as e:
        print(f"Chyba při načítání souboru: {e}")
        return

    print("- VÝSLEDKY WILCOXONOVA PÁROVÉHO TESTU --")

    for metoda in metody:
        # Kontrola, zda sloupec v datech opravdu existuje
        if metoda not in df_optuna.columns or metoda not in df_uni.columns:
            print(f"Upozornění: Sloupec '{metoda}' nebyl nalezen. Přeskakuji...\n")
            continue

        data_optuna = df_optuna[metoda]
        data_uni = df_uni[metoda]

        # Výpočet Wilcoxonova testu
        # alternative='two-sided' zjišťuje, zda je jakýkoliv rozdíl (zlepšení i zhoršení)
        stat, p_value = wilcoxon(data_optuna, data_uni, alternative='two-sided')

        print(f"Metoda: {metoda}")
        print(f"p-hodnota = {p_value:.5f}")

        if p_value < 0.05:
            print("- ROZDÍL JE STATISTICKY VÝZNAMNÝ (p < 0.05)\n")
        else:
            print("- ROZDÍL NENÍ STATISTICKY VÝZNAMNÝ (p >= 0.05)\n")
