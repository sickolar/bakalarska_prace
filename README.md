# 4D DCE-MRI: Nelineární registrace obrazů prsu

Tento repozitář obsahuje zdrojové kódy k bakalářské práci zaměřené na zpracování a registraci dynamických (4D) MRI snímků prsu s kontrastní látkou (DCE-MRI).

Během vyšetření dochází vlivem dýchání a přirozeného pohybu pacienta k deformacím tkáně, které ztěžují diagnostiku. Cílem implementovaných metod je automatická kompenzace tohoto pohybu pomocí registračních algoritmů a následná kvantifikace jejich úspěšnosti.

Projekt byl vyvíjen a primárně testován na standardizovaném veřejném datasetu [QIN-Breast-DCE-MRI](https://www.cancerimagingarchive.net/collection/qin-breast-dce-mri/) z databáze The Cancer Imaging Archive (TCIA)

## Implementované funkce

Projekt pokrývá následující kroky zpracování obrazových dat:

1. **Předzpracování dat:** Načítání DICOM souborů při zachování fyzikálních metadat (rozteč voxelů, počátek souřadnic).
2. **Globální zarovnání:** Počáteční rigidní registrace obrazů (Euler3DTransform).
3. **Optimalizace hyperparametrů:** Využití frameworku Optuna k systematickému hledání optimální hustoty mřížky, počtu iterací a přesnosti konvergence na maskované oblasti prsu.
4. **Nelineární registrace:**
   - B-spline využívající podobnostní metriku MSE (Mean Square Error).
   - B-spline využívající podobnostní metriku MI (Mutual Information).
   - Diffeomorfní algoritmus Demons (s předzpracováním přes Histogram Matching).
5. **Kvantitativní vyhodnocení:** Výpočet cílové chyby registrace (TRE – Target Registration Error) v milimetrech a statistické srovnání metod pomocí Wilcoxonova párového testu.

## Struktura repozitáře

Zdrojové kódy jsou v souladu s doporučenou strukturou Python projektů uloženy ve složce `src/`. Spouštění probíhá z kořenového adresáře přes rozcestník `main.py`.

```text
.
├── main.py
├── requirements.txt
└── src/
    ├── registrace.py
    ├── testovani_hyperparametru.py
    ├── grafy.py
    ├── Nacteni_image.py
    ├── zobrazeni_4d.py
    └── niftii_load.py
```

- `main.py` – Hlavní skript. Importuje funkce z jednotlivých modulů a zajišťuje spuštění pipeline z jednoho místa.
- `src/registrace.py` – Obsahuje sjednocené funkce pro všechny čtyři registrační přístupy a ukládá zregistrované 4D objemy do formátu NIfTI. Ukládá také vývoj metrik MSE a MI (`.csv`).
- `src/testovani_hyperparametru.py` – Skript využívající knihovnu Optuna pro prohledávání prostoru deformačních parametrů.
- `src/grafy.py` – Statistické vyhodnocení. Generuje konvergenční křivky, sdružené boxploty a provádí Wilcoxonův test.
- `src/Nacteni_image.py` – Skript pro načítání DICOM dat bez porušení prostorových metadat.
- `src/zobrazeni_4d.py` – Nástroj pro vizualizaci 4D DICOM dat v prostředí Napari.
- `src/niftii_load.py` – Skript pro zobrazení výsledného NIfTI obrazu (včetně fúze).

## Požadovaná struktura vstupních dat

Pro správný chod skriptů musí být vstupní data organizována v originální struktuře DICOM exportů. Je pouze potřeba přejmenovat soubory menší než 10 (např 8.00 na 08.00).

```text
Cesta_k_adresari_pacienta/
├── 08.000000-twist20sdynTRA.../   (T_00: referenční předkontrastní snímek)
│   ├── 1-001.dcm
│   └── ...
├── 11.000000-twist20sdynTRA.../   (T_01: první postkontrastní snímek)
│   ├── 1-001.dcm
│   └── ...
├── 13.000000-twist20sdynTRA.../   (T_02: druhý postkontrastní snímek)
└── ...
```

## Instalace a konfigurace prostředí

Projekt byl vyvíjen a testován v prostředí **Python 3.12.1**. Pro zajištění stability knihoven využijte izolované virtuální prostředí.

### 1. Stažení repozitáře

```bash
git clone https://github.com/sickolar/bakalarska_prace.git
cd bakalarska_prace
```

### 2. Instalace požadovaných knihoven

#### Varianta A – standardní instalace (pip)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

#### Varianta B – zrychlená instalace (uv)

```bash
uv venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

uv pip install -r requirements.txt
```

## Postup spuštění a obsluha pipeline

Veškerá obsluha projektu probíhá přes úpravu a spouštění souboru **main.py**.

1. **Nastavení cest:** Otevřete `main.py` a upravte zástupné cesty k vašim lokálním datům (`data_dcm`, `data_save`).
2. **Spuštění funkcí:** Moduly v `main.py` jsou logicky rozděleny do bloků. Vždy odkomentujte pouze ten řádek, který chcete aktuálně vykonat.
3. **Spuštění v terminálu:**

```bash
python main.py
```

## Důležitá upozornění pro evaluaci

- **Ruční přepis hyperparametrů:** Všechny funkce v souboru `src/registrace.py` jsou ve výchozím stavu nastaveny na univerzální hyperparametry. Pokud chcete aplikovat optimální hyperparametry získané z běhu Optuny, musíte tyto hodnoty ručně přepsat uvnitř příslušných registračních funkcí.
- **Přizpůsobení grafů a statistik:** Při volání analytických a vizualizačních funkcí z `grafy.py` (v souboru `main.py`) je nutné ručně upravit vstupní argumenty podle toho, jakou metodu vyhodnocujete. Nezapomeňte správně přepisovat suffixy (např. 'MSE' na 'MI' nebo 'Demons'), názvy Optuna databází a cesty k cílovým CSV tabulkám.
- **Doba běhu:** Nelineární registrace 4D obrazů je výpočetně náročný proces. Zpracování jednoho pacienta může v závislosti na hardwaru trvat i několik hodin.
- **Vizualizace** Soubor `src/niftii_load.py` obsahuje několik předpřipravených režimů zobrazení v Napari (barevné překryvy, šedotónové zobrazení, TRE body, zobrazení v mřížce). Před spuštěním funkce `niftii_view()` je vhodné upravit zakomentované/odkomentované řádky podle požadovaného způsobu vizualizace.
- **Změna identifikátoru pacienta (grafy):** Funkce pro hromadné vykreslování metrik (`zpracuj_a_vykresli_data`) využívá vnitřní slovník pro čisté formátování legendy. Skript je ve výchozím stavu připraven pro `pacient_id = 'Testovaci_Pacient'`. Pokud se rozhodnete analyzovat jiná data a spustíte skript na jakémkoliv **jiném ID pacienta**, nezapomeňte toto nové ID ručně přidat do slovníku `pacienti_mapa` uvnitř zdrojového kódu `src/grafy.py`. Bez této úpravy kód sice bez chyby proběhne, ale grafy pro nového pacienta se nevykreslí.
## Prohlášení o využití umělé inteligence (AI)

Při vývoji tohoto projektu byly využity nástroje na bázi velkých jazykových modelů (LLM). Umělá inteligence sloužila výhradně jako programátorský asistent. Konkrétně byla využita pro:
* Nápovědu při syntaxi a formátování zdrojového kódu.
* Návrh a generování pokročilých statistických vizualizací a grafů (knihovny `matplotlib`, `seaborn`).
* Stylizaci a strukturování dokumentace (README).

Teoretický základ, volba registračních metod a výchozí nastavení hyperparametrů vycházejí z publikovaných vědeckých článků, které jsou řádně citovány v textu bakalářské práce. Samotná softwarová implementace, architektura projektu, a interpretace výsledků jsou mým vlastním autorským dílem.

**Plný text bakalářské práce je k dispozici zde:** [Srovnání algoritmů pro pružnou registraci dynamických MR skenů prsu]( https://www.vut.cz/studenti/zav-prace/detail/175726.)

## Použitá data a software (Citace)

- **Dataset**
HUANG, W., TUDORICA, A., CHUI, S., KEMMER, K., NAIK, A., TRO-
XELL, M., OH, K., ROY, N., AFZAL, A. a HOLTORF, M. Variations of dy-
namic contrast-enhanced magnetic resonance imaging in evaluation of breast
cancer therapy response: a multicenter data analysis challenge (QIN Breast
DCE-MRI) [online datová sada]. Version 2. The Cancer Imaging Archive, 2014.
Dostupné z: https://doi.org/10.7937/k9/tcia.2014.a2n1ixox. [cit. 2026-06-02].
- **SimpleITK (registrace obrazu)**
LOWEKAMP, Bradley C.; CHEN, David T.; IBÁÑEZ, Luis a BLEZEK, Daniel. The Design of SimpleITK. Online. Frontiers in Neuroinformatics. 2013, roč. 7. ISSN 1662-5196. Dostupné z: https://doi.org/10.3389/fninf.2013.00045. [cit. 2026-06-02].
- **Optuna (optimalizace hyperparametrů)**
AKIBA, Takuya; SANO, Shotaro; YANASE, Toshihiko; OHTA, Takeru a KOYAMA, Masanori. Optuna: A Next-generation Hyperparameter Optimization Framework. Online. KDD '19: Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 25 July 2019, s. 2623 - 2631. Dostupné z: https://doi.org/10.1145/3292500.3330701. [cit. 2026-06-02].