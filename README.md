# 4D DCE-MRI: Nelineární registrace obrazů prsu

Tento repozitář obsahuje zdrojové kódy k bakalářské práci zaměřené na zpracování a registraci dynamických (4D) MRI snímků prsu s kontrastní látkou (DCE-MRI).

Během vyšetření dochází vlivem dýchání a přirozeného pohybu pacienta k deformacím tkáně, které ztěžují diagnostiku. Cílem implementovaných metod je automatická kompenzace tohoto pohybu pomocí registračních algoritmů a následná kvantifikace jejich úspěšnosti.

Projekt byl vyvíjen a primárně testován na standardizovaném veřejném datasetu **QIN-Breast-DCE-MRI** z databáze The Cancer Imaging Archive (TCIA).

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