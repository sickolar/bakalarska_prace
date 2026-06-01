# 4D DCE-MRI: Nelineární registrace obrazů prsu

Tento repozitář obsahuje zdrojové kódy k bakalářské práci zaměřené na zpracování a registraci dynamických (4D) MRI snímků prsu s kontrastní látkou (DCE-MRI).

Během vyšetření dochází vlivem dýchání a přirozeného pohybu pacienta k deformacím tkáně, které ztěžují diagnostiku. Cílem implementovaných metod je automatická kompenzace tohoto pohybu pomocí registračních algoritmů a následná kvantifikace jejich úspěšnosti.

## Implementované funkce
Projekt pokrývá následující kroky zpracování obrazových dat:
1. **Předzpracování dat:** Načítání syrových DICOM souborů při zachování fyzikálních metadat (rozteč voxelů, počátek souřadnic).
2. **Globální zarovnání:** Počáteční rigidní registrace obrazů (Euler3DTransform).
3. **Lokální deformace (Nelineární registrace):**
   * B-spline optimalizovaný pomocí metriky MSE (Mean Squares).
   * B-spline optimalizovaný pomocí metriky MI (Mutual Information).
   * Diffeomorfní algoritmus Demons (s předzpracováním přes Histogram Matching).
4. **Optimalizace hyperparametrů:** Využití frameworku Optuna k systematickému hledání optimální hustoty mřížky, počtu iterací a přesnosti konvergence na maskované oblasti prsu.
5. **Kvantitativní vyhodnocení:** Výpočet cílové chyby registrace (TRE - Target Registration Error) v milimetrech a statistické srovnání metod pomocí Wilcoxonova párového testu.

## Struktura repozitáře

* `registrace.py` - Hlavní modul. Obsahuje sjednocené funkce pro všechny registrační přístupy a ukládá zregistrované 4D objemy do formátu NIfTI.
* `testovani_hyperparametru.py` - Skript využívající knihovnu Optuna. Prohledává prostor deformačních parametrů a hodnotí úspěšnost registrace na základě oříznuté masky.
* `grafy.py` - Analytický a vizualizační modul. Generuje konvergenční křivky, sdružené boxploty (s využitím přerušené osy Y pro zobrazení extrémních odchylek) a zpracovává statistiku.
* `Nacteni_image.py` - Modul založený na SimpleITK pro načítání DICOM dat bez porušení prostorových metadat.
* `zobrazeni_4d.py` - Nástroj pro chronologické načtení hrubých 4D DICOM dat a jejich vizualizaci v prostředí Napari.
* `niftii_load.py` - Diagnostický modul pro vizuální kontrolu registrace pomocí aditivního prolínání barev (referenční vs. zaregistrovaný obraz).

## Instalace a spuštění

Pro běh skriptů je doporučeno vytvořit lokální virtuální prostředí a nainstalovat požadované knihovny:

```bash
pip install -r requirements.txt
