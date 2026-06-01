import SimpleITK as sitk

def data_load(path):
    """
    Načte sérii 2D DICOM snímků ze zadané složky a sestaví je do 3D obrazu
    ve formátu knihovny SimpleITK.

    Na rozdíl od prostého načtení obrazových matic (např. přes pydicom a numpy)
    tento přístup zachovává veškerá fyzikální metadata z DICOM hlaviček – zejména
    rozteč voxelů v milimetrech (spacing), směr os (direction) a počátek souřadného
    systému (origin). Tato metadata jsou nezbytná pro správné fungování
    registračních algoritmů a metrik podobnosti.

    Parametry:
    ----------
    path : str
        Cesta ke složce obsahující DICOM (.dcm) soubory pro jeden časový okamžik.

    Návratová hodnota:
    ------------------
    sitk.Image nebo None
        Vrací 3D obrazový objekt SimpleITK připravený pro registraci.
        Pokud ve složce nejsou žádné DICOM soubory, vrací None.
    """

    #Inicializace čtečky pro série snímků
    reader = sitk.ImageSeriesReader()

    #Získání seznamu všech DICOM souborů v zadaném adresáři
    dcm = reader.GetGDCMSeriesFileNames(path)

    #Ošetření nalezení dcm
    if not dcm:
        print('Neni nalezen DICOM soubor')
        return None

    print(f'Nalezeno {len(dcm)} DICOM souboru')

    #Předání seznamu souborů čtečce a provedení čtení
    reader.SetFileNames(dcm)
    image = reader.Execute()

    #Výpis rozměrů vytvořeného 3D objemu pro vizuální kontrolu (počet voxelů v osách X, Y, Z)
    size = image.GetSize()
    print(f'Rozmery (x,y,z): {size}')

    return image
