import numpy as np
import os
import pydicom
import napari

def zobrazeni_dcm_4d(path):
    """
    Načte 4D DCE-MRI data (DICOM) z adresářové struktury a zobrazí je
    v interaktivním prohlížeči Napari.

    Funkce očekává, že vstupní složka obsahuje podsložky představující
    jednotlivé časové okamžiky (např. T_00, T_01). V každé podsložce se
    musí nacházet jednotlivé DICOM řezy pro daný čas. Funkce data načte,
    uspořádá do 4D numpy pole (čas, řez, výška, šířka) a spustí vizualizaci.

    Parametry:
    ----------
    path : str
        Cesta ke kořenové složce pacienta, která obsahuje podsložky
        s DICOM daty pro jednotlivé časové okamžiky.
    """
    #Nahrání složek (čas) s daty (řez), pozor na SORTING
    time_stamps = os.listdir(path)
    dcm_files = []
    temp_data = []
    data = []

    #Procházení složky pacienta, obsahuje řezy v různých časech
    for file in time_stamps:
        time_folder = os.path.join(path, file)
        for dcm_file in os.listdir(time_folder):
            #Nalezení všech DICOM souborů v konkrétním čase a uložení jejich cest
            if dcm_file.endswith('.dcm'):
                dcm = os.path.join(time_folder, dcm_file)
                dcm_files.append(dcm)

        #Procházení nalezených souborů a načtení samotných obrazových dat
        for file in dcm_files:
            dcm = pydicom.dcmread(file)

            #Získání matice pixelů (pixel_array) z hlavičky DICOMu
            image = dcm.pixel_array
            temp_data.append(image)
        print(len(temp_data))
        #Přidání seznamu řezů z aktuálního času do hlavního datového pole
        data.append(temp_data)

        #Vynulování seznamů
        temp_data = []
        dcm_files = []

    #Převod standardního Python seznamu na vícerozměrné Numpy pole
    data = np.array(data)

    #Inicializace a spuštění  Napari
    viewer = napari.Viewer()
    viewer.add_image(data, colormap='gray', name = 'DCE-MRI')
    napari.run()

if __name__ == "__main__":
    zobrazeni_dcm_4d(r'C:\Users\andre\OneDrive\Plocha\School\BTB_3\Bakalářská práce\subjekty\pokus_1\manifest-data1407430404196\QIN Breast DCE-MRI\QIN-Breast-DCE-MRI-BC05\03-23-1997-NA-Breast CE-93044')