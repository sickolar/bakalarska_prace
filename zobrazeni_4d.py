import numpy as np
import os
import pydicom
import napari

#Nacteni 4d (cas, rez, vyska, sirka)
def data_load_4d(path):
    time_stamps = os.listdir(path) #Nahrani slozek (cas) s daty (rez), pozor na SORTING
    dcm_files = []
    temp_data = []
    data = []

    for file in time_stamps: #prochazeni slozky pacienta, obsahuje slozky s rezy v ruznych casech
        time_folder = os.path.join(path, file) #urcity casovy okamzik
        for dcm_file in os.listdir(time_folder): #prochazeni souboru v urcitem casovem okamziku
            if dcm_file.endswith('.dcm'): #pokud je to dcm soubor
                dcm = os.path.join(time_folder, dcm_file) #zisk cesty k danemu rezu
                dcm_files.append(dcm) #pridani do pole

        for file in dcm_files: #procazi vsechny dcm soubory v dane slozce (case)
            dcm = pydicom.dcmread(file) #nacte jednotlive soubory
            image = dcm.pixel_array #ziska tu cast kterou potrebujem - pixel_array
            temp_data.append(image) #prida hodnoty do seznamu
        print(len(temp_data))
        data.append(temp_data) #prida seznam pixel_array hodnot (rezu) z jednoho casoveho okamziku do seznamu
        #vynulovani seznamu, pro ukladani noveho casoveho okamziku
        temp_data = []
        dcm_files = []

    data = np.array(data) #prevod na numpy array
    return data



data = data_load_4d(r'C:\Users\andre\OneDrive\Plocha\School\BTB_3\Bakalářská práce\subjekty\pokus_1\manifest-data1407430404196\QIN Breast DCE-MRI\QIN-Breast-DCE-MRI-BC01\05-15-1996-NA-Breast CE-49782')

print(data.shape)
#vytvoreni vieweru
viewer = napari.Viewer()
viewer.add_image(data, colormap='gray', name='DCE-MRI', contrast_limits=[np.min(data), np.max(data)])  #contrast_limits moc nefunguje, furt mi to prijde zlvastni
napari.run()