import pydicom
import numpy as np
import os

#Nacteni souboru, vythoreni 3D pole (rez, vyska, sirka)
def data_load(path, return_metada = False):
    dcm_files = []
    data = []

    for file in os.listdir(path): #prochazeni slozky, hledani souboru koncici .dcm
        if file.endswith('.dcm'):
            dcm = os.path.join(path, file)
            dcm_files.append(dcm)

    if dcm_files == []: #osetreni - nenalezeni souboru
        print('Neni nalezen soubor')
        return
    else:
        print(f'Nalezeno {len(dcm_files)} DICOM souboru')

    for file_path in dcm_files: #prochazeni nalezenych dcm souboru
        file = pydicom.dcmread(file_path) #nacteni souboru
        image = file.pixel_array #zisk obrazovych dat
        data.append(image) #ulozeni vsech obrazovych dat do seznamu

    data = np.array(data) #vlozeni do numpy pole

    number_of_images = len(data)#pocet nactenych rezu
    print(f'Pocet nactenych rezu: {number_of_images}')
    return data





