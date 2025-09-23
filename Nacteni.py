import pydicom
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import widgets

#Nacteni souboru, vythoreni 4D pole (cas, rez, vyska, sirka)
def data_load(path):
    dcm_files = []
    data = []

    for file in os.listdir(path): #prochazeni slozky, hledani souboru koncici .dcm
        if file.endswith('.dcm'):
            dcm = os.path.join(path, file)
            dcm_files.append(dcm)
    print(dcm_files)

    if dcm_files == []: #osetreni - nenalezeni souboru
        print('Neni nalezen soubor')
        return
    else:
        print(f'Nalezeno {len(dcm_files)} DICOM souboru')

    for i, file_path in enumerate(dcm_files): #ocislovani a prochazeni souboru po jednom
        print(f'snime {i}/{len(dcm_files)}')

        file = pydicom.dcmread(file_path) #nacteni souboru

        plt.figure(figsize=(8,8)) #zobrazeni souboru
        plt.imshow(file.pixel_array, cmap='gray')
        plt.title(f'Obrazek cislo {i+1} - {os.path.basename(file_path)}')
        plt.show()

        if i < len(dcm_files) - 1:
            input('Enter pro dalsi obrazek')
        else:
            input('Posledni obrazek')

        plt.close()





data_load(r'C:\Users\andre\OneDrive\Plocha\School\BTB_3\Bakalářská práce\subjekty\pokus_1\pacient_1')




