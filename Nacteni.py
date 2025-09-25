import pydicom
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.widgets import Slider

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

    for file_path in dcm_files: #prochazeni nalezenych dcm souboru
        file = pydicom.dcmread(file_path) #nacteni souboru
        image = file.pixel_array #zisk obrazovych dat
        data.append(image) #ulozeni vsech obrazovych dat do seznamu

    data = np.array(data) #vlozeni do numpy pole

    number_of_images = len(data)#pocet nactenych obrazku


    fig, ax = plt.subplots() #fig - cele okno, ax - plocha pro vykresleni
    plt.subplots_adjust(bottom=0.2) #pro slider
    image_displayed = ax.imshow(data[0], cmap='grey') #imshow, pro prvni obrazek, ax konkretni osa
    ax.set_title(f'Snimek 1 z {number_of_images}')

    #Slider
    slider_position = plt.axes([0.2, 0.05, 0.6, 0.03]) #Osa pro slider - kde se bude nachazet v ramci fig
    slider = Slider(slider_position, label='Snímek', valmin=0, valmax=number_of_images - 1, valinit=0, valstep=1)#nastaveni slideru

    slider.on_changed(lambda val: slider_update(slider, data, fig, ax, image_displayed, number_of_images)) #lambad premostovaci funkce - prijme value, preda ji slider update
    plt.show() #zobrazi otevrenou fig
#Funkce pro posun slideru
def slider_update(slider, data, fig, ax, image_displayed, number_of_images):
    i = int(slider.val) #ziska kde se nachazime v ramci slideru
    image_displayed.set_data(data[i]) #zobrazi snimek podle pozice slideru
    ax.set_title(f'Snímek {i + 1} z {number_of_images}') #zmeni popisek
    fig.canvas.draw_idle() #aktualizujeme zobrazeny obrazek.







data_load(r'C:\Users\andre\OneDrive\Plocha\School\BTB_3\Bakalářská práce\subjekty\pokus_1\pacient_1')




