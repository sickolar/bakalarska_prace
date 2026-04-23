import SimpleITK as sitk

def data_load(path):
    reader = sitk.ImageSeriesReader()

    dcm = reader.GetGDCMSeriesFileNames(path) #Zisk dcm

    if not dcm: #Ostereni nalezeni dcm
        print('Neni nalezen DICOM soubor')
        return None

    print(f'Nalezeno {len(dcm)} DICOM souboru')

    reader.SetFileNames(dcm)
    image = reader.Execute()  #Zmena, nevracim ted pole ale ITK image

    size = image.GetSize()
    print(f'Rozmery (x,y,z): {size}') #Zobrazeni rozmeru voxelu

    return image
