import registrace_all

#Bude zde asi switch, uzivatel vybere jednu z moznosti, ktere boudou
#testing jeden moving, vsechny moving neboli registrace_all,
#Zobrazeni dcm, zobrazeni nii.gz
#Popripdae pozdeji pridani metrik
def main():
    #Deklarace cest
    read_path = r'C:\Users\andre\OneDrive\Plocha\School\BTB_3\Bakalářská práce\subjekty\pokus_1\manifest-data1407430404196\QIN Breast DCE-MRI\QIN-Breast-DCE-MRI-BC01\05-15-1996-NA-Breast CE-49782'
    save_path = r'C:\Bakalarka\Nifty_files'

    registrace_all.registration_full(read_path,save_path)


if __name__ == '__main__':
    main()