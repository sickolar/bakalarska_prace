import nibabel as nib
import napari
import numpy as np
def transposing(path_file):
    """
    Načte NIfTI soubor a přeskládá jeho osy pro správné zobrazení v prohlížeči Napari.

    Zdůvodnění:
    Medicínské formáty (jako NIfTI ukládané přes SimpleITK) často uchovávají 4D data
    v prostorovém pořadí os (X, Y, Z, Čas). Prohlížeč Napari ale pro správné fungování
    časové osy a řezů vyžaduje pořadí (Čas, Z, Y, X).
    """
    nifti_file = nib.load(path_file)
    image_data = nifti_file.get_fdata()

    #Transpozice přehodí osy z (0,1,2,3) na (3,2,1,0)
    final = np.transpose(image_data, (3,2,1,0))
    return final

def niftii_view(demons_file, mse_file, mi_file, rigid_file):
    """
    Načte výsledné 4D NIfTI objemy z jednotlivých registračních metod a zobrazí je
    ve společném interaktivním prostředí Napari pro vizuální porovnání.

    Funkce využívá techniku barevného prolnutí (additive blending). Referenční
    (nepohyblivý) obraz je zobrazen červeně a zregistrované pohyblivé obrazy zeleně.
    Kde se obě barvy dokonale překryjí (registrace je úspěšná), vznikne žlutá barva.

    Parametry:
    ----------
    demons_file : str
        Cesta k .nii.gz souboru s výsledkem Demons registrace.
    mse_file : str
        Cesta k .nii.gz souboru s výsledkem B-spline (MSE) registrace.
    mi_file : str
        Cesta k .nii.gz souboru s výsledkem B-spline (MI) registrace.
    rigid_file : str
        Cesta k .nii.gz souboru s výsledkem počáteční rigidní registrace.
    """

    #Načtení dat a oprava orientace os
    loaded_mse = transposing(mse_file)
    loaded_mi = transposing(mi_file)
    loaded_demons = transposing(demons_file)
    loaded_rigid = transposing(rigid_file)

    # Výpis rozměrů voxelů (spacing)
    print(f'Rigid spacing: {nib.load(rigid_file).header.get_zooms()}')
    print(f'Demons spacing: {nib.load(demons_file).header.get_zooms()}')
    print(f'MSE spacing: {nib.load(mse_file).header.get_zooms()}')
    print(f'MI spacing: {nib.load(mi_file).header.get_zooms()}')

    #Vytvoření umělého 4D referenčního obrazu.
    #Vezmeme nultý časový okamžik z rigidní registrace a naklonujeme ho do všech časových vrstev
    loaded_fixed = np.repeat(loaded_rigid[0:1, :, :, :], loaded_rigid.shape[0], axis=0)

    viewer = napari.Viewer()

    #Referenční obraz červeně, výsledky zeleně. Překryv = Žlutá barva.
    viewer.add_image(loaded_fixed, name='fixed_sede', colormap='gray', blending='additive')
    viewer.add_image(loaded_fixed, name='fixed', colormap='red', blending='additive')
    viewer.add_image(loaded_mse, name = 'MSE', colormap='green', blending='additive')
    viewer.add_image(loaded_mi, name='MI', colormap='green', blending='additive')
    viewer.add_image(loaded_demons, name='DEMONS', colormap='green', blending='additive')
    viewer.add_image(loaded_rigid, name='RIGID', colormap='green', blending='additive')

    #Zobrazení v šedotónu
    viewer.add_image(loaded_mse, name = 'MSE_gray', colormap='gray', blending='additive')
    viewer.add_image(loaded_mi, name='MI_gray', colormap='gray', blending='additive')
    viewer.add_image(loaded_demons, name='DEMONS_gray', colormap='gray', blending='additive')
    viewer.add_image(loaded_rigid, name='MSE_UNI_gray', colormap='gray', blending='additive')

    #Odkomentovat pro dělaní TRE statistiky
    #viewer.add_points(name='Referenční t=0', size=2, face_color='cyan', ndim=4)
    #viewer.add_points(name='MSE', size=2, face_color='white', ndim=4)
    #viewer.add_points(name='MI', size=2, face_color='blue', ndim=4)
    #viewer.add_points(name='Demons', size=2, face_color='orange', ndim=4)
    #viewer.add_points(name='MSE_uni', size=2, face_color='magenta', ndim=4)
    #viewer.add_points(name='MI_uni', size=2, face_color='red', ndim=4)
    #viewer.add_points(name='DEMONS_uni', size=2, face_color='yellow', ndim=4)

    #Odkomentovat pro zobrazení v mřížce
    #viewer.grid.enabled = True
    #viewer.grid.shape = (2, 2)

    napari.run()

