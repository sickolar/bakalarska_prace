import nibabel as nib
import napari
import numpy as np
#funkce pro nacteni a upravu os
def transposing(path_file):
    nifti_file = nib.load(path_file)
    image_data = nifti_file.get_fdata()
    final = np.transpose(image_data, (3,2,1,0)) #zmena os, jsou poprehazovane osy
    return final

#Zobrazeni niftii v mrizce pro TRE
def niftii_view(demons_file, mse_file, mi_file, rigid_file):

    loaded_mse = transposing(mse_file)
    loaded_mi = transposing(mi_file)
    loaded_demons = transposing(demons_file)
    loaded_rigid = transposing(rigid_file)

    print(f'Rigid spacing: {nib.load(rigid_file).header.get_zooms()}')
    print(f'Demons spacing: {nib.load(demons_file).header.get_zooms()}')
    print(f'MSE spacing: {nib.load(mse_file).header.get_zooms()}')
    print(f'MI spacing: {nib.load(mi_file).header.get_zooms()}')

    viewer = napari.Viewer()

    viewer.add_image(loaded_mse, name = 'MSE', colormap='gray', blending='additive')
    viewer.add_image(loaded_mi, name='MI', colormap='gray', blending='additive')
    viewer.add_image(loaded_demons, name='DEMONS', colormap='gray', blending='additive')
    viewer.add_image(loaded_rigid, name='RIGID', colormap='gray', blending='additive')

    viewer.add_points(name='Referenční t=0', size=2, face_color='blue', ndim=4)
    viewer.add_points(name='MSE', size=2, face_color='red', ndim=4)
    viewer.add_points(name='MI', size=2, face_color='green', ndim=4)
    viewer.add_points(name='Demons', size=2, face_color='yellow', ndim=4)
    viewer.add_points(name='Rigidní', size=2, face_color='magenta', ndim=4)
    viewer.grid.enabled = True
    viewer.grid.shape = (2, 2)
    napari.run()


niftii_view(r'C:\Bakalarka\Nifty_files\registered_4D_demons_59716.nii.gz', r'C:\Bakalarka\Nifty_files\registered_4D_mse_59716.nii.gz', r'C:\Bakalarka\Nifty_files\registered_4D_mi_59716.nii.gz', r'C:\Bakalarka\Nifty_files\registered_4D_rigid_59716.nii.gz')