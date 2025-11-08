import nibabel as nib
import napari
import numpy as np

#Simple kod jen na kontrolu, prepsat na funkci, vlozit do mainu
path_file = r'C:\Bakalarka\Nifty_files\registered_4D_2.nii.gz'
nifti_file = nib.load(path_file)
image_data = nifti_file.get_fdata()
final = np.transpose(image_data, (3,2,1,0)) #zmena os, z vystupu registrace_full() jsou poprehazovane osy
viewer = napari.Viewer()
viewer.add_image(final, colormap='grey')
napari.run()
