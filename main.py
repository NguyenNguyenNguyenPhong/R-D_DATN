import dicom2nifti
import nibabel as nib
import nilearn as nil
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import os
brain_vol = nib.load('D:\Data\evaluation\evaluation\\101.nii.gz')
brain_vol_data = brain_vol.get_fdata()

# What is the type of this object?
plt.imshow(brain_vol_data[2], cmap='bone')
plt.axis("scaled")
plt.show()

fig_rows = 4
fig_cols = 4
n_subplots = fig_rows * fig_cols
n_slice = brain_vol_data.shape[0]
step_size = n_slice // n_subplots
plot_range = n_subplots * step_size
start_stop = int((n_slice - plot_range) / 2)

fig, axs = plt.subplots(fig_rows, fig_cols, figsize=[10, 10])

for idx, img in enumerate(range(start_stop, plot_range, step_size)):
    axs.flat[idx].imshow(ndi.rotate(brain_vol_data[img, :, :], 90), cmap='gray')
    axs.flat[idx].axis('scaled')

plt.tight_layout()
plt.show()


from nilearn import plotting

plotting.plot_img(brain_vol)
plt.show()

from nilearn import plotting

fig, ax = plt.subplots(figsize=[10, 5])
plotting.plot_img(brain_vol, cmap='gray', axes=ax)
plt.show()

plotting.plot_img(brain_vol, display_mode='tiled', cmap='gray')
plt.show()

plotting.plot_img(brain_vol, cmap='gray', cut_coords=(-45, 40, 0))
plt.show()

plotting.plot_img(brain_vol, display_mode='x', cmap='gray')
plt.show()

plotting.plot_img(brain_vol, display_mode='mosaic', cmap='gray')
plt.show()


