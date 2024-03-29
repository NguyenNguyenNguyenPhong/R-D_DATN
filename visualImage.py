from vedo import load, show, Volume

path_nifti = "D:\Data\evaluation\evaluation\\102.nii.gz"



def visualize_nifti(path_to_file, bg=(1,1,1), mesh_color=(1,0,0)):
    # Load a NIfTI file
    vol = Volume(path_to_file)

    # Show the volume
    show(vol, bg=bg)

visualize_nifti(path_nifti)