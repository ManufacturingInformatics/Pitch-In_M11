import h5py
import numpy as np
from scipy.io import wavfile

path= "D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20.hdf5"
with h5py.File(path,'r') as file:
    max_dset = file['pi-camera-1'][()].max((0,1))

    np.nan_to_num(max_dset,copy=False)
    wavfile.write("max-temp.wav",2000,max_dset*255)
