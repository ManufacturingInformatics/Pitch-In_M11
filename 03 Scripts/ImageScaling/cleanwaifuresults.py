import h5py
import os
from os import scandir
from scipy.signal import find_peaks
import fnmatch

path = "D:\\BEAM\\Scripts\\CallWaifu2x\\arrowshape-temperature-HDF5.hdf5"
dres = 10
rres = 1
cres = 2
print("Scanning file")
with h5py.File(path,'r') as file:
    import numpy as np
    # get the maximum values of each frame
    D_max = file[list(file.keys())[0]][rres:,cres:,::dres].max(axis=(0,1))
    # find the peaks in the values
    peaks = find_peaks(D_max)[0]
    tol = 0.05
    D_abs_max = D_max.max()
    # find the peaks close to max
    # these indicies are used to filter images
    pki = peaks[np.where(D_max[peaks]>=(1-tol)*D_abs_max)]
print("Reducing file groups to {} files".format(pki.shape[0]))

def scantree(path):
    """Recursively yield DirEntry objects for given directory."""
    for entry in scandir(path):
        if not entry.name.startswith('.') and entry.is_dir(follow_symlinks=False):
            yield from scantree(entry.path)
        else:
            yield entry

path = "D:\\BEAM\\Scripts\\CallWaifu2x\\arrowshape-temperature-HDF5"
print("Searching for files to delete")
patts = ['pi-frame-*.png','pi-frame-*.tif','pi-frame-*.tiff']
for entry in scantree(path):
    # frames are named as pi-frame-<int>
    # so to check if it's in the list the filename is split and searched
    if any(fnmatch.fnmatch(entry.name,p) for p in patts):
        if not int(os.path.splitext(entry.name)[0].split('-')[-1]) in pki:
            os.remove(entry.path)
        
        
