import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar

path = "D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20.hdf5"
os.makedirs("Histograms",exist_ok=True)

# create plot axes
f,ax = plt.subplots(1,2)
# open file
with h5py.File(path,'r') as file:
    print("Running histograms + global contours")
    max_dset = file['pi-camera-1'][()].max((0,1))
    # remove nans
    np.nan_to_num(max_dset,copy=False)
    # sort in ascending
    max_dset.sort()
    # find the max that is below 700
    mmax = max_dset[max_dset<700][-1]
    mmin = file['pi-camera-1'][()].min((0,1,2))
    for ff in range(3,file['pi-camera-1'][()].shape[2]):
        for aa in ax:
            aa.clear()
        pop,edges = np.histogram(file['pi-camera-1'][:,:,ff],bins=5)
        ax[0].bar(edges[:-1]+(edges[1]-edges[0])/2,pop,width=(edges[1]-edges[0]))
        ax[0].set(xlabel='Temperature (C)',ylabel='Population',title='Histogram, Frame {}'.format(ff))
        ax[1].contourf(file['pi-camera-1'][:,:,ff],vmax=mmax,vmin=mmin)
        f.savefig(os.path.join("Histograms","pi-camera-hist-ct-f{}.png".format(ff)))

