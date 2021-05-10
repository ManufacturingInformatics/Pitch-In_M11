import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

path = "D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20.hdf5"
with h5py.File(path,'r') as file:

    max_dset = file['pi-camera-1'][()].max((0,1))
    # remove nans
    np.nan_to_num(max_dset,copy=False)
    # get sorted inicies in descending order
    ii = np.argsort(max_dset)[::-1]
    # find where the first max that is less than 700 occured
    jj=ii[max_dset[ii]<700][0]
    
    dd = file['pi-camera-1'][:,:,jj]
    pop,edges = np.histogram(file['pi-camera-1'][:,:,jj],bins=6)

    f = plt.figure(constrained_layout=True)
    gs = f.add_gridspec(2,3)
    hist_plot = f.add_subplot(gs[0,:])
    b1 = f.add_subplot(gs[1,0])
    b2 = f.add_subplot(gs[1,1])
    b3 = f.add_subplot(gs[1,2])

    os.makedirs("MaskHistograms",exist_ok=True)
    for ff in range(file['pi-camera-1'][()].shape[2]):
        dd = file['pi-camera-1'][:,:,ff]
        mmax,mmin = dd.max(),dd.min()
        np.nan_to_num(dd,copy=False)
        pop,edges = np.histogram(dd,bins=6)
        # plot histogram for top plot
        hist_plot.clear()
        hist_plot.bar(edges[:-1]+(edges[1]-edges[0])/2,pop,width=(edges[1]-edges[0]))
        mask = np.zeros(dd.shape,dd.dtype)
        mask[(dd>=edges[0]) & (dd<=edges[2])] = dd[(dd>=edges[0]) & (dd<=edges[2])]
        b1.contourf(mask,vmax=mmax,vmin=mmin)
        mask[...]=0.0
        mask[(dd>=edges[2]) & (dd<=edges[4])] = dd[(dd>=edges[2]) & (dd<=edges[4])]
        b2.contourf(mask,vmax=mmax,vmin=mmin)
        mask[...]=0.0
        mask[(dd>=edges[4]) & (dd<=edges[6])] = dd[(dd>=edges[4]) & (dd<=edges[6])]
        b3.contourf(mask,vmax=mmax,vmin=mmin)
        f.savefig(os.path.join("MaskHistograms","mask-hist-f{}.png".format(ff)))
    
