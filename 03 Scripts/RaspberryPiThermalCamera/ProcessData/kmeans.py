import h5py
import numpy as np
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

path = "D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20.hdf5"
with h5py.File(path,'r') as file:
    # number of clusters
    nc = 3
    # create estimator
    est = KMeans(n_clusters=nc,init='random')
    ## get an interesting frame
    max_dset = file['pi-camera-1'][()].max((0,1))
    # remove nans
    np.nan_to_num(max_dset,copy=False)
    # sort in ascending
    max_dset.sort()
    # find the max that is below 700
    mmax = max_dset[max_dset<700][-1]
    # find location
    ii = np.where(max_dset==mmax)[0]
    if ii.shape[0]>1:
        ii = ii[0]

    ii= 102500
    dd = file['pi-camera-1'][:,:,ii]
    # apply kmeans to data
    est.fit(file['pi-camera-1'][:,:,ii].ravel())
    print(tt)
    # get the labels
    labels = est.labels_
    # create an image based off the labels
    mask = np.zeros(file['pi-camera-1'].shape[:2],dtype='uint8')
    for ll in set(labels):
        mask[labels==ll]= 255//ll

    f,ax = plt.subplots(1,2)
    ax[0].contourf(file['pi-camera-1'][:,:,ii],cmap='gray')
    ax[1].imshow(mask)
    
    
