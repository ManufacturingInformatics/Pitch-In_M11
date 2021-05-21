import h5py
import numpy as np
from scipy.cluster.vq import whiten
#from sklearn.cluster import KMeans as kmeans
from sklearn.cluster import MiniBatchKMeans as kmeans
import os
import matplotlib.pyplot as plt
import cv2

# make figure
f,ax = plt.subplots()
# path to hdf5 files
path = "D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20.hdf5"
# create directory for results
os.makedirs("results",exist_ok=True)
# open file in context manager
with h5py.File(path,'r') as file:
    # get the shape of dataset
    r,c,d = file['pi-camera-1'].shape
    # iterate over each frame
    for ff in range(d):
        if (ff%100)==0:
            print(ff)
        # get frame and reshape into array of features
        frame = file['pi-camera-1'][:,:,ff]
        frame = frame.ravel().reshape(-1,1)
        # remove nans
        frame[np.isnan(frame)]=0.0
        # performs kmeans segmentation
        # fitting to data
        kk = kmeans(n_clusters=3,random_state=0).fit(frame)
        # get labels output and reshape to an image
        res = kk.labels_.reshape((r,c))
        # convert to 8-bit
        res = res.astype("uint8")
        # normalize
        cv2.normalize(res,res,0,255,cv2.NORM_MINMAX)
        # apply color map
        cv2.imwrite(os.path.join("results",f"pi-camera-kmeans-f{ff:06d}.png"),cv2.applyColorMap(res,cv2.COLORMAP_JET))
    
        
        
    
