import numpy as np
import pywt
import scaleogram as scg
import os
import h5py
import matplotlib.pyplot as plt

def iter_wavelets(data,subMean=False):
    for w in pywt.wavelist():
        print(f"Trying {w}")
        try:
            f,ax = plt.subplots()
            if subMean:
                scg.cws(data[:,0],data[:,1]-data[:,1].mean(),scales=np.arange(10,250),wavelet=w,ax=ax,cmap='hsv',ylabel="Period [s]",xlabel="Time [s]",title=f"Scaleogram for Max Temperature minus Mean using {w.replace('.','-')}")
                f.savefig(f"maxTemp-wavelets-minus-mean-w{w.replace('.','-')}.png")
            else:
                scg.cws(data[:,0],data[:,1],scales=np.arange(10,250),wavelet=w,ax=ax,cmap='hsv',ylabel="Period [s]",xlabel="Time [s]",title=f"Scaleogram for Max Temperature using {w.replace('.','-')}")
                f.savefig(f"maxTemp-wavelets-w{w.replace('.','-')}.png")
        except AttributeError:
            continue
        plt.close(f)

print("Finding limits to temperature")
# get max temperature
with h5py.File("../Data/pi-camera-data-127001-2019-10-14T12-41-20.hdf5",'r') as file:
	maxTemp = file['pi-camera-1'][()].max(axis=(0,1))

f,ax = plt.subplots()
# frame rate of dataset, approx
fps = 30.92
# time vector based off fps
time = np.arange(0.0,maxTemp.shape[0]*(fps**-1.0),fps**-1.0)
# combine datasets
data = np.concatenate((time.reshape((-1,1)),maxTemp.reshape((-1,1))),axis=1)
# iterate over wavelets
print("generating scaleogram for max temperature")
iter_wavelets(data[91399:150000],False)
print("generating scaleogram for max temperature minus mean")
iter_wavelets(data[91399:150000],True)
