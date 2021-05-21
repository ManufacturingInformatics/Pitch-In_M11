import h5py
import numpy as np
import os
import cv2
from scipy import fftpack
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
from scipy.constants import value as scipy_c_val

path = "D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20.hdf5"

# get wien wavelength displacement law constant
wien_w = scipy_c_val("Wien wavelength displacement law constant")

with h5py.File(path,'r') as file:
    w,h,d = file['pi-camera-1'].shape
    # convert temperature to Kelvin and estimate
    # wavelength of temperature based off black body radiator
    wave = wien_w/(file['pi-camera-1'][()]+273.15)
    np.nan_to_num(wave,copy=False)
    # remove negatives as it is known that the temperature does not go
    # below 0
    wave[wave<0.0]=0.0
    wavemax = (wave).max(axis=(0,1)).astype('float64')
    wavemin = (wave).min(axis=(0,1)).astype('float64')
    waverng = wavemax-wavemin
    f,ax = plt.subplots()
    ax.plot(wavemax*(10.0**9.0))
    ax.set(xlabel='Frame Index',ylabel='Max Estimated Wavelength (nm)')
    f.suptitle('Estimated Max Wavelength of Energy Using Wiens\n Displacement Law and Pi Temperature Information')
    f.savefig('est-max-wavelength.png')

    ax.clear()
    ax.plot(wavemax*(10.0**9.0))
    ax.set_ylim(bottom=10000.0,top=11000.0)
    ax.set(xlabel='Frame Index',ylabel='Max Estimated Wavelength (nm)')
    f.suptitle('Estimated Max Wavelength of Energy Using Wiens\n Displacement Law and Pi Temperature Information')
    f.savefig('est-max-wavelength-2.png')

    ax.clear()
    ax.plot(wavemin*(10.0**9.0))
    ax.set(xlabel='Frame Index',ylabel='Min Estimated Wavelength (nm)')
    f.suptitle('Estimated Min Wavelength of Energy Using Wiens\n Displacement Law and Pi Temperature Information')
    f.savefig('est-min-wavelength.png')

    ax.clear()
    ax.plot(waverng*(10.0**9.0))
    ax.set(xlabel='Frame Index',ylabel='Est Wavelength Range (nm)')
    f.suptitle('Est. Range of Wavelengths of Energy Using Wiens\n Displacement Law and Pi Temperature Information')
    f.savefig('est-range-wavelength.png')
