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

path = "D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20.hdf5"

os.makedirs("CircleFilt",exist_ok=True)
os.makedirs("LineFilt",exist_ok=True)
with h5py.File(path,'r') as file:
    # get shape of the dataset
    w,h,d = file['pi-camera-1'].shape
    ## create mask
    mask = np.zeros((w,h),dtype=np.uint8)
    r=8
    # draw circles with the centres in the corners of the iamge
    mask = cv2.circle(mask,(0,0),r,1,-1)
    mask = cv2.circle(mask,(0,w),r,1,-1)
    mask = cv2.circle(mask,(h,0),r,1,-1)
    mask = cv2.circle(mask,(h,w),r,1,-1)
    cv2.imwrite(os.path.join("CircleFilt","mask-circlefilt-ref.png"),mask*255)

    mask_ln = mask.copy()
    mask_ln[...]=0.0
    mask_ln[10:15,:] = 1.0
    cv2.imwrite(os.path.join("LineFilt","mask-linefilt-ref.png"),mask_ln*255)

    # get axes plot
    f,ax = plt.subplots(2,1)
    for ff in range(91399,93000):
        # get frame
        frame = file['pi-camera-1'][:,:,ff]
        ax[0].imshow(frame,cmap='gray')
        np.nan_to_num(frame,copy=False)
        # perform fft
        dc = fftpack.fftn(frame)
        # apply mask and show result
        ax[1].imshow(np.abs(fftpack.ifftn(dc*mask)),cmap='gray')
        f.savefig(os.path.join("CircleFilt","circle-filt-r{}-ff{}.png".format(r,ff)))

        ax[1].imshow(np.abs(fftpack.ifftn(dc*mask_ln)),cmap='gray')
        f.savefig(os.path.join("LineFilt","line-filt-r{}-ff{}.png".format(r,ff)))
