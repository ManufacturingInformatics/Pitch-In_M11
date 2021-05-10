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

data_ranges = [[47500,48050],[47500,49500],[60000,60450],[60000,62000],[68000,71000],[68200,68820],[90000,93000],[96800,100750],[100750,131750],[90000,150000]]
path = "D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20.hdf5"

os.makedirs("FFT\Spectrum",exist_ok=True)
os.makedirs("FFT\Gaussian",exist_ok=True)
os.makedirs("FFT\Filter\Brute",exist_ok=True)
with h5py.File(path,'r') as file:
    keep_r = 0.1
    f,ax = plt.subplots(2,1)
    for dd in data_ranges:
        print(dd)
        os.makedirs("FFT\Spectrum\{}-{}".format(dd[0],dd[1]),exist_ok=True)
        os.makedirs("FFT\Gaussian\{}-{}".format(dd[0],dd[1]),exist_ok=True)
        os.makedirs("FFT\Filter\Brute\{}-{}".format(dd[0],dd[1]),exist_ok=True)
        os.makedirs("FFT\Filter\FB\{}-{}".format(dd[0],dd[1]),exist_ok=True)
        for ff in range(dd[0],dd[1]):
            #print("\r{}".format(ff),end='')
            for aa in ax:
                aa.clear()
                
            frame = file['pi-camera-1'][:,:,ff]
            np.nan_to_num(frame,copy=False)
            ax[0].imshow(frame,'gray')
            ff2 = fftpack.fft2(frame)
            ax[1].imshow(np.abs(ff2),norm=LogNorm())
            f.savefig(os.path.join("FFT\Spectrum\{}-{}".format(dd[0],dd[1]),'pi-camera-fft-f{}.png'.format(ff)))

##            frame_norm = (((frame-frame.min())/frame.max())*255).astype('uint8')
##            frame_norm = np.dstack([frame_norm]*3)
##            ff2_abs = np.abs(ff2)
##            ff2_abs = ((ff2_abs/ff2_abs.max())*255).astype('uint8')
##            ff2_abs = cv2.applyColorMap(ff2_abs,cv2.COLORMAP_HSV)
##            frame_stack = np.concatenate((frame_norm,ff2_abs))
##            cv2.imwrite(os.path.join("FFT\Spectrum\{}-{}".format(dd[0],dd[1]),'pi-camera-fft-f{}.png'.format(ff)),frame_stack)

            ## apply gaussian blue to try and clear up the images
            img_blur = ndimage.gaussian_filter(frame,4)
            ax[1].imshow(img_blur,plt.cm.gray)
            f.savefig(os.path.join("FFT\Gaussian\{}-{}".format(dd[0],dd[1]),'pi-camera-gaussian-f{}.png'.format(ff)))

            ## filtering frequencies
            r,c = frame.shape
            # set the rows with indicies between r*keep and r*(1-keep)
            ff2[int(r*keep_r):int(r*(1-keep_r))] = 0
            ff2[:, int(c*keep_r):int(c*(1-keep_r))] = 0
            ax[1].imshow(fftpack.ifft2(ff2).real,cmap='gray')
            f.savefig(os.path.join("FFT\Filter\Brute\{}-{}".format(dd[0],dd[1]),"pi-camera-filter-brute-f{}.png".format(ff)))

            ## foreground and background thresholding
            # normalize to 8-bit
            frame_norm = ((frame/frame.max())*255).astype('uint8')
            # threshold using mean value
            frame_norm = cv2.threshold(frame_norm,frame_norm.mean(),frame_norm.mean(),cv2.THRESH_BINARY)[1]
            ax[1].imshow(frame_norm,'gray')
            f.savefig(os.path.join("FFT\Filter\FB\{}-{}".format(dd[0],dd[1]),"pi-camera-fbk-filter-f{}.png".format(ff)))
            

            
        
        
