import numpy as np
import pywt
import os
import h5py
import matplotlib.pyplot as plt
from pywt._doc_utils import wavedec2_keys, draw_2d_wp_basis

max_lev = 3       # how many levels of decomposition to draw
label_levels = 3  # how many levels to explicitly label on the plots

def decomp_image(frame,wavelet='db2'):
    shape = frame.shape
    fig, axes = plt.subplots(2, 4, figsize=[14, 8])
    for level in range(0, max_lev + 1):
        if level == 0:
            # show the original image before decomposition
            axes[0, 0].set_axis_off()
            axes[1, 0].imshow(frame, cmap=plt.cm.gray)
            axes[1, 0].set_title('Image')
            axes[1, 0].set_axis_off()
            continue

        # plot subband boundaries of a standard DWT basis
        draw_2d_wp_basis(shape, wavedec2_keys(level), ax=axes[0, level],
                         label_levels=label_levels)
        axes[0, level].set_title('{} level\ndecomposition'.format(level))

        # compute the 2D DWT
        c = pywt.wavedec2(frame, wavelet, mode='periodization', level=level)
        # normalize each coefficient array independently for better visibility
        c[0] /= np.abs(c[0]).max()
        for detail_level in range(level):
            c[detail_level + 1] = [d/np.abs(d).max() for d in c[detail_level + 1]]
        # show the normalized coefficients
        arr, slices = pywt.coeffs_to_array(c)
        axes[1, level].imshow(arr, cmap=plt.cm.gray)
        axes[1, level].set_title('Coefficients\n({} level)'.format(level))
        axes[1, level].set_axis_off()

    plt.tight_layout()
    plt.savefig(f"wavelets/imageDecomp/waveletdecomp-array-thermal-camera-{ff:06d}.png")
    plt.close(fig)

with h5py.File("../Data/pi-camera-data-127001-2019-10-14T12-41-20.hdf5",'r') as file:
    os.makedirs("wavelets/imageDecomp",exist_ok=True)
    for ff in range(91399,150000,1):
        frame = file['pi-camera-1'][:,:,ff]
        # remove nans
        frame[np.isnan(frame)] = 0.0
        # decomp
        decomp_image(frame)
