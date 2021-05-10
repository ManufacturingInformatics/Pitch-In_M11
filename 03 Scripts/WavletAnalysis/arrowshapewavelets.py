import h5py
import numpy as np
import matplotlib.pyplot as plt
import pywt
from pywt._doc_utils import wavedec2_keys, draw_2d_wp_basis
import scaleogram as scg
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

def iter_wavelets_pixels(dset,shape=None,tmin=None,tmax=None,fps=30.92):
    if shape is None:
        nr,nc,nframes=dset[()].shape
    else:
        nr,nc,nframes = shape

    if tmin is None:
        tmin = dset[()].min()

    if tmax is None:
        tmax = dset[()].max()
    # create folder for results
    plot_path=os.path.join(r"C:\Users\david\Documents\CoronaWork\Plots","pixelDecomp")
    os.makedirs(plot_path,exist_ok=True)
    # build time vector
    tt = fps**-1.0
    time = np.arange(0.0,nframes*tt,tt)
    print("iterating over pixels")
    for rr in range(nr):
        for cc in range(nc):
            print(f"Pixel {rr},{cc}")
            for w in pywt.wavelist():
                print(f"Trying {w}")
                try:
                    # decomp pixel over dataset
                    f,ax = plt.subplots()
                    scg.cws(time,dset[rr,cc,:],scales=np.arange(23.0,tmax),wavelet=w,ax=ax,cmap='hsv',ylabel="Period [s]",xlabel="Time [s]",title=f"Scaleogram for Arrow Shape Temperature Pixel {rr},{cc}\n using {w.replace('.','-')}")
                    f.savefig(os.path.join(plot_path,f"arrow-shape-w-{w}-r-{rr}-c-{cc}.png"))
                    plt.close('f')
                    # decomp pixel minus mean across dataset
                    f,ax = plt.subplots()
                    scg.cws(time,dset[rr,cc,:]-dset[rr,cc,:].mean(),scales=np.arange(23.0,tmax),wavelet=w,ax=ax,cmap='hsv',ylabel="Period [s]",xlabel="Time [s]",title=f"Scaleogram for Arrow Shape Temperature Pixel {rr},{cc} minus mean\n using {w.replace('.','-')}")
                    f.savefig(os.path.join(plot_path,f"arrow-shape-mean-w-{w}-r-{rr}-c-{cc}.png"))
                    plt.close('f')
                except AttributeError:
                    continue

def thermal_decomp(dataset,shape,maxlvl=3,label_levels=3):
    nr,nc,nframes = shape
    # create folders for rows decomp
    plot_path=os.path.join(r"C:\Users\david\Documents\CoronaWork\Plots","thermalDecomp")
    os.makedirs(plot_path,exist_ok=True)
    # build time vector
    f,axes = plt.subplots(2,label_levels+1,figsize=[14,8])
    print("iterating over frames")
    # iterate over each frame
    for ff in range(nframes):
        # compute 2D DWT
        for level in range(0,maxlvl+1):
            if level == 0:
                # show the original image before decomposition
                axes[0, 0].set_axis_off()
                axes[1, 0].imshow(dataset[:,:,ff], cmap=plt.cm.gray)
                axes[1, 0].set_title(f'Image {ff}')
                axes[1, 0].set_axis_off()
                continue
            # plot subband boundaries of a standard DWT basis
            axes[0, level].clear()
            draw_2d_wp_basis(shape, wavedec2_keys(level), ax=axes[0, level],
                             label_levels=label_levels)
            axes[0, level].set_title('{} level\ndecomposition'.format(level))
            # compute the 2D DWT
            c = pywt.wavedec2(dataset[:,:,ff], 'db2', mode='periodization', level=level)
            # normalize each coefficient array independently for better visibility
            c[0] /= np.abs(c[0]).max()
            for detail_level in range(level):
                c[detail_level + 1] = [d/np.abs(d).max() for d in c[detail_level + 1]]
            # show the normalized coefficients
            arr, slices = pywt.coeffs_to_array(c)
            axes[1, level].imshow(arr, cmap=plt.cm.gray)
            axes[1, level].set_title('Coefficients\n({} level)'.format(level))
            axes[1, level].set_axis_off()
        f.savefig(os.path.join(plot_path,f"arrow-shape-decomp-{ff}.png"))

def thermal_decomp_colorbar(dataset,shape,maxlvl=3,label_levels=3):
    nr,nc,nframes = shape
    # create folders for rows decomp
    plot_path=os.path.join(r"..\Plots","thermalDecompCbar")
    os.makedirs(plot_path,exist_ok=True)
    # build time vector
    f,axes = plt.subplots(2,label_levels+1,figsize=[14,8])
    # make axes for colorbars
    divs = [make_axes_locatable(axes[1,lvl]) for lvl in range(label_levels+1)]
    cax = [dd.append_axes('right','5%','3%') for dd in divs]
    cbars = []
    # setup colorbars
    cbars_setup = False
    print("iterating over frames")
    # iterate over each frame
    for ff in range(nframes):
        print(f"ff {ff}")
        # compute 2D DWT
        for level in range(0,maxlvl+1):
            print(f"level {level}")
            if level == 0:
                # show the original image before decomposition
                axes[0, 0].set_axis_off()
                im=axes[1, 0].imshow(dataset[:,:,ff], cmap='hsv')
                axes[1, 0].set_title(f'Image {ff}')
                axes[1, 0].set_axis_off()
                # clear colorbar axes
                cax[0].cla()
                # setup colorbars
                if not cbars_setup:
                    cbars.append(f.colorbar(im,cax=cax[level]))
                else:
                    cbars[level]=f.colorbar(im,cax=cax[level])
                continue
            # plot subband boundaries of a standard DWT basis
            axes[0, level].clear()
            draw_2d_wp_basis(shape, wavedec2_keys(level), ax=axes[0, level],
                             label_levels=label_levels)
            axes[0, level].set_title('{} level\ndecomposition'.format(level))
            # compute the 2D DWT
            c = pywt.wavedec2(dataset[:,:,ff], 'db2', mode='periodization', level=level)
            # normalize each coefficient array independently for better visibility
            c[0] /= np.abs(c[0]).max()
            for detail_level in range(level):
                c[detail_level + 1] = [d/np.abs(d).max() for d in c[detail_level + 1]]
            # show the normalized coefficients
            arr, slices = pywt.coeffs_to_array(c)
            im=axes[1, level].imshow(arr, cmap='hsv')
            axes[1, level].set_title('Coefficients\n({} level)'.format(level))
            axes[1, level].set_axis_off()
            ## update colorbars
            # clear axes
            cax[level].cla()
            # if the colorbars have not been built yet
            # build the list of colorbars
            if not cbars_setup:
                cbars.append(f.colorbar(im,cax=cax[level]))
                if len(cbars)==len(cax):
                    cbars_setup=True
            else:
                print(f"cbars {len(cbars)},{len(cax)}")
                cbars[level]=f.colorbar(im,cax=cax[level])
        f.savefig(os.path.join(plot_path,f"arrow-shape-decomp-{ff}.png"))

if __name__ == "__main__":
    with h5py.File("../Data/pi-camera-data-127001-2019-10-14T12-41-20.hdf5",'r') as file:
        ds = file['pi-camera-1']
        # get size of dataset
        nr,nc,nframes = ds.shape
        print(f"Dataset {nr},{nc},{nframes}")
        # get range
        tmin,tmax = np.nanmin(ds[()],axis=(0,1,2)),np.nanmax(ds[()],axis=(0,1,2))
        print(f"Dataset limits {tmin},{tmax}")
        thermal_decomp_colorbar(ds,(nr,nc,nframes))
        
    
        
