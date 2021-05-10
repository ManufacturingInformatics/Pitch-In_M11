import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from scipy.fftpack import fftn,ifftn,fftshift
from scipy.signal import find_peaks
from skimage.io import imsave as sk_imsave
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
import matplotlib.animation as animation

path = "D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20.hdf5"
# get the name of the file
fname = os.path.splitext(os.path.basename(path))[0]
# create folder for plots
folder_path = fname+"-plots"
os.makedirs(folder_path,exist_ok=True)

# create plot axes
f,ax = plt.subplots()
# open file
with h5py.File(path,'r') as file:
    print("Getting min max mean datasets")
    # get min and max temperatures
    max_dset = np.max(file['pi-camera-1'][()],axis=(0,1))
    min_dset = np.min(file['pi-camera-1'][()],axis=(0,1))
    mean_dset = np.mean(file['pi-camera-1'][()],axis=(0,1))
    print("Saving data")
    # save data as CSVs
    np.savetxt(os.path.join(folder_path,"pi-camera-max-temp.csv"),max_dset,delimiter=',')
    np.savetxt(os.path.join(folder_path,"pi-camera-min-temp.csv"),min_dset,delimiter=',')
    np.savetxt(os.path.join(folder_path,"pi-camera-mean-temp.csv"),mean_dset,delimiter=',')
    ## create time dataset for x axis
    # frames per second
    fps = 31
    # number of frames divided by fps gives the total time recorded
    total_time = max_dset.shape[0]/fps
    # create dataset
    time_set = np.arange(0,total_time,1/fps)

    print("plotting datasets")
    ax.plot(max_dset)
    ax.set(xlabel='Frame Index',ylabel='Max Temperature (C)')
    f.suptitle('Maximum Temperature Recorded by the Raspberry Pi Thermal Camera')
    f.savefig(os.path.join(folder_path,"pi-camera-max-temp.png"))

    ax.clear()
    ax.plot(time_set,max_dset)
    ax.set(xlabel='Time (secs)',ylabel='Max Temperature (C)')
    f.savefig(os.path.join(folder_path,"pi-camera-max-time-temp.png"))
    
    ax.clear()
    ax.plot(min_dset)
    ax.set(xlabel='Frame Index',ylabel='Minumum Temperature (C)')
    f.suptitle('Minumum Temperature Recorded by the Raspberry Pi Thermal Camera')
    f.savefig(os.path.join(folder_path,"pi-camera-min-temp.png"))

    ax.clear()
    ax.plot(time_set,min_dset)
    ax.set(xlabel='Time (secs)',ylabel='Minimum Temperature (C)')
    f.savefig(os.path.join(folder_path,"pi-camera-min-time-temp.png"))

    ax.clear()
    ax.plot(mean_dset)
    ax.set(xlabel='Frame Index',ylabel='Average Temperature (C)')
    f.suptitle('Average Temperature Recorded by the Raspberry Pi Thermal Camera')
    f.savefig(os.path.join(folder_path,"pi-camera-mean-temp.png"))
    
    ax.clear()
    ax.plot(time_set,mean_dset)
    ax.set(xlabel='Time (secs)',ylabel='Average Temperature (C)')
    f.savefig(os.path.join(folder_path,"pi-camera-mean-time-temp.png"))

## find peaks
pks = find_peaks(max_dset)[0]
ax.clear()
ax.plot(max_dset,'b-',pks,max_dset[pks],'rx')
ax.set(xlabel='Frame Index',ylabel='Max Temperature (C)')
f.suptitle('Maximum Temperature Recorded by the Raspberry Pi Thermal Camera\nwith the Peaks Marked')
f.savefig(os.path.join(folder_path,'pi-camera-max-temp-peaks.png'))

## filter peaks
tol = 0.30
pki = np.where(max_dset>=(max_dset[pks].min()*(1+tol)))[0]
ax.clear()
ax.plot(max_dset,'b-',pki,max_dset[pki],'rx')
ax.set(xlabel='Frame Index',ylabel='Max Temperature (C)')
f.suptitle('Maximum Temperature Recorded by the Raspberry Pi Thermal Camera\nwith the Peaks Marked and Filtered')
f.savefig(os.path.join(folder_path,'pi-camera-max-temp-peaks-filt.png'))

# save images that match the data points filtered
##print("Writing peak images to file")
##os.makedirs(os.path.join(folder_path,'PeakImages'),exist_ok=True)
##with h5py.File(path,'r') as file:
##    for ii in pki:
##        frame = file['pi-camera-1'][:,:,ii]
##        # if not between 0 and 1, normalize to limits
##        if (frame.min() != 0.0) or (frame.max() != 1.0):
##            frame = (frame-frame.min())/(frame.max()-frame.min())
##        # save as tif
##        #sk_imsave(os.path.join(folder_path,'PeakImages','pi-camera-f{}.png'.format(ii)),frame)
##        cv2.imwrite(os.path.join(folder_path,'PeakImages','pi-camera-f{}.png'.format(ii)),(frame*255).astype('uint8'))
##    
## fft on specific data ranges
##data_ranges = [[47500,48050],[47500,49500],[60000,60450],[60000,62000],[68000,71000],[68200,68820],[90000,93000],[96800,100750],[100750,131750],[90000,150000]]
##f,ax = plt.subplots(2,1)
##for dd in data_ranges:
##    for aa in ax:
##        aa.clear()    
##    # perform fft on the data range specified
##    ff = fftn(max_dset[dd[0]:dd[1]])
##    # get the number of datasets specified in that range
##    N = max_dset[dd[0]:dd[1]].shape[0]
##    ax[0].plot(np.arange(dd[0],dd[1],1)/fps,max_dset[dd[0]:dd[1]])
##    ax[0].set(xlabel='Time (s)',ylabel='Max Temperature (C)',title='Data Range')
##    # define the sampling data to plot it against
##    # N//2 is to only get the positive components
##    xf = np.linspace(0.0,1.0/(2.0*(1/fps)),N//2)
##    # ploot the magnitude data
##    ax[1].plot(xf,2.0/N * np.abs(ff[0:N//2]))
##    ax[1].set_xscale('log')
##    ax[1].set_yscale('log')
##    ax[1].set(xlabel='Frequency (Hz)',ylabel='Magnitude')
##    ax[1].set_title('FFT')
##    f.suptitle('Magnitude of the FFT Performed on the Max Temperature\n Data Range [{:.2f},{:.2f}]'.format(dd[0]*fps,dd[1]*fps))
##    f.tight_layout()
##    f.subplots_adjust(top=0.85)
##    f.savefig(os.path.join(folder_path,'fft-alt-max-pi-camera-d{0}-d{1}.png'.format(dd[0],dd[1])))

os.makedirs("Histograms",exist_ok=True)
def plotHist(ax,edges,pop):
    ax.bar(edges[:-1]+(edges[1]-edges[0])/2,pop,width=(edges[1]-edges[0]))

os.makedirs("Contours",exist_ok=True)
os.makedirs("LocalContours",exist_ok=True)
mmax = max_dset.max()
mmin = min_dset.min()
f,ax = plt.subplots()
with h5py.File(path,'r') as file:
    # skipping first 3 frames due to corrupted data
    print("Running histograms")
    for ff in range(3,file['pi-camera-1'][()].shape[2]):
        # get histogram of data
        pop,edges = np.histogram(file['pi-camera-1'][:,:,ff],bins=5)
        ax.clear()
        # plot data
        #plotHist(ax,edges,pop)
        ax.bar(edges[:-1]+(edges[1]-edges[0])/2,pop,width=(edges[1]-edges[0]))
        ax.set(xlabel='Temperature (C)',ylabel='Population')
        f.suptitle("Histogram of Pi Camera Temperature Frame {}".format(ff))
        f.savefig(os.path.join("Histograms","pi-camera-hist-f{}.png".format(ff)))
    plt.close(f)

##    ## contour plot
##    f,ax = plt.subplots()
##    print("Running global contours")
##    # create color bar axes
##    div = make_axes_locatable(ax)
##    cax = div.append_axes("right",size="7%",pad="2%")
##    for ff in range(3,file['pi-camera-1'][()].shape[2]):
##        # draw globally normed 
##        im= ax.contourf(file['pi-camera-1'][:,:,ff],vmax=mmax,vmin=mmin)
##        colorbar(im,cax=cax)
##        f.suptitle('Pi Camera Temperature Frame {}'.format(ff))
##        f.savefig(os.path.join("Contours","pi-camera-contour-f{}.png".format(ff)))
##    plt.close(f)
##
##    # locally normed contours
##    print("Running local contours")
##    f,ax = plt.subplots()
##    for ff in range(3,file['pi-camera-1'][()].shape[2]):
##        ax.contourf(file['pi-camera-1'][:,:,ff])
##        f.savefig(os.path.join("LocalContours","pi-camera-local-contour-f{}.png".format(ff)))
##    plt.close(f)
##    
##    f,ax = plt.subplots()
##    print("Running histograms+ct compare")
##    for ff in range(3,file['pi-camera-1'][()].shape[2]):
##        for aa in ax:
##            aa.clear()
##        ax[0].bar(edges[:-1]+(edges[1]-edges[0])/2,pop,width=(edges[1]-edges[0]))
##        ax[0].set(xlabel='Temperature (C)',ylabel='Population',title='Histogram, Frame {}'.format(ff))
##        ax[1].contourf(file['pi-camera-1'][:,:,ff],vmax=mmax,vmin=mmin)
##        f.savefig(os.path.join("Histograms","pi-camera-hist-ct-f{}.png".format(ff)))
        
# clear the figures from memory
plt.close(f)

print(tt)

## animation
f = plt.figure(constrained_layout=True)
# create a 2x2 array for subplots
gs = f.add_gridspec(2,2)
# data plot that covers the top row spanning two columns
data_plot = f.add_subplot(gs[0,:])
data_plot.set_title('Data')
# contour plot for bottom left
global_ct = f.add_subplot(gs[1,0])
#global_ct.set_title('Globally Normed CT')
# grayscale imaage
gscale = f.add_subplot(gs[1,1])
#gscale.set_title('Grayscale (Locally Normed)')

with h5py.File(path,'r') as file:
    def init():
        ## initialize plots
        # line plot
        data_plot.plot(time_set,max_dset)
        data_plot.set(xlabel='Time (s)',ylabel='Max Temperature (C)')
        # moving dot representing current frame
        dp = data_plot(time_set[0],max_dset[0],'ro',markersize=5)
        # contour plot
        global_ct.contourf(file['pi-camera-1'][:,:,0],vmax=mmax,vmin=mmin)
        # grayscale plot
        gscale.contourf(file['pi-camera-1'][:,:,0],cmap='gray')

    def update(ff):
        # update moving data point on data graph
        dp.set_data(time_set[ff],max_dset[ff])
        global_ct.contourf(file['pi-camera-1'][:,:,ff],vmax=mmax,vmin=mmin)
        gscale.contourf(file['pi-camera-1'][:,:,ff],cmap='gray')
        return dp,

    # create and save animation
    print("Saving animation")
    ani = animation.FuncAnimation(f,update,max_dset.shape[0],init_func=init,interval=1,blit=False)
    Writer = animation.writers['ffmpeg']
    fps = 1100
    writer = Writer(fps=fps,metadata=dict(artist='DBM'),bitrate=-1)
    ani.save('pi-camera-compare-timelapse.mp4',writer=writer)
    
