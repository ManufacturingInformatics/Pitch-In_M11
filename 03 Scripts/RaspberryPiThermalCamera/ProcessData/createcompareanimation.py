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

## animation
f = plt.figure(constrained_layout=True)
# create a 2x2 array for subplots
gs = f.add_gridspec(2,1)
# data plot that covers the top row spanning two columns
data_plot = f.add_subplot(gs[0,0])
data_plot.set_title('Data')
gscale = f.add_subplot(gs[1,0])
#gscale.set_title('Grayscale (Locally Normed)')
# open file
with h5py.File(path,'r') as file:
    print("Getting max datasets")
    # get min and max temperatures
    max_dset = file['pi-camera-1'][()].max((0,1))
    min_dset = file['pi-camera-1'][()].min((0,1))
    # remove nans
    np.nan_to_num(max_dset,copy=False)
    np.nan_to_num(min_dset,copy=False)
    # sort in ascending
    max_dset_sort = np.sort(max_dset)
    # find the max that is below 700
    mmax = max_dset_sort[max_dset_sort<700][-1]
    mmin = min_dset.min()
    # frames per second
    fps = 31
    # number of frames divided by fps gives the total time recorded
    total_time = max_dset.shape[0]/fps
    # create dataset
    time_set = np.arange(0,total_time,1/fps)
    # moving dot representing current frame
    dp, = data_plot.plot([],[],'ro',markersize=5)
    
    def update(ff,mmax,mmin):
        # update moving data point on data graph
        dp.set_data(time_set[ff],max_dset[ff])
        # generate contour that is normalized to the data range
        frame = file['pi-camera-1'][:,:,ff]
        gscale.contourf(np.rot90(frame),cmap='viridis',vmax=mmax,vmin=mmin)
        return dp,

    # create and save animation
    print("Saving animation")
    data_ranges = [[47500,48050],[47500,49500],[60000,60450],[60000,62000],[68000,71000],[68200,68820],[90000,93000],[96800,100750],[100750,131750],[90000,150000]]
    for dd in data_ranges:
        # set data plot for the target range
        data_plot.clear()
        data_plot.plot(time_set[dd[0]:dd[1]],max_dset[dd[0]:dd[1]],'b-')
        data_plot.set(xlabel='Time (s)',ylabel='Max Temperature (C)',title='Maximum Temperature for Range [{},{}]'.format(time_set[dd[0]],time_set[dd[1]]))
        # finding maximum and minimum for contours
        mmax = max_dset[dd[0]:dd[1]]
        np.nan_to_num(mmax,copy=True)
        mmax = mmax.max()
        mmin = mmin.min()
        print("Creating animation for range {} {}".format(dd[0],dd[1]))
        ani = animation.FuncAnimation(f,update,list(range(dd[0],dd[1],1)),fargs=(mmax,mmin),interval=1,blit=False)
        Writer = animation.writers['ffmpeg']
        # if the time period for data range is less than 120 secs, leave as is
        # else scale the fps to ensure it's 120 seconds
        if (dd[1]-dd[0])/fps >= 120:
            fps = (dd[1]-dd[0])/120
        else:
            fps = 31
        # create and set metadata for animation writer
        writer = Writer(fps=fps,metadata=dict(artist='DBM'),bitrate=-1)
        ani.save('pi-camera-compare-timelapse-{}-{}.mp4'.format(dd[0],dd[1]),writer=writer)
