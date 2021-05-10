import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
import matplotlib.animation as animation

path = "D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20.hdf5"
os.makedirs("Contours",exist_ok=True)

# create plot axes
f,ax = plt.subplots()
# open file
with h5py.File(path,'r') as file:
    print("Running local contours")
    # create color bar axes
    div = make_axes_locatable(ax)
    cax = div.append_axes("right",size="7%",pad="2%")

    def update(ff):
        ax.clear()
        im= ax.contourf(file['pi-camera-1'][:,:,ff])
        #colorbar(im,cax=cax)
        #f.savefig(os.path.join("LocalContours","pi-camera-local-contour-f{}.png".format(ff)))

    ani = animation.FuncAnimation(f,update,file['pi-camera-1'][()].shape[2],blit=False,repeat=False)
    Writer = animation.writers['ffmpeg']
    fps = 1100
    writer = Writer(fps=fps,metadata=dict(artist='DBM'),bitrate=-1)
    ani.save('pi-camera-local-contour-timelapse.mp4',writer=writer)
