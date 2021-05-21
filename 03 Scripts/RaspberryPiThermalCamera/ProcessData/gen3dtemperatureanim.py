import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import h5py
import matplotlib.animation as anim

path = "D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20.hdf5"

with h5py.File(path,'r') as file:
    # get image sizes
    w,h,d = file['pi-camera-1'].shape
    print(w,h,d)
    max_dset = file['pi-camera-1'][()].max((0,1))
    np.nan_to_num(max_dset,copy=False)
    zmax = max_dset[max_dset<700.0].max()
    # set up figure
    f = plt.figure()
    ax = f.add_subplot(111,projection='3d')
    # setup meshgrid for surface plotting
    x,y = np.meshgrid(np.arange(0.0,h,1.0),np.arange(0.0,w,1.0))
    print(x.shape)    
    
    def update(ff):
        #print("\rff={}".format(ff,end=''))
        ax.clear()
        ax.plot_surface(x,y,file['pi-camera-1'][:,:,ff],cmap='magma')
        ax.set_zlim(top=zmax)
        ax.set(xlabel='Row',ylabel='Column',zlabel='Temperature (C)')

    ani = anim.FuncAnimation(f,update,d)
    ani.save('pi-camera-temperature-timelapse.mp4',writer='ffmpeg',fps=550)
    ani.save('pi-camera-temperature-timelapse.gif',writer='imagemagick',fps=550)
    
