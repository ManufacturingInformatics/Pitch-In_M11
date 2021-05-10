import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
from matplotlib.colors import SymLogNorm
import matplotlib.cm as cm

path = "D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20.hdf5"

# plot showing data as is, locally normed with global max
f = plt.figure()
ax = f.add_subplot(221,projection='3d')

# normal dataset, log normed
axlog = f.add_subplot(222,projection='3d')

# setting max to half value
axhalf = f.add_subplot(223,projection='3d')

# setting normed to half max
axmhalf = f.add_subplot(224,projection='3d')

ax.set_title('Locally Normed')
axlog.set_title('Locally Log Normed')
axhalf.set_title('Displayed Max Set to Half')
axmhalf.set_title('Normed Max Set to Half')

## generate the plots for the specific data ranges
data_ranges = [[47500,49500],[60000,62000],[68000,71000],[90000,93000],[96800,100750],[100750,131750]]
f_dd = []
# for each data range, generate a figure and 2 element subplot
for dd in data_ranges:
    f_dd.append(plt.figure())
    f_dd[-1].add_subplot(111,projection='3d')
    f_dd[-1].suptitle('Range: {} - {}'.format(dd[0],dd[1]))

# cut off point on dataset
cut_off = 0.5

# get size of the dataset
print("Opening file")
frames_dd = []
max_dd = []
with h5py.File(path,'r') as file:
    w,h,d = file['pi-camera-1'].shape
    # get maximum to set zlim
    print("Getting max dataset")
    max_dset = file['pi-camera-1'][()].max((0,1))
    min_dset = file['pi-camera-1'][()].min((0,1))
    np.nan_to_num(max_dset,copy=False)
    np.nan_to_num(min_dset,copy=False)
    zmax = max_dset[max_dset<700.0].max()
    # get the max of the first half of the dataset
    zhmax = max_dset[:d//2].max()
    zmin = min_dset.min()
    print("Generating meshgrid")
    x,y = np.meshgrid(np.arange(0.0,h,1.0),np.arange(0.0,w,1.0))

    frame = file['pi-camera-1'][:,:,0]
    for dd in data_ranges:
        frames_dd.append(file['pi-camera-1'][:,:,dd[0]])
        max_dd.append(max_dset[dd[0]:dd[1]].max())
    
print("Generating initial plot")
ax.plot_surface(x,y,frame,cmap='magma')
ax.set_zlim(top=zmax)

print("Generating log norm plot")
log_cmap = cm.get_cmap('magma')
log_cmap.set_bad((0,0,0))
axlog.plot_surface(x,y,frame,cmap='magma',norm=SymLogNorm(linthresh=1,vmin=0,vmax=zmax))
axlog.set_zlim(top=zmax)

print("Generating half max plot")
axhalf.plot_surface(x,y,frame,cmap='magma')
axhalf.set_zlim(top=zmax/2)

print("Generating normed half max plot")
axmhalf.plot_surface(x,y,frame,cmap='magma',vmax=zmax/2)
axmhalf.set_zlim(top=zhmax)

print("Generating data range plots")
for fi,ff in enumerate(f_dd):
    ff.axes[0].plot_surface(x,y,frames_dd[fi],cmap='magma')

print("Adding sliders")
# setup slider to change idx being displayed
axidx = f.add_axes([0.25, 0.1, 0.65, 0.03])
# add sliders
sindex = Slider(axidx,'Index',0,int((d-1)*cut_off),valinit=0,dragging=True,valstep=1,valfmt='%0.0f')

slider_dd = []
for fi,ff in enumerate(f_dd):
    axidx = ff.add_axes([0.25, 0.1, 0.65, 0.03])
    slider_dd.append(Slider(axidx,'Index',data_ranges[fi][0],data_ranges[fi][1],valinit=data_ranges[fi][0],dragging=True,valstep=1,valfmt='%0.0f'))
    
def update(*args):
    #print(args)
    # get index
    h = sindex.val
    # clear axes
    for aa in range(len(f.axes)-1):
        f.axes[aa].clear()
    # get desired frame
    with h5py.File(path,'r') as file:
        frame = file['pi-camera-1'][:,:,int(h)]
    # plot frame
    ax.plot_surface(x,y,frame,cmap='magma')
    axlog.plot_surface(x,y,frame,cmap='magma',norm=SymLogNorm(linthresh=1,vmin=0,vmax=zmax))
    axhalf.plot_surface(x,y,frame,cmap='magma')
    axmhalf.plot_surface(x,y,frame,cmap='magma',vmax=zhmax)

    # set limits
    ax.set_zlim(top=zmax)
    axlog.set_zlim(top=zmax)
    axhalf.set_zlim(top=zmax/2)
    axmhalf.set_zlim(top=zhmax)

    # set titles
    ax.set_title('Locally Normed')
    axlog.set_title('Locally Log Normed')
    axhalf.set_title('Displayed Max Set to Half')
    axmhalf.set_title('Normed Max Set to Half')

def update_dd(val):
    hh = [ss.val for ss in slider_dd]
    with h5py.File(path,'r') as file:
        for ff,h,mm in zip(f_dd,hh,max_dd):
            ff.axes[0].clear()
            ff.axes[0].plot_surface(x,y,file['pi-camera-1'][:,:,int(h)],cmap='magma',vmax=mm)
            ff.axes[0].set_zlim(top=mm)
                    
# register on change update function
sindex.on_changed(update)

for ss in slider_dd:
    ss.on_changed(update_dd)

plt.show()

    

