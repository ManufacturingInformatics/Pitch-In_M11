import h5py
import numpy as np
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as ani
import matplotlib.pyplot as plt
# resolution for display
res = 100
print("Reading in rad head data")
local_data = np.genfromtxt("thermal-hough-circle-metrics-local.csv",
                           delimiter=',',
                           dtype='float',
                           names=True)
qr = local_data['Radiated_Heat_Times_Area_W']

# open file, cleans everything up on exit
print("Opening data file")
with h5py.File("gaussian-kde-best-circle-centre-arrow-shape.hdf5",'r') as file:
    # get reference to dataset
    dset = file['P-xy-Qr'][:,:,::res]
    rows,cols,depth = dset.shape
    # create axes
    f = plt.figure()
    ax = f.add_subplot(111,projection='3d')
    # construct meshgrid to plot results
    xx,yy,zz = np.meshgrid(range(rows),range(cols),
                           # set resolution of height
                           range(depth))
    # plot data on axes
    print("Plotting data at every {} data points".format(res))
    ax.scatter(xx,yy,zz,c=dset.flat)

    ## create animation writer
    Writer = ani.writers['ffmpeg']
    # frame rate
    mpfps = int(depth/60)
    # construct writer object
    writer = Writer(fps=mpfps,metadata=dict(artist='DBM'),bitrate=-1)

    def rotate(angle):
        ax.view_init(elev=35,azim=angle)

    print("Creating rotating animation")
    anim_obj = ani.FuncAnimation(f,rotate,frames=360,interval=10,blit=False)
    print("Saving rotating animation to file")
    anim_obj.save('kde-3d-rad-heat-best-circle-centre.avi',writer=writer)
    plt.close(f)

    ## iterate through qr data as slices
    f,ax = plt.subplots()
    # turn off axes labels
    ax.axis('off')
    # display slice data 
    def iterateSlice(ff,data):
        ax.imshow(data,cmap='hot')

    ## create animation writer
    Writer = ani.writers['ffmpeg']
    # frame rate
    mpfps = 200
    # construct writer object
    writer = Writer(fps=mpfps,metadata=dict(artist='DBM'),bitrate=-1)
    print("Creating slice animation")
    # genereate animation
    anim_obj = ani.FuncAnimation(f,iterateSlice,int(dset.shape[2]/res),fargs=dset,interval=1,blit=True)
    print("Saving slice animation")
    anim_obj.save("kde-3d-rad-heat-best-circle-centre-f" + str(mpfps)+".mp4",writer=writer)
    writer=None
