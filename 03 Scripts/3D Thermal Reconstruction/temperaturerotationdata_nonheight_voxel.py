import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.interpolate
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.widgets import Slider
from skimage.filters import sobel,threshold_otsu
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import pdist
import cv2

starter_f = 91398

path = "D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20.hdf5"
    
def findBB(frame):
    sb = sobel(frame)
    sb[:,:15] = 0
    sb[:,25:] = 0
    thresh = threshold_otsu(sb)
    img = (sb > thresh).astype('uint8')*255
    img=cv2.morphologyEx(img,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    ct = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]
    if len(ct)>0:
        if len(ct)>1:
            ct.sort(key=lambda x : cv2.contourArea(x),reverse=True)
        return cv2.boundingRect(ct[0]),ct[0]

def customCube(loc,size):
    # generate a unit cube
    # each set in the list is the 4 points for a face
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]    
    X = np.array(X).astype(float)
    # increase cube to desired size
    for i in range(3):
        X[:,:,i] *= size[i]
    # shift the position of the cube so it is in the target location
    X += np.array(loc)
    return X

def genRotBlocks(x,y,z,T):
    cubes = []
    fcols = []
    #print(x.shape,y.shape,z.shape,T.shape)
    cmap = cm.get_cmap('viridis')
    tmax,tmin = T.max(),T.min()
    norm = Normalize(vmax=T.max(),vmin=T.min())
    #sz = pdist(np.array([[xx,yy,zz] for xx,yy,zz in zip(x,y,z)]))
    #sz = sz[sz>0.0].min()
    for xx,yy,zz,t in zip(x,y,z,T):
        #print(cmap(norm(t)))
        # add cube to list
        cubes.append(customCube([xx,yy,zz],[.1,.1,.1]))
        # generate color for all facecolors
        fcols.append(cmap(norm(t)))
    # create collection of 3d objects from the cubes
    return Poly3DCollection(np.concatenate(cubes),facecolor=fcols)

# 3d rotation of unfiltered data
f = plt.figure()
f.suptitle("Estimated Temperature Volume")
ax = f.add_subplot(131,projection='3d')
ax.set(xlabel='X',ylabel='Y',zlabel='Z')
ax.auto_scale_xyz([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
axorig = f.add_subplot(132)
axorig.set(xlabel='X',ylabel='Y',title="Data Inside Box")
aximg = f.add_subplot(133)
aximg.set_title("Original Image")
# add the axes that would need clearing to a list
fax_clear = [ax]

# added slider axes
axidx = f.add_axes([0.25, 0.01, 0.65, 0.03])
sindex = Slider(axidx,'Index',starter_f,150000,valinit=starter_f,dragging=True,valstep=1,valfmt='%0.0f')

def update(val):
    # get wanted frame index
    ff = sindex.val
    # clear required axes
    for aa in fax_clear:
        aa.clear()
    # get frame
    with h5py.File(path,'r') as file:
        frame = file['pi-camera-1'][:,:,ff]
    aximg.contourf(frame,cmap='gray')
    # find bounding box
    bb,ct = findBB(frame)
    # get values within bounding box
    img = frame[bb[1]:bb[1]+bb[3],bb[0]:bb[0]+bb[2]]
    img_c = img.copy()
    # get all cols on second half
    img_c = img_c[img_c.shape[0]//2:,:]
    # get location of values above 20.0
    xx,yy = np.where(img_c>20.0)
    # get temperature values for marked areas
    zz = img_c[xx,yy]
    # create copies of the data to update
    xxr = xx.copy()
    yyr = yy.copy()
    zzr = np.zeros(xxr.shape,xxr.dtype)
    dd = zz.copy()
    # form coordinate data set, xyz
    data  = np.array([[x,y,0.0] for x,y in zip(xx,yy)])
    # resolution of rotation
    rot_res = 10.0
    ## rotate the dataset around the y axis
    for r in np.arange(0.0,360.0,rot_res):
        # generate rotation object
        rot = R.from_euler('xyz',[0.0,r,0.0],degrees=True)
        # apply to matrix
        vv = rot.apply(data)
        # udpate position matrix
        xxr = np.concatenate((xxr,vv[:,0]))
        yyr = np.concatenate((yyr,vv[:,1]))
        zzr = np.concatenate((zzr,vv[:,2]))
        # add the temperature values onto the end
        dd = np.concatenate((dd,zz))

    blocks = genRotBlocks(xxr,yyr,zzr,dd)
    # plot the original data frame as a reference
    axorig.scatter(*np.where(img>20.0),c=img[img>20.0])
    # plot the rotated dataset where color indicates temperature
    #ax.scatter3D(xxr,yyr,zzr,c=dd,depthshade=False)
    #ax.add_collection3d(blocks)
    #ax.auto_scale_xyz([xxr.min()-0.1,xxr.max()+0.1],[yyr.min()-0.1,yyr.max()+0.1],[zzr.min()-0.1,zzr.max()+0.1])

    volume_shape = (xxr.max(),yyr.max(),zzr.max())
    f.canvas.draw()

# assign update parameter to 
sindex.on_changed(update)
plt.show()
