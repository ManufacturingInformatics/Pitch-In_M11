import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider
from skimage.filters import sobel,threshold_otsu
from skimage.feature import canny
import cv2
from scipy.spatial.distance import cdist

from scipy.spatial.transform import Rotation as R

path = "D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20.hdf5"

start_idx = 128983

# construct rotation matrix
f = plt.figure()
aximg = f.add_subplot(131)
ax = f.add_subplot(132,projection='3d')
axsf = f.add_subplot(133,projection='3d')

with h5py.File(path,'r') as file:
    w,hh,d = file['pi-camera-1'].shape
    print("Collecting limits")
    max_dset = file['pi-camera-1'][()].max((0,1))
    min_dset = file['pi-camera-1'][()].min((0,1))
    np.nan_to_num(max_dset,copy=False)
    np.nan_to_num(min_dset,copy=False)
    zmax = max_dset[max_dset<700.0].max()
    # get the max of the first half of the dataset
    zmin = min_dset.min()
    print("Constructing meshgrid")
    x,y = np.meshgrid(np.arange(0.0,hh,1.0),np.arange(0.0,w,1.0))
    
def findDrawCt(img):
    ct = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]
    ct.sort(key=lambda x : cv2.contourArea(x),reverse=True)
    mask = np.dstack((img,img,img))
    return cv2.drawContours(mask,ct,0,(0,255,0),1),ct[0]

def findDrawBB(img,ct):
    bb = cv2.boundingRect(ct)
    if len(img.shape)<3:
        mask = np.dstack((img,img,img))
    else:
        mask = img.copy()
    return cv2.rectangle(mask,(int(bb[0]),int(bb[1])),(int(bb[0]+bb[2]),int(bb[1]+bb[3])),(255,0,0),1),bb

axidx = f.add_axes([0.25, 0.01, 0.65, 0.03])
sindex = Slider(axidx,'Index',1,int((d-1)),valinit=0,dragging=True,valstep=1,valfmt='%0.0f')

mask = np.zeros((w,hh),'float32')

uu = None
def update(val):
    ax.clear()
    axsf.clear()
    h = sindex.val
    with h5py.File(path,'r') as file:
        # get starting temperature frame
        temp = file['pi-camera-1'][:,:,h]
        # normalize locally
        norm_local = (temp-temp.min())/(temp.max()-temp.min())
        # normalize globally
        norm_global = (temp-zmin)/(zmax-zmin)
    aximg.imshow(norm_global,cmap='gray')
    sb = sobel(norm_global)
    sb[:,:15] = 0
    sb[:,25:] = 0
    thresh = threshold_otsu(sb)
    img = (sb > thresh).astype('uint8')*255
    img=cv2.morphologyEx(img,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    img,ct = findDrawCt(img)
    img,bb = findDrawBB(img,ct)

    # extract temperature values
    mask[...]=0.0
    # image space is opposite coordinates of numpy space
    mask[bb[1]:bb[1]+bb[3],bb[0]:bb[0]+bb[2]] = temp[bb[1]:bb[1]+bb[3],bb[0]:bb[0]+bb[2]]
    # find where not zero
    x,y = np.where(mask>0.0)
    z = mask[x,y]
    # convert to set of vectors
    vv = np.array([[ii,jj,tt] for ii,jj,tt in zip(x,y,z)])
    dd = np.diag(cdist(np.array([[0.0,0.0,0.0]]*len(vv)),vv))
    # get maximum
    dmax = dd.max()
    # find points whose distance is within 80% of max
    mm = np.where(dd>=0.5*dmax)
    pp = None
    for theta in np.arange(0.0,360.0,5.0):
        rot = R.from_euler('xyz',[0.0,theta,0.0],degrees=True)
        uu = rot.apply(vv)
        if pp is None:
            pp = uu.copy()
        else:
            #print(pp.shape,uu[mm,:].shape)
            pp = np.concatenate((pp,uu[mm,:][0]),0)
            
        # get distance of each point from centre
        ax.scatter(uu[mm,0],uu[mm,1],uu[mm,2],c='b')

    #axsf.plot_trisurf(pp[mm,0][0],pp[mm,1][0],pp[mm,2][0])
    axsf.plot_trisurf(pp[:,0],pp[:,1],pp[:,2])
    #ax.set_zlim(bottom=0.0)
    #axsf.set_zlim(bottom=0.0)

sindex.on_changed(update)
plt.show()
