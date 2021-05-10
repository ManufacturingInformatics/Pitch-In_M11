import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.interpolate
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from skimage.filters import sobel,threshold_otsu
import cv2
from scipy.spatial.transform import Rotation as R
from matplotlib.tri.triangulation import Triangulation as mtri

ff = 93199

path = "D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20.hdf5"

# get frame
with h5py.File(path,'r') as file:
    frame = file['pi-camera-1'][:,:,ff]

np.nan_to_num(frame,copy=False)
x,y = np.meshgrid(np.arange(0.0,frame.shape[0]),np.arange(0.0,frame.shape[1]))

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

def getDataInCT(frame,ct):
    mask = np.zeros(frame.shape,np.uint8)
    mask = cv2.drawContours(mask,[ct],0,1,-1)
    return frame*mask

bb,ct = findBB(frame)
# get data within bounding box
img = frame[bb[1]:bb[1]+bb[3],bb[0]:bb[0]+bb[2]]

img_ct = getDataInCT(frame,ct)
xx,yy = np.where(img_ct)
zz = img_ct[xx,yy]
ct_data = np.array([[aa,bb,zz] for aa,bb,zz in zip(xx,yy,zz)])
aa,bb = np.meshgrid(np.arange(0,ct_data.shape[0]),np.arange(0,ct_data.shape[1]))

##rot = R.from_euler('xyz',[0.0,90.0,0.0],degrees=True)
##rr = rot.apply(ct_data)
##xx = np.concatenate((xx,rr[:,0]))
##yy = np.concatenate((yy,rr[:,1]))
##zz = np.concatenate((zz,rr[:,2]-20.0))
##
##rot = R.from_euler('xyz',[0.0,180.0,0.0],degrees=True)
##rr = rot.apply(ct_data)
##xx = np.concatenate((xx,rr[:,0]))
##yy = np.concatenate((yy,rr[:,1]))
##zz = np.concatenate((zz,rr[:,2]-20.0))
##
##rot = R.from_euler('xyz',[0.0,270.0,0.0],degrees=True)
##rr = rot.apply(ct_data)
##xx = np.concatenate((xx,rr[:,0]))
##yy = np.concatenate((yy,rr[:,1]))
##zz = np.concatenate((zz,rr[:,2]-20.0))

## mirror
zz -= zz.min()
xx = np.concatenate((xx,xx))
yy = np.concatenate((yy,yy))
zz = np.concatenate((zz,-zz))
tri = mtri(xx,yy)

f = plt.figure()
ax = f.add_subplot(111,projection='3d')
ax.scatter3D(xx,yy,zz,c='r',depthshade=False)
ax.plot_trisurf(tri,zz)
plt.show()
