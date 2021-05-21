import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.interpolate
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from skimage.filters import sobel,threshold_otsu
import cv2
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import Rbf
import os

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
    else:
        return -1,-1

def getDataInCT(frame,ct):
    mask = np.zeros(frame.shape,np.uint8)
    mask = cv2.drawContours(mask,[ct],0,1,-1)
    return frame*mask

def getData(ff,rot_res=10.0,pp="D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20.hdf5"):
    with h5py.File(pp,'r') as file:
        frame = file['pi-camera-1'][:,:,ff]
    np.nan_to_num(frame,copy=False)
    # find bounding box and contour used to generate it
    bb,ct = findBB(frame)
    if (bb == -1) or (bb is None):
        print("Failed to find contours or bounding box")
        return np.empty(0),np.empty(0),np.empty(0),np.empty(0)
    # get the data masked to inside the contour
    #img_ct = getDataInCT(frame,ct)
    img = frame[bb[1]:bb[1]+bb[3],bb[0]:bb[0]+bb[2]]
    img_c = img.copy()
    # get all cols on second half
    img_c = img_c[img_c.shape[0]//2:,:]
    # get location of values above 20.0
    xx,yy = np.where(img_c>20.0)
    # get temperature values for marked areas
    zz = img_c[xx,yy]

    # try and form RBF, doesn't always work hence the except statement
    # if it fails bail early, result is a blank frame being displayed
    rr = Rbf(xx,yy,zz)
        
    interp_shape = xx.shape[0]*2
    XX,YY = np.meshgrid(np.linspace(xx.min(),xx.max(),interp_shape),np.linspace(yy.min(),yy.max(),interp_shape))
    zz = rr(XX,YY)

    # create copies to update
    dd = zz.ravel()
    zz = zz.ravel()
    xxr = XX.ravel()
    yyr = YY.ravel()
    zzr = np.zeros(xxr.shape,xxr.dtype)
    # compile data into xyz array
    data  = np.array([[x,y,0.0] for x,y in zip(xxr,yyr)])
    # rotate according to set resolution
    for r in np.arange(0.0,360.0,rot_res):
        # generate rotation object
        rot = R.from_euler('xyz',[0.0,r,0.0],degrees=True)
        # apply to matrix
        vv = rot.apply(data)
        # udpate position matrix
        xxr = np.concatenate((xxr,vv[:,0]))
        yyr = np.concatenate((yyr,vv[:,1]))
        zzr = np.concatenate((zzr,vv[:,2]))
        dd = np.concatenate((dd,zz))
    return xxr,yyr,zzr,dd

from scipy.spatial import cKDTree
def getEdgeDataKD(x,y,z,T,metric="avg"):
    # construct tree from xyz points
    tree = cKDTree(np.array([[xx,yy,zz] for xx,yy,zz in zip(x,y,z)]))
    # tree temperature edge
    Tedge = []
    # iterate over data
    for ii in range(0,tree.data.shape[0]):
        # get the four closest neighbours inc the point itself
        dist,nn = tree.query(tree.data[ii,:],k=4)
        if metric=="avg":
        # get the average difference in neighbouring temperature values
            Tedge.append(np.mean(np.diff(T[nn])))
        elif metric=="max":
            Tedge.append(np.max(np.diff(T[nn])))
        elif metric=="min":
            Tedge.append(np.min(np.diff(T[nn])))
    # return temperature edge values vector
    return Tedge

starter_f = 91399
endf = 150000
# collect data
x,y,z,T =getData(starter_f)
# get limits
mmin,mmax = np.nanmin(T),np.nanmax(T)
# downsample rate
r=2
# generate downsampled coordinate matrix
#data = np.vstack(np.meshgrid(x[::r],y[::r],z[::r],T[::r])).reshape(4,-1).T

f = plt.figure()
ax = np.empty((2,3),dtype='object')
gs = f.add_gridspec(2,3)
# add plots for histogram
ax[0,0] = f.add_subplot(gs[0,0])
ax[0,1] = f.add_subplot(gs[0,1])
ax[0,2] = f.add_subplot(gs[0,2])
# add plots for scatter
ax[1,0] = f.add_subplot(gs[1,0],projection='3d')
ax[1,1] = f.add_subplot(gs[1,1],projection='3d')
ax[1,2] = f.add_subplot(gs[1,2],projection='3d')
# create folder for results
os.makedirs("KDTreeEdge\Raw",exist_ok=True)
for ff in range(starter_f,endf,1):
    for aa in ax.ravel():
        aa.clear()
    x,y,z,T =getData(ff)
    if x.shape[0]==0:
        continue
    # get edge values and plot the histogram of the values
    Te = getEdgeDataKD(x,y,z,T,metric="avg")
    pop,edges = np.histogram(Te,bins=5,range=[min(Te),max(Te)])
    ax[0,0].bar(edges[:-1]+(edges[1]-edges[0])/2,pop,width=(edges[1]-edges[0]))
    ax[0,0].set_xlim(edges.min(),edges.max())
    ax[0,0].set(xlabel='Temperature ($^\circ$C)',ylabel='Population') 
    ax[0,0].set_title('Avg. Edge Values by cKDTree, {}'.format(ff))

    ax[1,0].scatter(x,y,z,c=Te)

    # get edge values and plot the histogram of the values
    Te = getEdgeDataKD(x,y,z,T,metric="max")
    pop,edges = np.histogram(Te,bins=5,range=[min(Te),max(Te)])
    ax[0,1].bar(edges[:-1]+(edges[1]-edges[0])/2,pop,width=(edges[1]-edges[0]))
    ax[0,1].set_xlim(edges.min(),edges.max())
    ax[0,1].set(xlabel='Temperature ($^\circ$C)',ylabel='Population') 
    ax[0,1].set_title('Max Edge Values by cKDTree, {}'.format(ff))

    ax[1,1].scatter(x,y,z,c=Te)

    # get edge values and plot the histogram of the values
    Te = getEdgeDataKD(x,y,z,T,metric="min")
    pop,edges = np.histogram(Te,bins=5,range=[min(Te),max(Te)])
    ax[0,2].bar(edges[:-1]+(edges[1]-edges[0])/2,pop,width=(edges[1]-edges[0]))
    ax[0,2].set_xlim(edges.min(),edges.max())
    ax[0,2].set(xlabel='Temperature ($^\circ$C)',ylabel='Population') 
    ax[0,2].set_title('Min Edge Values by cKDTree, {}'.format(ff))

    ax[1,2].scatter(x,y,z,c=Te)

    f.savefig("KDTreeEdge\Raw\kdtree-edge-min-max-avg-f{}.png".format(ff))


