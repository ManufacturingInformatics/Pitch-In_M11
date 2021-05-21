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

def getSetLimits(pp="D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20.hdf5"):
    with h5py.File(pp,'r') as file:
        return np.nanmax(file['pi-camera-1'][()],axis=(0,1,2)),np.nanmin(file['pi-camera-1'][()],axis=(0,1,2))

def getSetLimitsRange(l,h,pp="D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20.hdf5"):
    with h5py.File(pp,'r') as file:
        return np.nanmax(file['pi-camera-1'][:,:,l:h],axis=(0,1,2)),np.nanmin(file['pi-camera-1'][:,:,l:h],axis=(0,1,2))

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
    try:
        rr = Rbf(xx,yy,zz)
    except:
        print("Error! Skipping frame {}".format(ff))
        return np.empty(0),np.empty(0),np.empty(0),np.empty(0)
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

print("Getting set limits")

# starting index
starter_f = 96316
# ending index
end_f = 150000

mmax,mmin = getSetLimitsRange(starter_f,end_f)
# create directory
os.makedirs("RotationHistogram",exist_ok=True)

f,ax = plt.subplots()
print("Starting histogram run")
for ff in range(starter_f,end_f,1):
    # collect data
    _,_,_,T =getData(ff)
    ax.clear()
    # perform histogram
    pop,edges = np.histogram(T,bins=20,range=[mmin,mmax])
    ax.bar(edges[:-1]+(edges[1]-edges[0])/2,pop,width=(edges[1]-edges[0]))
    ax.set_xlim(edges.min(),edges.max())
    ax.set(xlabel='Temperature ($^\circ$C)',ylabel='Population') 
    ax.set_title('Histogram of Rotated Temperature Values, {}'.format(ff))
    f.savefig("RotationHistogram/rotate-T-hist-f{}.png".format(ff))
