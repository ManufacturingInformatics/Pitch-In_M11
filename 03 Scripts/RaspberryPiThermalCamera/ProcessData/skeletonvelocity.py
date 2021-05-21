import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider
from skimage.filters import sobel,threshold_otsu
import cv2
import os

path = "D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20.hdf5"

os.makedirs("SkeletonEst",exist_ok=True)
f,ax = plt.subplots()
fps=31.0
t = 1.0/fps
with h5py.File(path,'r') as file:
    w,h,d = file['pi-camera-1'].shape
    ffs = list(range(91399,93000))
    bbw = np.zeros(len(ffs))
    bbh = np.zeros(len(ffs))
    vx = np.zeros(len(ffs))
    vy = np.zeros(len(ffs))
    for fi,ff in enumerate(ffs):
        # get frame
        frame = file['pi-camera-1'][:,:,ff]
        # normalize
        frame = (frame-frame.min())/(frame.max()-frame.min())
        # perform sobel operation
        sb = sobel(frame)
        # mask sobel matrix
        sb[:,:15] = 0
        sb[:,25:] = 0
        # calculate otsu threshold limit
        thresh = threshold_otsu(sb)
        # create mask of values thresholded
        img = (sb > thresh).astype('uint8')*255
        # apply opening to improve contour estimation
        img=cv2.morphologyEx(img,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
        # search for contours
        ct = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]
        
        if len(ct)>0:
            # find non-rotated bounded rectangle for contour
            bb = cv2.boundingRect(ct[0])
            # sort in descending order if necessary
            if len(ct)>1:
                ct.sort(key=lambda x : cv2.contourArea(x),reverse=True)

            mask = np.dstack((frame.astype('uint8'),frame.astype('uint8'),frame.astype('uint8')))
            # draw bounded rectangle on mask
            mask = cv2.rectangle(mask,(int(bb[0]),int(bb[1])),(int(bb[0]+bb[2]),int(bb[1]+bb[3])),(255,0,0),1)
            # log width and height of bounding box
            bbw[fi] = bb[2]
            bbh[fi] = bb[3]
            # extract x and y coordinates of largest contour
            xx = [cc[0][0] for cc in ct[0]]
            yy = [cc[0][1] for cc in ct[0]]

            ## form skeleton
            # width line
            mask = cv2.circle(mask,(min(xx),yy[xx.index(min(xx))]),2,(0,0,255),1,3)
            mask = cv2.circle(mask,(min(xx)+bb[2],yy[xx.index(min(xx))]),2,(0,0,255),1,3)
            mask = cv2.line(mask,(min(xx),yy[xx.index(min(xx))]),(min(xx)+bb[2],yy[xx.index(min(xx))]),(0,0,255),3)
            # height line
            mask = cv2.circle(mask,(xx[yy.index(min(yy))],min(yy)),2,(0,0,255),1,3)
            mask = cv2.circle(mask,(xx[yy.index(min(yy))],min(yy)+bb[3]),2,(0,0,255),1,3)
            mask = cv2.line(mask,(xx[yy.index(min(yy))],min(yy)),(xx[yy.index(min(yy))],min(yy)+bb[3]),(0,0,255),3)

            cv2.imwrite(os.path.join("SkeletonEst","skele-est-ff{}.png".format(ff)),mask)

            if fi>0:
                vx[fi] = (bb[0]-bb0[0])/t
                vy[fi] = (bb[1]-bb0[1])/t
                bb0 = bb[:2]
            else:
                bb0 = bb[:2]
        else:
            vx[fi] = vx[fi-1]
            vy[fi] = vy[fi-1]

    ax.plot([float(a)/fps for a in ffs],vx)
    ax.set(xlabel='Time(s)',ylabel='X Velocity (pixels/s)')
    f.suptitle('X Velocity of the TL Corner of the Non-Rotated\nBounded Box Calculated for Temperature Data')
    f.savefig('bounding-box-tl-vx.png')
    
    ax.clear()
    ax.plot([float(a)/fps for a in ffs],vy)
    ax.set(xlabel='Time(s)',ylabel='Y Velocity (pixels/s)')
    f.suptitle('Y Velocity of the TL Corner of the Non-Rotated\nBounded Box Calculated for Temperature Data')
    f.savefig('bounding-box-tl-vy.png')

    ax.clear()
    ax.plot([float(a)/fps for a in ffs],((vx**2.0)+(vy**2.0))**0.5)
    ax.set(xlabel='Time (s)',ylabel='Vector Velocity (pixels/s)')
    f.suptitle('Vector Velocity of the TL Corner of the Non-Rotated\nBounded Box Calculated for Temperature Data')
    f.savefig('bounding-box-tl-vv.png')

    ax.clear()
    ang = np.tan(vy/vx)
    np.nan_to_num(ang,copy=False)
    ax.plot([float(a)/fps for a in ffs],ang)
    ax.set(xlabel='Time (s)',ylabel='Vector Heading (degrees)')
    f.suptitle('Vector Velocity Heading of the TL Corner of the Non-Rotated\nBounded Box Calculated for Temperature Data')
    f.savefig('bounding-box-tl-vangle.png')
        #break
        
