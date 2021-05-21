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

f = plt.figure()
gs = f.add_gridspec(2,3)

start_idx = 0
box_depth = 0.9

axsb = f.add_subplot(gs[0,0])
axsbgb = f.add_subplot(gs[1,0])
axth = f.add_subplot(gs[0,1])
axthgb = f.add_subplot(gs[1,1])
axsf = f.add_subplot(gs[0,2],projection='3d')
axmsf = f.add_subplot(gs[1,2])

axth.axis('off')
axthgb.axis('off')

path = "D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20.hdf5"

def findDrawCt(img):
    ct = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]
    mask = np.dstack((img,img,img))
    if len(ct)==0:
        return mask,ct
    ct.sort(key=lambda x : cv2.contourArea(x),reverse=True)
    return cv2.drawContours(mask,ct,0,(0,255,0),1),ct[0]

def findDrawBB(img,ct):
    if len(img.shape)<3:
        mask = np.dstack((img,img,img))
    else:
        mask = img.copy()
    if len(ct)==0:
        return mask
    bb = cv2.boundingRect(ct)
    return cv2.rectangle(mask,(int(bb[0]),int(bb[1])),(int(bb[0]+bb[2]),int(bb[1]+bb[3])),(255,0,0),1),bb
    

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
    print("Constructing first diff plot")
    # get starting temperature frame
    temp = file['pi-camera-1'][:,:,start_idx]
    mask_temp = temp.copy()
    mask_temp[mask_temp<18.0]=0.0
    # normalize locally
    img = (temp-temp.min())/(temp.max()-temp.min())
    # normalize globally
    diff = (temp-zmin)/(zmax-zmin)

#img = img[:,15:25]
#diff = diff[:,15:25]
# apply edge detection to normed image
sb = sobel(img)
sb_gb = sobel(diff)
# draw contour of sobel results
axsb.contourf(sb,cmap='magma')
axsbgb.contourf(sb_gb,cmap='magma')
# mask sobel range to narrow range
sb[:,:15] = 0
sb[:,25:] = 0

sb_gb[:,:15] = 0
sb_gb[:,25:] = 0
# find otsu threshold of masked sobel
#thresh = threshold_otsu(sb)
img = (sb > sb.mean()).astype('uint8')*255
img=cv2.morphologyEx(img,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
# find and draw largest contour in masked sobel image
img,ct = findDrawCt(img)
img,bb = findDrawBB(img,ct)
# display drawn contour results
axth.imshow(cv2.flip(img,0),cmap='gray')

#thresh = threshold_otsu(sb_gb)
img = (sb_gb > sb_gb.mean()).astype('uint8')*255
# find and draw largest contour in masked sobel image
img,ct = findDrawCt(img)
img,bb = findDrawBB(img,ct)
# display drawn contour results
axthgb.imshow(cv2.flip(img,0),cmap='gray')
# plot temperature surface
axsf.plot_surface(x,y,temp,cmap='magma')
#axsf.set_zlim(zmin,zmax)
#axmsf.plot_surface(x,y,mask_temp,cmap='magma')
axmsf.imshow(cv2.flip(diff,0),cmap='gray')

axidx = f.add_axes([0.25, 0.01, 0.65, 0.03])
sindex = Slider(axidx,'Index',1,int((d-1)),valinit=start_idx,dragging=True,valstep=1,valfmt='%0.0f')

def update(val):
    axsf.clear()
    axmsf.clear()
    h = sindex.val
    with h5py.File(path,'r') as file:
        temp = file['pi-camera-1'][:,:,h]
        mask_temp = temp.copy()
        mask_temp[mask_temp<18.0]=0.0
        img = (temp-temp.min())/(temp.max()-temp.min())
        diff = (temp-zmin)/(zmax-zmin)

    sb = sobel(img)
    sb_gb = sobel(diff)
    axsb.contourf(sb,cmap='magma')
    axsb.set_title('Sobel')
    axsbgb.contourf(sb_gb,cmap='magma')
    axsbgb.set_title('Sobel (Global)')
    sb[:,:15] = 0
    sb[:,25:] = 0
    sb_gb[:,:15] = 0
    sb_gb[:,25:] = 0
    #thresh = threshold_otsu(sb)
    img = (sb > sb.mean()).astype('uint8')*255
    img=cv2.morphologyEx(img,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    img,ct = findDrawCt(img)
    img,bb = findDrawBB(img,ct)
    axth.imshow(cv2.flip(img,0),cmap='gray')
    axth.set_title('Above Mean')
    
    #thresh = threshold_otsu(sb_gb)
    img = (sb_gb > sb_gb.mean()).astype('uint8')*255
    img=cv2.morphologyEx(img,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    img,ct = findDrawCt(img)
    img,bb = findDrawBB(img,ct)
    axthgb.imshow(cv2.flip(img,0),cmap='gray')
    axthgb.set_title('Above Mean (Global)')
    
    axsf.plot_surface(x,y,temp,cmap='magma')
    axsf.set_title('Surface')
    #axsf.set_zlim(zmin,zmax)
    #axmsf.plot_surface(x,y,mask_temp,cmap='magma')
    #axmsf.set_title('Masked Surface, t=20.0')
    axmsf.imshow(cv2.flip(diff,0),cmap='gray')
    axmsf.set_title('Globally Normed Temp')
sindex.on_changed(update)
plt.show()
