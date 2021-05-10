import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider
from skimage.filters import sobel,laplace
from skimage.feature import canny
import cv2

f = plt.figure(constrained_layout=True)
gs = f.add_gridspec(3,3)
# axes to draw the surface
#ax = f.add_subplot(131,projection='3d')

## plot different edge detection methods
axsb = f.add_subplot(gs[0,0],projection='3d')
axlp = f.add_subplot(gs[1,0],projection='3d')
axcn = f.add_subplot(gs[2,0],projection='3d')
# axes to display cropped image with found contour points
#aximg = f.add_subplot(132)
aximgsb = f.add_subplot(gs[0,1])
aximglp = f.add_subplot(gs[1,1])
aximgcn = f.add_subplot(gs[2,1])

aximg = f.add_subplot(gs[2,2])
aximg.set_title('Locally Normed')
axsf = f.add_subplot(gs[:-1,2],projection='3d')
# contour drawn on blank canvas
##axct = f.add_subplot(133)
#axct = f.add_subplot(gs[:,2])

start_idx = 99402
crop_r = [15,25]

path = "D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20.hdf5"

def prepdata(frame):
    frame[frame<0] = 0.0
    frame[frame>300.0]=300.0
    return frame

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
    temp = file['pi-camera-1'][:,:,start_idx]
    img = (file['pi-camera-1'][:,:,start_idx]-file['pi-camera-1'][:,:,start_idx].min())/(file['pi-camera-1'][:,:,start_idx].max()-file['pi-camera-1'][:,:,start_idx].min())
    diff = (file['pi-camera-1'][:,:,start_idx]-zmin)/(zmax-zmin)

# plot initial surfaces    
axsb.plot_surface(x,y,diff,cmap='magma')
axlp.plot_surface(x,y,diff,cmap='magma')
axcn.plot_surface(x,y,diff,cmap='magma')
axsf.plot_surface(x,y,temp,cmap='magma')

# plot initial images
mask = img[:,crop_r[0]:crop_r[1]]
aximgsb.imshow(mask,cmap='gray')
aximglp.imshow(mask,cmap='gray')
aximgcn.imshow(mask,cmap='gray')

# change x tick labels to better indicate range
aximgsb.set_xticklabels([str(xx) for xx in range(crop_r[0],crop_r[1],1)])
aximglp.set_xticklabels([str(xx) for xx in range(crop_r[0],crop_r[1],1)])
aximgcn.set_xticklabels([str(xx) for xx in range(crop_r[0],crop_r[1],1)])

# show image as is
aximg.imshow(diff,cmap='gray')

#axct.imshow(np.zeros(diff.shape))
tsb = aximgsb.text(0,1,"xr={} yr={}px".format(0,0),bbox={'facecolor': 'red', 'alpha': 1.0})
tlp = aximglp.text(0,1,"xr={} yr={}px".format(0,0),bbox={'facecolor': 'red', 'alpha': 1.0})
tcn = aximgcn.text(0,1,"xr={} yr={}px".format(0,0),bbox={'facecolor': 'red', 'alpha': 1.0})

print("Setting up slider")
axidx = f.add_axes([0.25, 0.01, 0.65, 0.03])
sindex = Slider(axidx,'Index',1,int((d-1)),valinit=start_idx,dragging=True,valstep=1,valfmt='%0.0f')

def drawct(i,j,diff,c=(0,255,0)):
    ct = np.array([ [[jj,ii]] for ii,jj in zip(i,j)])
    diff = np.dstack((diff,diff,diff))
    diff = (diff*255).astype('uint8')
    diff = cv2.drawContours(diff,[ct],0,c,1)
    return diff
    
def update(val):
    # clear axes
    axsb.clear()
    axcn.clear()
    axlp.clear()
    axsf.clear()
    # get slider index
    h = sindex.val
    with h5py.File(path,'r') as file:
        temp = file['pi-camera-1'][:,:,h]
        img = (file['pi-camera-1'][:,:,h]-file['pi-camera-1'][:,:,h].min())/(file['pi-camera-1'][:,:,h].max()-file['pi-camera-1'][:,:,h].min())
        diff = (file['pi-camera-1'][:,:,h]-zmin)/(zmax-zmin)

    axsf.plot_surface(x,y,temp,cmap='magma')
    axsf.set_title('Temperature')
    
    # plot edge surfaces
    sb = sobel(diff)
    lp = laplace(diff)
    cn = canny(diff)
    axsb.plot_surface(x,y,sb,cmap='magma')
    axlp.plot_surface(x,y,lp,cmap='magma')
    axcn.plot_surface(x,y,cn,cmap='magma')

    axsb.set_title('Sobel (Cropped)')
    axlp.set_title('Laplace (Cropped)')
    axcn.set_title('Canny (Cropped)')
    
    ## search for interesting edges according to search condition
    # sobel
    #i,j = np.where(sb>=0.015)
    sb_m = np.zeros(sb.shape,sb.dtype)
    sb_m[:,crop_r[0]:crop_r[1]]=sb[:,crop_r[0]:crop_r[1]]
    i,j = np.where((sb_m>3.0*sb.mean()) & (sb_m>=0.015))

    if (i.shape[0]>0) and (j.shape[0]>0):
        sb_ct = drawct(i,j,img)
        tsb.set_text("xr={} yr={}px".format(i.max()-i.min(),j.max()-j.min()))
        aximgsb.imshow(cv2.rotate(sb_ct[:,crop_r[0]:crop_r[1]],cv2.ROTATE_90_CLOCKWISE),cmap='gray')
    else:
        tsb.set_text("0 0px")
        aximgsb.imshow(cv2.rotate(img[:,crop_r[0]:crop_r[1]],cv2.ROTATE_90_CLOCKWISE),cmap='gray')
        
    # laplacian
    lp_m = np.zeros(lp.shape,lp.dtype)
    lp_m[:,crop_r[0]:crop_r[1]]=lp[:,crop_r[0]:crop_r[1]]
    il,jl = np.where((lp_m>3.0*lp.mean())*(lp_m>0.05))

    if (il.shape[0]>0) and (jl.shape[0]>0):
        lp_ct = drawct(il,jl,img)
        tlp.set_text("xr={} yr={}px".format(il.max()-il.min(),jl.max()-jl.min()))
        aximglp.imshow(cv2.rotate(lp_ct[:,crop_r[0]:crop_r[1]],cv2.ROTATE_90_CLOCKWISE),cmap='gray')
    else:
        tlp.set_text("0 0px")
        aximglp.imshow(cv2.rotate(img[:,crop_r[0]:crop_r[1]],cv2.ROTATE_90_CLOCKWISE),cmap='gray')

    # canny
    ic,jc = np.where(cn>0.01)

    if (ic.shape[0]>0) and (jc.shape[0]>0):
        cn_ct = drawct(ic,jc,img)
        tcn.set_text("xr={} yr={}px".format(ic.max()-ic.min(),jc.max()-jc.min()))
        aximgcn.imshow(cv2.rotate(cn_ct[:,crop_r[0]:crop_r[1]],cv2.ROTATE_90_CLOCKWISE),cmap='gray')
    else:
        tcn.set_text("0 0px")
        aximgcn.imshow(cv2.rotate(img[:,crop_r[0]:crop_r[1]],cv2.ROTATE_90_CLOCKWISE),cmap='gray')

    aximg.imshow(cv2.rotate(diff,cv2.ROTATE_90_CLOCKWISE),cmap='gray')
    
print("Showing")
sindex.on_changed(update)
update(start_idx)
plt.show()
