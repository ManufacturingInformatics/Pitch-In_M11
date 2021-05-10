import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.interpolate
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider,Button
from skimage.filters import sobel,threshold_otsu
from scipy.spatial.transform import Rotation as R
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

# 3d rotation of unfiltered data
f = plt.figure()
f.suptitle("Estimated Temperature Volume")
ax = f.add_subplot(131,projection='3d')
ax.set(xlabel='X',ylabel='Y',zlabel='Z')
ax.view_init(-84,-90)
rotscat = ax.scatter3D([],[],[])
axorig = f.add_subplot(132)
axorig.set(xlabel='X',ylabel='Y',title="Data Inside Box")
aximg = f.add_subplot(133)
aximg.set_title("Original Image")
# add the axes that would need clearing to a list
fax_clear = [ax,axorig]

# added slider axes
axidx = f.add_axes([0.25, 0.01, 0.65, 0.03])
sindex = Slider(axidx,'Index',starter_f,150000,valinit=starter_f,dragging=True,valstep=1,valfmt='%0.0f')
# adding a button to reset the view of the rotated scatter points
axreset = f.add_axes([0.02, 0.01, 0.1, 0.035])
rview = Button(axreset,"RESET")
rview.on_clicked(lambda x : ax.view_init(-84,-90))

# edge detection 
f2,ax2 = plt.subplots()
f2.suptitle("Temperature Edge Values")
fax_clear.append(ax2)

## rotation of edge values
f3 = plt.figure()
f3.suptitle("Rotated Edge Values")
axedge = f3.add_subplot(121,projection='3d')
axedge.set(xlabel='X',ylabel='Y',zlabel='Z')
axorigedge = f3.add_subplot(122)
axorigedge.set(xlabel='X',ylabel='Y')

fax_clear.append(axedge)
fax_clear.append(axorigedge)

# tolerance used for temperature filtering
temp_tol = 20.0

def update(val):
    # get wanted frame index
    ff = sindex.val
    # clear required axes
    for aa in fax_clear:
        aa.clear()
    # get frame
    with h5py.File(path,'r') as file:
        frame = file['pi-camera-1'][:,:,ff]
    # convert data to 8-bit image
    frame_norm = (frame-frame.min())/(frame.max()-frame.min())
    frame_norm *= 255
    frame_norm = frame_norm.astype('uint8')
    frame_norm = np.dstack((frame_norm,frame_norm,frame_norm))
    # find bounding box
    bb,ct = findBB(frame)
    # draw rectangle on frame
    cv2.rectangle(frame_norm,(bb[0],bb[1]),(bb[0]+bb[2],bb[1]+bb[3]),(255,0,0),1)
    # rotate 90 degs clockwise
    frame_norm = cv2.rotate(frame_norm,cv2.ROTATE_90_CLOCKWISE)
    # show image with bounding box
    aximg.imshow(frame_norm,cmap='gray')
    # get values within bounding box
    img = frame[bb[1]:bb[1]+bb[3],bb[0]:bb[0]+bb[2]]
    img_c = img.copy()
    # get all cols on second half
    img_c = img_c[img_c.shape[0]//2:,:]
    # get location of values above 20.0
    xx,yy = np.where(img_c>temp_tol)
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

    # plot the original data frame as a reference
    axorig.scatter(xx,yy,c=zz)
    axorig.set_title("Bounding Box Values >{}($^\circ$C),Half Set".format(temp_tol))
    # invert the y axis to "flip" the data so that the top of the data matches the
    # top of the reference image and dataset
    axorig.invert_yaxis()
    
    # plot the rotated dataset where color indicates temperature
    ax.scatter3D(xxr,yyr,zzr,c=dd,depthshade=False)
    ax.set_title("Temperature Vol from Half Set")

    #print(bb[2:])
    ## get edge values for the half dataset
    edge = []
    # for each of the unique column values
    for c in set(yy):
        # get the highest row value, i.e. location of edge value
        edge.append(xx[yy==c].max()+bb[3]//2)
    
    # generate plot
    # plot halved dataset
    ax2.scatter(*np.where(img>temp_tol),c=img[img>temp_tol])
    # plot the edge values
    # x values are shifted so they lie on the right side of the 
    ax2.scatter(edge,list(set(yy)),marker='o',c='r')
    ax2.legend(["BB >{}($^\circ$C)".format(temp_tol),"Edge Values"])

    ## rotate the edge values to form a new shape
    # create new initial arrays
    #print(edge,len(edge))
    #print(xx,xx.shape)
    xx = np.array(edge)
    yy = np.unique(yy)
    zz = img[xx,yy]
    # create copies to update
    dd = zz.copy()
    xxr = xx.copy()
    yyr = yy.copy()
    zzr = np.zeros(xxr.shape,xxr.dtype)
    # compile data into xyz array
    data  = np.array([[x,y,0.0] for x,y in zip(xx,yy)])
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
    # plot original image for reference
    axorigedge.scatter(*np.where(img>temp_tol),c=img[img>temp_tol])
    # plot the rotated edge data
    axedge.scatter3D(xxr,yyr,zzr,c=dd,depthshade=False)
    # update the other canvases
    f2.canvas.draw()
    f3.canvas.draw()

# assign update parameter to 
sindex.on_changed(update)
plt.show()
