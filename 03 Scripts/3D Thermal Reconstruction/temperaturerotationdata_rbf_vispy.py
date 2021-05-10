import numpy as np
import sys
import h5py
import numpy as np
import cv2
from scipy.interpolate import Rbf
from numba import jit,cuda
from vispy import app, visuals, scene
import vispy.color
from scipy.spatial.transform import Rotation as R

# build your visuals, that's all
Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)

# The real-things : plot using scene
# build canvas
canvas = scene.SceneCanvas(keys='interactive', show=True)

# Add a ViewBox to let the user zoom/rotate
view = canvas.central_widget.add_view()
view.camera = 'arcball'
view.camera.fov = 45
view.camera.distance = 0

path = "D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20.hdf5"
def findBB(frame):
    from skimage.filters import sobel,threshold_otsu
    # perform sobel operation to find edges on frame
    # assumes frame is normalized
    sb = sobel(frame)
    # clear values outside of known range of target area
    sb[:,:15] = 0
    sb[:,25:] = 0
    # get otsu threshold value
    thresh = threshold_otsu(sb)
    # create mask for thresholded values
    img = (sb > thresh).astype('uint8')*255
    # perform morph open to try and close gaps and create a more inclusive mask
    img=cv2.morphologyEx(img,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    # search for contours in the thresholded image
    ct = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]
    # if a contour was found
    if len(ct)>0:
        # if there is more than one contour, sort contours by size
        if len(ct)>1:
            ct.sort(key=lambda x : cv2.contourArea(x),reverse=True)
        # return the bounding box for the largest contour and the largest contour
        return cv2.boundingRect(ct[0]),ct[0]

starter_f = 91399
# get frame
print("Getting data")
with h5py.File(path,'r') as file:
    frame = file['pi-camera-1'][:,:,starter_f]
print("Normalizing data")
# normalize
frame_norm = (frame-frame.min())/(frame.max()-frame.min())
frame_norm *= 255
frame_norm = frame_norm.astype('uint8')
frame_norm = np.dstack((frame_norm,frame_norm,frame_norm))
print("Getting bounding box and data")
# find bounding box
bb,ct = findBB(frame)
# get values within bounding box
img = frame[bb[1]:bb[1]+bb[3],bb[0]:bb[0]+bb[2]]
img_c = img.copy()
# get all cols on second half
img_c = img_c[img_c.shape[0]//2:,:]
# get location of values above 20.0
temp_tol = 20.0
xx,yy = np.where(img_c>temp_tol)
# get temperature values for marked areas
zz = img_c[xx,yy]

rr = Rbf(xx,yy,zz)
# generate a meshgrid for the inbetween values
interp_shape = xx.shape[0]*2
XX,YY = np.meshgrid(np.linspace(xx.min(),xx.max(),interp_shape),np.linspace(yy.min(),yy.max(),interp_shape))
# use rbf to predict temperature across the entire x,y range
zz = rr(XX,YY)

# convert coordinate meshgrids to 1d coordinate vectors and make a copy to update
xxr = XX.ravel()
yyr = YY.ravel()
# as the data is in a X-Y plane the z coordinate is 0.0
zzr = np.zeros(xxr.shape,xxr.dtype)
# make a copy of the temperature prediction to update
T= zz.ravel()
Tmin = T.min()
Tmax = T.max()

@jit(parallel=True,forceobj=True)
def rotate_data(xxr,yyr,zzr,T,rot_res=1.0):
    # form coordinate data set, xyz
    data  = np.array([[x,y,z] for x,y,z in zip(xxr,yyr,zzr)])
    ## rotate the dataset around the y axis
    for r in np.arange(0.0,360.0,rot_res):
        # generate rotation matrix for y rotation
        rot = np.array([[np.cos(180.0*(np.pi**-1)*r),    0,      np.sin(180.0*(np.pi**-1)*r)  ],
                    [0,                     1,      0                   ],
                    [-np.sin(180.0*(np.pi**-1)*r),   0,      np.cos(180.0*(np.pi**-1)*r)  ]
                    ])
        # apply rotation matrix
        vv = np.dot(data,rot.T)
        # udpate position matrix
        xxr = np.concatenate((xxr,vv[:,0]))
        yyr = np.concatenate((yyr,vv[:,1]))
        zzr = np.concatenate((zzr,vv[:,2]))
        # add the temperature values onto the end
        T = np.concatenate((T,zz.ravel()))
    return xxr,yyr,zzr,T

xxr,yyr,zzr,T = rotate_data(xxr,yyr,zzr,T)

# convert data to array
pos = np.array([[x,y,z] for x,y,z in zip(xxr,yyr,zzr)])
# choose colormap
cm = vispy.color.get_colormap('hot')
# generate colors
colors = cm[(T-Tmin)*(Tmax-Tmin)**-1]

# plot ! note the parent parameter
p1 = Scatter3D(parent=view.scene)
p1.set_gl_state('translucent', blend=True, depth_test=True)
p1.set_data(pos, face_color=colors, symbol='o', size=10,
edge_width=0.5, edge_color='blue')

if __name__ == '__main__':
    app.run()
