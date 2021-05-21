import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.interpolate
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider,Button
from skimage.filters import sobel,threshold_otsu
from scipy.spatial.transform import Rotation as R
import matplotlib.tri as mtri
from scipy.spatial import ConvexHull
from pyntcloud import PyntCloud
from skimage import measure
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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

def customCube(loc,size=(1,1,1)):
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

def genCubesVoxelID(xx,yy,zz,T):
    cubes = []
    cols = []
    cmap = cm.get_cmap('viridis')
    norm = Normalize(vmax=T.max(),vmin=T.min())
    for x,y,z,t in zip(xx,yy,zz,T):
        cubes.append(customCube((x,y,z)))
    return Poly3DCollection(np.concatenate(cubes),facecolor='b',edgecolor='k')

def fillVoxelIDUp(vi):
    vj = vi.copy()
    for x in range(0,vj.shape[0]):
        for y in range(0,vj.shape[1]):
            for z in range(0,vj.shape[2]-1):
                if vj[x,y,z+1]==1:
                    vj[x,y,z]=1
    return vj

## rotation of edge values
f3 = plt.figure()
f3.suptitle("Rotated Edge Values")
gs = gridspec.GridSpec(3,2,f3)
axedge = f3.add_subplot(gs[0,0],projection='3d')
axedge.set(xlabel='X',ylabel='Y',zlabel='Z')
axcube = f3.add_subplot(gs[1,0],projection='3d')
axvoxel = f3.add_subplot(gs[2,0],projection='3d')
axorigedge = f3.add_subplot(gs[:,1])
axorigedge.set(xlabel='X',ylabel='Y')

# added slider axes
axidx = f3.add_axes([0.25, 0.01, 0.65, 0.03])
sindex = Slider(axidx,'Index',starter_f,150000,valinit=starter_f,dragging=True,valstep=1,valfmt='%0.0f')
# adding a button to reset the view of the rotated scatter points
axreset = f3.add_axes([0.02, 0.01, 0.1, 0.035])
rview = Button(axreset,"RESET")
rview.on_clicked(lambda x : axedge.view_init(-84,-90))

# update list of axes to clean
fax_clear = [axedge,axcube,axorigedge,axvoxel]

# tolerance used for temperature filtering
temp_tol = 20.0

def update(val):
    import pandas as pd
    # get wanted frame index
    ff = sindex.val
    # clear required axes
    for aa in fax_clear:
        aa.clear()
    # get frame
    with h5py.File(path,'r') as file:
        frame = file['pi-camera-1'][:,:,ff]
    # convert data to 8-bit image
##    frame_norm = (frame-frame.min())/(frame.max()-frame.min())
##    frame_norm *= 255
##    frame_norm = frame_norm.astype('uint8')
##    frame_norm = np.dstack((frame_norm,frame_norm,frame_norm))
##    # find bounding box
    bb,ct = findBB(frame)
##    # draw rectangle on frame
##    cv2.rectangle(frame_norm,(bb[0],bb[1]),(bb[0]+bb[2],bb[1]+bb[3]),(255,0,0),1)
##    # rotate 90 degs clockwise
##    frame_norm = cv2.rotate(frame_norm,cv2.ROTATE_90_CLOCKWISE)
##    # show image with bounding box
##    aximg.imshow(frame_norm,cmap='gray')
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

    #print(bb[2:])
    ## get edge values for the half dataset
    edge = []
    # for each of the unique column values
    for c in set(yy):
        # get the highest row value, i.e. location of edge value
        edge.append(xx[yy==c].max()+bb[3]//2)

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
    # generate convex hull
    points = np.array([[x,y,z] for x,y,z in zip(xxr,yyr,zzr)])
    # generate convex hull from the edge points
    hull = ConvexHull(points,qhull_options='Qi')
    # draw lines representing the convex hull on the plot
    for simplex in hull.simplices:
        axedge.plot(points[simplex,0],points[simplex,1],points[simplex,2],'k-')

    ## marching cubes
    # convert coordinated to PyntCloud
    cloud = PyntCloud(pd.DataFrame(points,columns=['x','y','z']))
    # convert PyntCloud to voxel grid
    # n_* sets the "resolution" for the number of voxels in that respective direction
    res_inc = 10
    voxelgrid_id = cloud.add_structure("voxelgrid",n_x=int(xxr.max()-xxr.min())+res_inc,n_y=int(yyr.max()-yyr.min())+res_inc,n_z=int(zzr.max()-zzr.min())+res_inc)
    # get voxel grid
    vg = cloud.structures[voxelgrid_id].get_feature_vector()
    print(vg.shape)
    # fill gaps between voxel grid
    #vg = fillVoxelIDUp(vg)
    # give voxel grid to marching cubes algorithm 
    verts,faces,normals,values=measure.marching_cubes_lewiner(vg,0)
    # generate 3d objects to add
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    # update limits to match point other axis
    axcube.add_collection3d(mesh)
    axcube.set_xlim3d(verts[:,0].min(),verts[:,0].max())
    axcube.set_ylim3d(verts[:,1].min(),verts[:,1].max())
    axcube.set_zlim3d(verts[:,2].min(),verts[:,2].max())

    # use voxel grid to generate and add set of voxels
    axvoxel.add_collection3d(genCubesVoxelID(*np.where(vg),dd))
    axvoxel.set_xlim3d(verts[:,0].min(),verts[:,0].max())
    axvoxel.set_ylim3d(verts[:,1].min(),verts[:,1].max())
    axvoxel.set_zlim3d(verts[:,2].min(),verts[:,2].max())

        
    
# assign update parameter to 
sindex.on_changed(update)
plt.show()
