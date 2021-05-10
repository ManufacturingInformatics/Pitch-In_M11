import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.interpolate
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from skimage.filters import sobel,threshold_otsu
import cv2
from scipy.spatial.transform import Rotation as R
import vtk
from vtk.util import numpy_support
from vtk.numpy_interface import dataset_adapter as dsa
from vtk.numpy_interface import algorithms as alg
  
from scipy.interpolate import Rbf

# index to start on
starter_f = 93199
# initial ff
ff = starter_f
# filtering limit
filt = 0.6
# rotation resolution
rot_res = 10.0
# path to hdf5 file
path = "D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20.hdf5"

print("Collecting data limits")
with h5py.File(path,'r') as file:
    mmax = np.nanmax(file['pi-camera-1'][:,:,starter_f:150000],axis=(0,1,2))
    mmin = np.nanmin(file['pi-camera-1'][:,:,starter_f:150000],axis=(0,1,2))

print(mmax,mmin)

print("Getting dataset limits")
path = "D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20.hdf5"
with h5py.File(path,'r') as file:
    mmax,mmin = np.nanmax(file['pi-camera-1'][()],axis=(0,1,2)),np.nanmin(file['pi-camera-1'][()],axis=(0,1,2))

def norm2image(T,vmax=mmax,vmin=mmin):
    return int((T-vmin)*((2**16)/(vmax-vmin)))

def findBB(frame):
    # perform sobel edge detection
    sb = sobel(frame)
    # mask to a central area
    sb[:,:15] = 0
    sb[:,25:] = 0
    # threshold the sobel results using otsu
    thresh = threshold_otsu(sb)
    img = (sb > thresh).astype('uint8')*255
    # use morphology open operation to close holes in the thresholded shape
    # to get a better contour result
    img=cv2.morphologyEx(img,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    # find contours in image
    ct = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]
    # if a contour was found
    if len(ct)>0:
        # if more than one contour was found, sort it by size, descending order
        if len(ct)>1:
            ct.sort(key=lambda x : cv2.contourArea(x),reverse=True)
        # return bounding box and largest contour
        return cv2.boundingRect(ct[0]),ct[0]
    else:
        # if no contour was found then no bounding box can be found
        # return -1,-1 to indicate this
        return -1,-1

def getDataInCT(frame,ct):
    mask = np.zeros(frame.shape,np.uint8)
    mask = cv2.drawContours(mask,[ct],0,1,-1)
    return frame*mask

## setup renderer, window and interactor
ren = vtk.vtkRenderer()
# set background color
ren.SetBackground(1.0,1.0,1.0)
renwin = vtk.vtkRenderWindow()
renwin.AddRenderer(ren)
renwin.SetSize(500,500)
iren = vtk.vtkRenderWindowInteractor()
# define interactor
iren.SetRenderWindow(renwin)

# setup data connection
pointSource = vtk.vtkProgrammableSource()
def readPointsRotRbf():
    print("Updating data points")
    print(ff)
    # clear data and register blank point set
    output = pointSource.GetPolyDataOutput()
    points = vtk.vtkPoints()
    output.SetPoints(points)
    # collect frame
    with h5py.File(path,'r') as file:
        frame = file['pi-camera-1'][:,:,ff]
    np.nan_to_num(frame,copy=False)
    # find bounding box and contour used to generate it
    bb,ct = findBB(frame)
    if (bb == -1) or (bb is None):
        print("Failed to find contours or bounding box")
        # add in a single point so the next frame can be processed without interrupting the recording
        return points.InsertNextPoint(0.0,0.0,0.0)
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
        print("Rbf Failed!")
        return points.InsertNextPoint(0.0,0.0,0.0)
    
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

    lim = zz.min() + (zz.max()-zz.min())*filt
    for x,y,z in zip(xxr[dd>=lim],yyr[dd>=lim],zzr[dd>=lim]):
        points.InsertNextPoint(float(z),float(y),float(x))
    ## adding temperature values
    # create array from temperature values
    np.nan_to_num(dd,copy=False)
    # convert temperature to vtk array
    tt = numpy_support.numpy_to_vtk(dd,array_type=vtk.VTK_FLOAT)
    # name it
    tt.SetName("Temperature")
    # add temperature array to the position array and mark it as the active scalar set
    # the active set will be used to generate color maps
    pointSource.GetOutput().GetPointData().AddArray(tt)
    pointSource.GetOutput().GetPointData().SetActiveScalars("Temperature")
    print("Finished updating points!")

pointSource.SetExecuteMethod(readPointsRotRbf)
pointSource.Update()

# set up voxel modeller
print("Setting up voxel modeller")
voxel = vtk.vtkVoxelModeller()
# updating the sampling bounds with the bounds of the data
voxel.SetModelBounds(pointSource.GetOutput().GetBounds())
voxel.SetSampleDimensions(100,100,100)
voxel.SetScalarTypeToFloat()
voxel.SetInputConnection(pointSource.GetOutputPort())
voxel.Update()

# setup marching contour filter
print("Setting up marching cubes")
marching_cubes = vtk.vtkMarchingCubes()
marching_cubes.SetInputConnection(voxel.GetOutputPort())
marching_cubes.ComputeScalarsOn()
marching_cubes.ComputeGradientsOn()
marching_cubes.ComputeNormalsOn()
marching_cubes.SetValue(0,norm2image(18.0))
marching_cubes.Update()

# create mapper
print("Setting up mapper")
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(marching_cubes.GetOutputPort())
mapper.SetColorModeToMapScalars()
mapper.SetScalarRange(pointSource.GetOutput().GetScalarRange())
mapper.Update()

# create actor, an object in a rendered scene
print("Setting up actor")
actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetRepresentationToSurface()
actor.GetProperty().SetColor(0,1,0)
actor.GetProperty().SetSpecular(.3)
actor.GetProperty().SetSpecularPower(20)
actor.GetProperty().SetOpacity(.5)

# add actor to renderer
ren.AddActor(actor)
# initialize window interactor
iren.Initialize()

# generate window
renwin.Render()
# start interactor
iren.Start()
