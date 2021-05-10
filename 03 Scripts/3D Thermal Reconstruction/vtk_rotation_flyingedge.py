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
import math

set_bk_black = True

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

ren = vtk.vtkRenderer()
renwin = vtk.vtkRenderWindow()
renwin.AddRenderer(ren)
iren = vtk.vtkRenderWindowInteractor()
# define interactor
iren.SetRenderWindow(renwin)

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

pointSource = vtk.vtkProgrammableSource()
bounds = []
def readPointsRotRbf():
    global ff
    print(ff)
    # clear data and register blank point set
    imgput = pointSource.GetPolyDataOutput()
    points = vtk.vtkPoints()
    imgput.SetPoints(points)
    # collect frame
    with h5py.File(path,'r') as file:
        frame = file['pi-camera-1'][:,:,ff]
    np.nan_to_num(frame,copy=False)
    # find bounding box and contour used to generate it
    bb,ct = findBB(frame)
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

    lim = zz.min() + (zz.max()-zz.min())*filt
    for x,y,z in zip(xxr[dd>=lim],yyr[dd>=lim],zzr[dd>=lim]):
        points.InsertNextPoint(float(z),float(y),float(x))

    bounds = points.GetBounds()
    ## adding temperature values
    # create array from temperature values
    np.nan_to_num(dd,copy=False)
    tt = numpy_support.numpy_to_vtk(dd[dd>=lim],array_type=vtk.VTK_FLOAT)
    tt.SetName("Temperature")
    pointSource.GetOutput().GetPointData().AddArray(tt)
    pointSource.GetOutput().GetPointData().SetActiveScalars("Temperature")
    print("Finished updating points!")

pointSource.SetExecuteMethod(readPointsRotRbf)


## convert data to image data
imageSource = vtk.vtkProgrammableFilter()
imageSource.SetInputConnection(pointSource.GetOutputPort())
procImg = vtk.vtkImageData()
# adapted from
# https://vtk.org/Wiki/VTK/Examples/Cxx/PolyData/PolyDataToImageData
def convertPolyDataToImageData():
    print("Converting polydata to imagedata")
    # poly data
    inp = imageSource.GetInput()

    img = vtk.vtkImageData()
    # get bounds of the input
    bounds = inp.GetBounds()
    # setup spacing
    sp = 0.01
    img.SetSpacing(sp,sp,sp)
    dims = [0,0,0]
    # calculate dimensions given spacing
    for i in range(3):
        dims[i] = int(math.ceil((bounds[i*2+1]-bounds[i*2])/sp))
    img.SetDimensions(dims)
    img.SetExtent(0,dims[0]-1,0,dims[1]-1,0,dims[2]-1)

    # calculate centre of the image
    origin = [0]*3
    origin[0] = bounds[0]+sp/2
    origin[1] = bounds[2]+sp/2
    origin[2] = bounds[4]+sp/2
    img.SetOrigin(origin)

    # set the values of the image
    img.AllocateScalars(vtk.VTK_UNSIGNED_CHAR,1)
    for i in range(img.GetNumberOfPoints()):
        # fill image with foreground voxels
        img.GetPointData().GetScalars().SetTuple1(i,255)
        
    poly2stencil = vtk.vtkPolyDataToImageStencil()
    poly2stencil.SetInputData(inp)
    poly2stencil.SetOutputSpacing(sp,sp,sp)
    poly2stencil.SetOutputWholeExtent(img.GetExtent())
    poly2stencil.Update()

    # cut the white image and set the background using stencil
    stencil = vtk.vtkImageStencil()
    stencil.SetInputData(img)
    stencil.SetStencilConnection(poly2stencil.GetOutputPort())
    stencil.ReverseStencilOff()
    stencil.SetBackgroundValue(0.0)
    stencil.Update()
    # update output
    global procImg
    procImg = stencil.GetOutput()

imageSource.SetExecuteMethod(convertPolyDataToImageData)
imageSource.Update()

fedge = vtk.vtkFlyingEdges3D()
fedge.SetInputData(procImg)
fedge.ComputeNormalsOff()
fedge.ComputeGradientsOff()
# generate one contour across the 
fedge.GenerateValues(1,[0,255])
fedge.Update()

geo = vtk.vtkGeometryFilter()
geo.SetInputConnection(fedge.GetOutputPort())
geo.Update()

# set mapper to handle the shrinking
mm = vtk.vtkDataSetMapper()
mm.SetInputConnection(geo.GetOutputPort())
triangulation = vtk.vtkActor()
triangulation.SetMapper(mm)
triangulation.GetProperty().SetColor(1, 0, 0) # set drawn surface to red

# Add the actors to the renderer, set the background and size
ren.AddActor(triangulation)
ren.SetBackground(1, 1, 1) # white
renwin.SetSize(500, 500)
renwin.Render()

cam1 = ren.GetActiveCamera()
cam1.Zoom(1.0)
# rotate piece by 180 degrees so the laser part is at the top and the
# heating area is at the bottom
cam1.Roll(180.0)

iren.Initialize()
renwin.Render()
iren.Start()
