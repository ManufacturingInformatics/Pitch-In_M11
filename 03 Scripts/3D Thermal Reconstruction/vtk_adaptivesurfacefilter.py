import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.interpolate
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from skimage.filters import sobel,threshold_otsu
import cv2
from scipy.spatial.transform import Rotation as R
from matplotlib.tri.triangulation import Triangulation as mtri
import vtk
from vtk.util import numpy_support
from vtk.numpy_interface import dataset_adapter as dsa
from vtk.numpy_interface import algorithms as alg
  
from scipy.interpolate import Rbf

# filtering limit
filt = 0.6
# rotation resolution
rot_res = 10.0
# path to hdf5 file
path = "D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20.hdf5"
#path = "D:\BEAM\RPiWaifu\pi-camera-data-127001-2019-10-14T12-41-20-waifu-noise-scale-91398-150000.hdf5"

if "waifu" not in path:
    # index to start on
    starter_f = 93199
    # initial ff
    ff = starter_f
    end_f = 150000
else:
    starter_f = 0
    ff = starter_f
    end_f = 58601

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

pointSource = vtk.vtkProgrammableSource()
def readPointsRotRbf():
    global ff
    # clear data and register blank point set
    output = pointSource.GetPolyDataOutput()
    points = vtk.vtkPoints()
    output.SetPoints(points)
    # collect frame
    if "waifu" not in path:
        with h5py.File(path,'r') as file:
            frame = file['pi-camera-1'][:,:,ff]
    else:
        with h5py.File(path,'r') as file:
            frame = file['pi-camera-1-scale'][:,:,ff]
            
    np.nan_to_num(frame,copy=False)
    # find bounding box and contour used to generate it
    bb,ct = findBB(frame)
    if (bb == -1) or (bb is None):
        print("Failed to find contours or bounding box")
        return np.empty(0),np.empty(0),np.empty(0)
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
    np.nan_to_num(dd,copy=False)
    tt = numpy_support.numpy_to_vtk(dd,array_type=vtk.VTK_FLOAT)
    tt.SetName("Temperature")
    pointSource.GetOutput().GetPointData().AddArray(tt)
    pointSource.GetOutput().GetPointData().SetActiveScalars("Temperature")
    print("Finished updating points")

pointSource.SetExecuteMethod(readPointsRotRbf)

delny = vtk.vtkDelaunay3D()
delny.SetInputConnection(pointSource.GetOutputPort())
delny.SetTolerance(10.0)
delny.SetAlpha(0.2)
delny.BoundingTriangulationOff()

adapt = vtk.vtkAdaptiveDataSetSurfaceFilter()
adapt.SetInputConnection(delny.GetOutputPort())
adapt.Update()

# Shrink the result to help see it better.
shrink = vtk.vtkShrinkFilter()
shrink.SetInputConnection(adapt.GetOutputPort())
shrink.SetShrinkFactor(0.9)
# set mapper to handle the shrinking
mm = vtk.vtkDataSetMapper()

## estimate volume
geo = vtk.vtkGeometryFilter()
geo.SetInputConnection(shrink.GetOutputPort())
geo.Update()
tri = vtk.vtkTriangleFilter()
tri.SetInputConnection(geo.GetOutputPort())
tri.Update()
vol = vtk.vtkMassProperties()
vol.SetInputConnection(tri.GetOutputPort())
#print("Vol: {} p^3, Proj Vol: {} p^3".format(vol.GetVolume(),vol.GetVolumeProjected()))
# add observer to print updated values
def print_MassProperties(obj,event):
    print("Object Mass Properties")
    print("=======================")
    print("Est Volume: \t{} pixels^3".format(obj.GetVolume()))
    print("Projected Volume: \t{} pixels^3".format(obj.GetVolumeProjected()))
    print("Surface Area: \t{} pixels^3".format(obj.GetSurfaceArea()))
# setup observer so that whenever mass properties is modified, some of the
# properties are printed
vol.AddObserver("ModifiedEvent",print_MassProperties)
vol.Update()
vol.Modified()

# after passing the delauny output through the geometry filter
# we get a smooth surface rather than the delauny triangular surface
mm.SetInputConnection(geo.GetOutputPort())
triangulation = vtk.vtkActor()
triangulation.SetMapper(mm)
#triangulation.GetProperty().SetColor(1, 0, 0) # set drawn surface to red
triangulation.GetProperty().SetInterpolationToGouraud()

# Create graphics stuff
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.SetWindowName("Rotated Raspberry Pi Dataset")
renWin.AddRenderer(ren)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

## create slider to frame index
# set aesthetics
slrep = vtk.vtkSliderRepresentation2D()
slrep.SetMinimumValue(float(starter_f)) # limits of slider
slrep.SetMaximumValue(float(end_f))
slrep.SetValue(float(starter_f))
slrep.SetLabelFormat("%0.f") # format the index is displayed in
slrep.SetTitleText("Index") # slider title
slrep.GetTitleProperty().SetColor(0,0,0) # set text to black
slrep.GetSliderProperty().SetColor(1,0,0) # Set nob to red
slrep.GetLabelProperty().SetColor(0,0,0) # set color of value label text
slrep.GetSelectedProperty().SetColor(0,1,0) # set color when selected
slrep.GetTubeProperty().SetColor(0,0,0) # set the bit that moves to black
slrep.GetCapProperty().SetColor(0,0,0) # set the end points to black
# set position
slrep.GetPoint1Coordinate().SetCoordinateSystemToDisplay()
slrep.GetPoint1Coordinate().SetValue(30,40)
slrep.GetPoint2Coordinate().SetCoordinateSystemToDisplay()
slrep.GetPoint2Coordinate().SetValue(200,40)
# create slider object
sl = vtk.vtkSliderWidget()
# set interaction with window
sl.SetInteractor(iren)
sl.SetRepresentation(slrep)
# remove animation iterating over process
sl.SetAnimationModeToJump()
sl.EnabledOn()
# create callback
def index_callback(obj,event):
    # update global index counter so when the point source is updated
    # it accesses the desired frame index
    global ff
    ff = int(obj.GetRepresentation().GetValue())
    # call modified method to update point data which will in recall
    # delauny to show new set
    pointSource.Modified()
    vol.Modified()

sl.AddObserver("EndInteractionEvent",index_callback)

## create filter slider
slfiltrep = vtk.vtkSliderRepresentation2D()
slfiltrep.GetPoint1Coordinate().SetCoordinateSystemToDisplay()
slfiltrep.GetPoint1Coordinate().SetValue(300,40)
slfiltrep.GetPoint2Coordinate().SetCoordinateSystemToDisplay()
slfiltrep.GetPoint2Coordinate().SetValue(470,40)
slfiltrep.SetMinimumValue(0.0) # limits of slider
slfiltrep.SetMaximumValue(1.0)
slfiltrep.SetValue(filt)
slfiltrep.SetLabelFormat("%0.1f") # format the index is displayed in
slfiltrep.SetTitleText("Filter") # slider title
slfiltrep.GetTitleProperty().SetColor(0,0,0) # set text to black
slfiltrep.GetSliderProperty().SetColor(0,1,0) # Set nob to red
slfiltrep.GetLabelProperty().SetColor(0,0,0) # set color of value label text
slfiltrep.GetSelectedProperty().SetColor(0,0,1) # set color when selected
slfiltrep.GetTubeProperty().SetColor(0,0,0) # set the bit that moves to black
slfiltrep.GetCapProperty().SetColor(0,0,0) # set the end points to black

slfilt = vtk.vtkSliderWidget()
slfilt.SetRepresentation(slfiltrep)
slfilt.SetInteractor(iren)
slfilt.SetAnimationModeToJump()
slfilt.EnabledOn()

def set_filt(obj,event):
    global filt
    filt = obj.GetRepresentation().GetValue()
    pointSource.Modified()
    vol.Modified()
    
slfilt.AddObserver("EndInteractionEvent",set_filt)

## create slider to update rotation resolution
slrotrep = vtk.vtkSliderRepresentation2D()
slrotrep.GetPoint1Coordinate().SetCoordinateSystemToDisplay()
slrotrep.GetPoint1Coordinate().SetValue(30,470)
slrotrep.GetPoint2Coordinate().SetCoordinateSystemToDisplay()
slrotrep.GetPoint2Coordinate().SetValue(200,470)
slrotrep.SetMinimumValue(1.0) # limits of slider
slrotrep.SetMaximumValue(50.0)
slrotrep.SetValue(rot_res)
slrotrep.SetLabelFormat("%0.f") # format the index is displayed in
slrotrep.SetTitleText("Rotation Res") # slider title
slrotrep.GetTitleProperty().SetColor(0,0,0) # set text to black
slrotrep.GetSliderProperty().SetColor(0,0,1) # Set nob to 
slrotrep.GetLabelProperty().SetColor(0,0,0) # set color of value label text
slrotrep.GetSelectedProperty().SetColor(1,0,0) # set color when selected
slrotrep.GetTubeProperty().SetColor(0,0,0) # set the bit that moves to black
slrotrep.GetCapProperty().SetColor(0,0,0) # set the end points to black

slrot = vtk.vtkSliderWidget()
slrot.SetRepresentation(slrotrep)
slrot.SetInteractor(iren)
slrot.SetAnimationModeToJump()
slrot.EnabledOn()

def set_rot_res(obj,event):
    global rot_res
    rot_res = obj.GetRepresentation().GetValue()
    pointSource.Modified()

slrot.AddObserver("EndInteractionEvent",set_rot_res)

# Add the actors to the renderer, set the background and size
ren.AddActor(triangulation)
ren.SetBackground(1, 1, 1) # white
renWin.SetSize(500, 500)
renWin.Render()

cam1 = ren.GetActiveCamera()
cam1.Zoom(1.0)
# rotate piece by 180 degrees so the laser part is at the top and the
# heating area is at the bottom
cam1.Roll(180.0)

print("Stopping messages")
vtk.vtkOutputWindow.GetInstance().SetGlobalWarningDisplay(0)

iren.Initialize()
renWin.Render()
iren.Start()
