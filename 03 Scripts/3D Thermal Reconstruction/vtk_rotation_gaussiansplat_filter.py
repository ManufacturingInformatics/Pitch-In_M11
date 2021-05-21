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
def readPointsRotRbf():
    global ff
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
    ## adding temperature values
    # create array from temperature values
    np.nan_to_num(dd,copy=False)
    tt = numpy_support.numpy_to_vtk(dd[dd>=lim],array_type=vtk.VTK_FLOAT)
    tt.SetName("Temperature")
    pointSource.GetOutput().GetPointData().AddArray(tt)
    pointSource.GetOutput().GetPointData().SetActiveScalars("Temperature")
    print("Finished updating points!")

##
def readPointsRotRbfFiltMin():
    global ff
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

    lim = zz.min()+1.0
    for x,y,z in zip(xxr[dd>=lim],yyr[dd>=lim],zzr[dd>=lim]):
        points.InsertNextPoint(float(z),float(y),float(x))
    ## adding temperature values
    # create array from temperature values
    np.nan_to_num(dd,copy=False)
    tt = numpy_support.numpy_to_vtk(dd[dd>=lim],array_type=vtk.VTK_FLOAT)
    tt.SetName("Temperature")
    pointSource.GetOutput().GetPointData().AddArray(tt)
    pointSource.GetOutput().GetPointData().SetActiveScalars("Temperature")
    print("Finished updating points!")

def readPointsRotRbfBk():
    print("Updating data")
    output = pointSource.GetPolyDataOutput()
    points = vtk.vtkPoints()
    output.SetPoints(points)
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
    for r in np.arange(0.0,360.0,1.0):
        # generate rotation object
        rot = R.from_euler('xyz',[0.0,r,0.0],degrees=True)
        # apply to matrix
        vv = rot.apply(data)
        # udpate position matrix
        xxr = np.concatenate((xxr,vv[:,0]))
        yyr = np.concatenate((yyr,vv[:,1]))
        zzr = np.concatenate((zzr,vv[:,2]))
        dd = np.concatenate((dd,zz))

    print("Filtering data from background")
    dd -= dd.min()
    lim = np.diff(dd).max()*filt
    #lim = 1.0
    print(lim)
    print(xxr[dd>lim].shape,yyr[dd>lim].shape,zzr[dd>lim].shape)
    print(dd.min(),dd.max())
    # background values are a base value plus some noise
    # to ignore tha background, we ignore values less than 1
    # this should reveal the values that stand out from the bk
    for x,y,z in zip(xxr[dd>lim],yyr[dd>lim],zzr[dd>lim]):
        points.InsertNextPoint(float(z),float(y),float(x))

    np.nan_to_num(dd,copy=False)
    tt = numpy_support.numpy_to_vtk(dd[dd>lim],array_type=vtk.VTK_FLOAT)
    tt.SetName("Temperature")
    pointSource.GetOutput().GetPointData().AddArray(tt)
    pointSource.GetOutput().GetPointData().SetActiveScalars("Temperature")
    print("Finished updating points!")

pointSource.SetExecuteMethod(readPointsRotRbfBk)
# apply gaussian splatter to interpolate between values, inc temperature
# applies gaussians of a certain size to interpolate
print("Setting up Gaussian splatter")
gsplat = vtk.vtkGaussianSplatter()
gsplat.SetInputConnection(pointSource.GetOutputPort())
# set ratio between minor and major axis of elliptical base
gsplat.SetEccentricity(0.02)
# number of data points to generate in interpolation
gsplat.SetSampleDimensions(50,50,50)
# exponent factor of gaussians
gsplat.SetExponentFactor(-5)
gsplat.SetRadius(0.5)
gsplat.Update()

# volume renderer
print("Setting up smart mapper")
volmap = vtk.vtkSmartVolumeMapper()
volmap.SetBlendModeToComposite()
volmap.SetInputConnection(gsplat.GetOutputPort())
volmap.SetRequestedRenderModeToGPU()
volprop = vtk.vtkVolumeProperty()
volprop.ShadeOff()
volprop.SetInterpolationType(vtk.VTK_LINEAR_INTERPOLATION)
volmap.Update()

# clipping to display interior
print("Setting up cropping planes")
volmap.CroppingOn()
bds = volmap.GetBounds()
volmap.SetCroppingRegionPlanes(bds[0],bds[1],bds[2],bds[3],bds[4]+0.5*(bds[5]-bds[4]),bds[5])
volmap.SetCroppingRegionFlagsToSubVolume()

## create filter slider to adjust cropping
print("Setting up sliders")
slcliprep = vtk.vtkSliderRepresentation2D()
slcliprep.GetPoint1Coordinate().SetCoordinateSystemToDisplay()
slcliprep.GetPoint1Coordinate().SetValue(300,40)
slcliprep.GetPoint2Coordinate().SetCoordinateSystemToDisplay()
slcliprep.GetPoint2Coordinate().SetValue(470,40)
slcliprep.SetMinimumValue(0.0) # limits of slider
slcliprep.SetMaximumValue(1.0)
slcliprep.SetValue(0.5)
slcliprep.SetLabelFormat("%0.1f") # format the index is displayed in
slcliprep.SetTitleText("Cropping") # slider title
slcliprep.GetSliderProperty().SetColor(0,1,0) # Set nob to red
slcliprep.GetSelectedProperty().SetColor(0,0,1) # set color when selected
# if the background is set to white
if not set_bk_black:
    slcliprep.GetTitleProperty().SetColor(0,0,0) # set text to black
    slcliprep.GetLabelProperty().SetColor(0,0,0) # set color of value label text
    slcliprep.GetTubeProperty().SetColor(0,0,0) # set the bit that moves to black
    slcliprep.GetCapProperty().SetColor(0,0,0) # set the end points to black
else:
    slcliprep.GetTitleProperty().SetColor(1,1,1) # set text to black
    slcliprep.GetLabelProperty().SetColor(1,1,1) # set color of value label text
    slcliprep.GetTubeProperty().SetColor(1,1,1) # set the bit that moves to black
    slcliprep.GetCapProperty().SetColor(1,1,1) # set the end points to black


slclip = vtk.vtkSliderWidget()
slclip.SetRepresentation(slcliprep)
slclip.SetInteractor(iren)
slclip.SetAnimationModeToJump()
slclip.EnabledOn()

def set_clip(obj,event):
    print("Clip changed")
    bds = volmap.GetBounds()
    clip = obj.GetRepresentation().GetValue()
    # set plane to clip 3d shape
    # Set*Planes(xmin,xmax,ymin,ymax,zmin,zmax)
    volmap.SetCroppingRegionPlanes(bds[0],bds[1],bds[2],bds[3],bds[4]+clip*(bds[5]-bds[4]),bds[5])
    volmap.SetCroppingRegionFlagsToSubVolume()
    volmap.Modified()
    
slclip.AddObserver("InteractionEvent",set_clip)

## create filter slider
slfiltrep = vtk.vtkSliderRepresentation2D()
slfiltrep.GetPoint1Coordinate().SetCoordinateSystemToDisplay()
slfiltrep.GetPoint1Coordinate().SetValue(30,470)
slfiltrep.GetPoint2Coordinate().SetCoordinateSystemToDisplay()
slfiltrep.GetPoint2Coordinate().SetValue(200,470)
slfiltrep.SetMinimumValue(0.0) # limits of slider
slfiltrep.SetMaximumValue(1.0)
slfiltrep.SetValue(filt)
slfiltrep.SetLabelFormat("%0.1f") # format the index is displayed in
slfiltrep.SetTitleText("Filter") # slider title
slfiltrep.GetSliderProperty().SetColor(0,1,0) # Set nob to red
slfiltrep.GetSelectedProperty().SetColor(0,0,1) # set color when selected
if not set_bk_black:
    slfiltrep.GetTitleProperty().SetColor(0,0,0) # set text to black
    slfiltrep.GetLabelProperty().SetColor(0,0,0) # set color of value label text
    slfiltrep.GetTubeProperty().SetColor(0,0,0) # set the bit that moves to black
    slfiltrep.GetCapProperty().SetColor(0,0,0) # set the end points to black
else:
    slfiltrep.GetTitleProperty().SetColor(1,1,1) # set text to black
    slfiltrep.GetLabelProperty().SetColor(1,1,1) # set color of value label text
    slfiltrep.GetTubeProperty().SetColor(1,1,1) # set the bit that moves to black
    slfiltrep.GetCapProperty().SetColor(1,1,1) # set the end points to black

slfilt = vtk.vtkSliderWidget()
slfilt.SetRepresentation(slfiltrep)
slfilt.SetInteractor(iren)
slfilt.SetAnimationModeToJump()
slfilt.EnabledOn()

def set_filt(obj,event):
    global filt
    print("Filter changed")
    filt = obj.GetRepresentation().GetValue()
    pointSource.Modified()
    vol.Modified()
    
slfilt.AddObserver("EndInteractionEvent",set_filt)

## create slider for changing index
slrep = vtk.vtkSliderRepresentation2D()
slrep.SetMinimumValue(float(starter_f)) # limits of slider
slrep.SetMaximumValue(150000.0)
slrep.SetValue(float(starter_f))
slrep.SetLabelFormat("%0.f") # format the index is displayed in
slrep.SetTitleText("Index") # slider title
slrep.GetSliderProperty().SetColor(1,0,0) # Set nob to red
slrep.GetSelectedProperty().SetColor(0,1,0) # set color when selected
if not set_bk_black:
    slrep.GetTitleProperty().SetColor(0,0,0) # set text to black
    slrep.GetLabelProperty().SetColor(0,0,0) # set color of value label text
    slrep.GetTubeProperty().SetColor(0,0,0) # set the bit that moves to black
    slrep.GetCapProperty().SetColor(0,0,0) # set the end points to black
else:
    slrep.GetTitleProperty().SetColor(1,1,1) # set text to black
    slrep.GetLabelProperty().SetColor(1,1,1) # set color of value label text
    slrep.GetTubeProperty().SetColor(1,1,1) # set the bit that moves to black
    slrep.GetCapProperty().SetColor(1,1,1) # set the end points to black
    
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

print("Setting up color and opacity")
# opacity
opacity = vtk.vtkPiecewiseFunction()
opacity.AddPoint(0.0,0.0)
opacity.AddPoint(0.2,1.0)
opacity.AddPoint(0.4,1.0)
opacity.AddPoint(1.0,1.0)
volprop.SetScalarOpacity(opacity)

# color
color = vtk.vtkColorTransferFunction()
#color.AddRGBPoint(0.0,0.0,0.0,1.0)
#color.Build()
def buildFunctionFromColormap(ctf,name='hot',numpoints=1000):
    import matplotlib.cm as cmap
    cm = cmap.get_cmap(name)
    lims = pointSource.GetOutput().GetScalarRange()
    for cc in np.linspace(mmin,mmax,num=numpoints):
        ctf.AddRGBPoint(cc,*cm((cc-lims[0])*(lims[1]-lims[0])**-1)[:-1])
    ctf.Build()
    
buildFunctionFromColormap(color)
volprop.SetColor(color)

# thing that actually generates volume
vol = vtk.vtkVolume()
vol.SetMapper(volmap)
vol.SetProperty(volprop)

# set up renderer
ren.AddViewProp(vol)
ren.ResetCamera()

# rotate camera
cam1 = ren.GetActiveCamera()
# rotate image 180.0 to have laser portion at the top
cam1.Roll(180.0)
#cam1.Yaw(-180.0)

# setting for depth peeling
renwin.SetAlphaBitPlanes(1)
renwin.SetMultiSamples(0)
ren.SetUseDepthPeeling(1)
ren.SetMaximumNumberOfPeels(100)
ren.SetOcclusionRatio(0.1)

print("Starting")
# set bk color
if set_bk_black:
    ren.SetBackground(0.0,0.0,0.0)
else:
    ren.SetBackground(1.0,1.0,1.0)
renwin.SetSize(500, 500)
renwin.Render()

# start process
iren.Initialize()
iren.Start()
