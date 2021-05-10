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

setblackbk = True

# index to start on
starter_f = 93199
# initial ff
ff = starter_f
# filtering limit
filt = 0.0
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
    else:
        return -1,-1

def getDataInCT(frame,ct):
    mask = np.zeros(frame.shape,np.uint8)
    mask = cv2.drawContours(mask,[ct],0,1,-1)
    return frame*mask

pointSource = vtk.vtkProgrammableSource()
def readPointsRotRbf():
    print("Updating data points")
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
    tt = numpy_support.numpy_to_vtk(dd,array_type=vtk.VTK_FLOAT)
    tt.SetName("Temperature")
    pointSource.GetOutput().GetPointData().AddArray(tt)
    pointSource.GetOutput().GetPointData().SetActiveScalars("Temperature")
    print("Finished updating points!")

pointSource.SetExecuteMethod(readPointsRotRbf)
# apply gaussian splatter to interpolate between values, inc temperature
# applies gaussians of a certain size to interpolate
gsplat = vtk.vtkGaussianSplatter()
gsplat.SetInputConnection(pointSource.GetOutputPort())
# set ratio between minor and major axis of elliptical base
gsplat.SetEccentricity(0.02)
# number of data points to generate in interpolation
gsplat.SetSampleDimensions(20,20,20)
# exponent factor of gaussians
gsplat.SetExponentFactor(-5)
gsplat.SetRadius(0.5)
gsplat.Update()

# volume renderer
volmap = vtk.vtkSmartVolumeMapper()
volmap.SetBlendModeToComposite()
volmap.SetInputConnection(gsplat.GetOutputPort())
volmap.SetRequestedRenderModeToGPU()
volprop = vtk.vtkVolumeProperty()
volprop.ShadeOff()
volprop.SetInterpolationType(vtk.VTK_LINEAR_INTERPOLATION)
volmap.Update()

# clipping to display interior
volmap.CroppingOn()
bds = volmap.GetBounds()
volmap.SetCroppingRegionPlanes(bds[0],bds[1],bds[2],bds[3],bds[4],bds[5]-0.5*(bds[5]-bds[4]))
volmap.SetCroppingRegionFlagsToSubVolume()

## create filter slider to adjust cropping
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
slcliprep.GetSelectedProperty().SetColor(0,0,1) # set color when selected
slcliprep.GetSliderProperty().SetColor(0,1,0) # Set nob to red
if setblackbk:
    slcliprep.GetTitleProperty().SetColor(1,1,1) # set text to black
    slcliprep.GetLabelProperty().SetColor(1,1,1) # set color of value label text
    slcliprep.GetTubeProperty().SetColor(1,1,1) # set the bit that moves to black
    slcliprep.GetCapProperty().SetColor(1,1,1) # set the end points to black
else:
    slcliprep.GetTitleProperty().SetColor(0,0,0) # set text to black
    slcliprep.GetLabelProperty().SetColor(0,0,0) # set color of value label text
    slcliprep.GetTubeProperty().SetColor(0,0,0) # set the bit that moves to black
    slcliprep.GetCapProperty().SetColor(0,0,0) # set the end points to black

slclip = vtk.vtkSliderWidget()
slclip.SetRepresentation(slcliprep)
slclip.SetInteractor(iren)
slclip.SetAnimationModeToJump()
slclip.EnabledOn()

def set_clip(obj,event):
    print("Clip changed")
    # get current limits of the plot
    bds = volmap.GetBounds()
    # get current clip value
    clip = obj.GetRepresentation().GetValue()
    # update cropping plane along z axis
    volmap.SetCroppingRegionPlanes(bds[0],bds[1],bds[2],bds[3],bds[4],bds[5]-clip*(bds[5]-bds[4]))
    volmap.SetCroppingRegionFlagsToSubVolume()
    #pointSource.Modified()
    volprop.Modified()
    # indicate that the volume mapping has been modified
    volmap.Modified()
# assign set_clip function as callback to whenever the slider is used
# interaction rather than end interaction as the update is very fast
slclip.AddObserver("EndInteractionEvent",set_clip)

## create filter slider
slfiltrep = vtk.vtkSliderRepresentation2D()
slfiltrep.GetPoint1Coordinate().SetCoordinateSystemToDisplay()
slfiltrep.GetPoint1Coordinate().SetValue(30,470)
slfiltrep.GetPoint2Coordinate().SetCoordinateSystemToDisplay()
slfiltrep.GetPoint2Coordinate().SetValue(200,470)
slfiltrep.SetMinimumValue(0.0) # limits of slider
slfiltrep.SetMaximumValue(1.0)
slfiltrep.SetValue(filt)
slfiltrep.SetLabelFormat("%0.2f") # format the index is displayed in
slfiltrep.SetTitleText("Filter") # slider title
slfiltrep.GetSelectedProperty().SetColor(0,0,1) # set color when selected
slfiltrep.GetSliderProperty().SetColor(0,1,0) # Set nob to red
if setblackbk:
    slfiltrep.GetTitleProperty().SetColor(1,1,1) # set text to black
    slfiltrep.GetLabelProperty().SetColor(1,1,1) # set color of value label text
    slfiltrep.GetTubeProperty().SetColor(1,1,1) # set the bit that moves to black
    slfiltrep.GetCapProperty().SetColor(1,1,1) # set the end points to black
else:
    slfiltrep.GetTitleProperty().SetColor(0,0,0) # set text to black
    slfiltrep.GetLabelProperty().SetColor(0,0,0) # set color of value label text
    slfiltrep.GetTubeProperty().SetColor(0,0,0) # set the bit that moves to black
    slfiltrep.GetCapProperty().SetColor(0,0,0) # set the end points to black

slfilt = vtk.vtkSliderWidget()
slfilt.SetRepresentation(slfiltrep)
slfilt.SetInteractor(iren)
slfilt.SetAnimationModeToJump()
slfilt.EnabledOn()

def set_filt(obj,event):
    global filt
    #print("Filter changed")
    # update filtering parameter to the value of the slider
    filt = obj.GetRepresentation().GetValue()
    # indicate that the point source has been modified and needs updating
    pointSource.Modified()
    # indicate that the volume has been
    volmap.Modified()
    
slfilt.AddObserver("EndInteractionEvent",set_filt)

## create slider for changing index
slrep = vtk.vtkSliderRepresentation2D()
slrep.SetMinimumValue(float(starter_f)) # limits of slider
slrep.SetMaximumValue(150000.0)
slrep.SetValue(float(starter_f))
slrep.SetLabelFormat("%0.f") # format the index is displayed in
slrep.SetTitleText("Index") # slider title
slrep.GetSelectedProperty().SetColor(0,0,1) # set color when selected
slrep.GetSliderProperty().SetColor(0,1,0) # Set nob to red
if setblackbk:
    slrep.GetTitleProperty().SetColor(1,1,1) # set text to black
    slrep.GetLabelProperty().SetColor(1,1,1) # set color of value label text
    slrep.GetTubeProperty().SetColor(1,1,1) # set the bit that moves to black
    slrep.GetCapProperty().SetColor(1,1,1) # set the end points to black
else:
    slrep.GetTitleProperty().SetColor(0,0,0) # set text to black
    slrep.GetLabelProperty().SetColor(0,0,0) # set color of value label text
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
def set_index(obj,event):
    print("Slider index callback")
    # update global index counter so when the point source is updated
    # it accesses the desired frame index
    global ff
    ff = int(obj.GetRepresentation().GetValue())
    # call modified method to update point data which will in recall
    # delauny to show new set
    pointSource.Modified()
    volmap.Modified()

sl.AddObserver("EndInteractionEvent",set_index)

# opacity
opacity = vtk.vtkPiecewiseFunction()
# set fuunction to be transparent at 0 scalar
# and opaque at pretty much every other value
opacity.AddPoint(0.0,0.0)
opacity.AddPoint(0.2,1.0)
opacity.AddPoint(0.4,1.0)
opacity.AddPoint(1.0,1.0)
volprop.SetScalarOpacity(opacity)

# color
color = vtk.vtkColorTransferFunction()
# build the color transfer function using values from matplotlib colormap
def buildFunctionFromColormap(ctf,name='hot',numpoints=1000):
    import matplotlib.cm as cmap
    # get the desired color map
    cm = cmap.get_cmap(name)
    # get the range of temperature values assigned to the Scalar array in
    # pointSource
    lims = pointSource.GetOutput().GetScalarRange()
    # iterate over a simulated range of scalar (temperature) values
    # normalize each value and extract the color from the colormap
    # the colormap returns BGRA
    for cc in np.linspace(mmin,mmax,num=numpoints):
        ctf.AddRGBPoint(cc,*cm((cc-lims[0])*(lims[1]-lims[0])**-1)[:-1])
    # build functions using the newly added values
    ctf.Build()
    return
    
buildFunctionFromColormap(color)
volprop.SetColor(color)

# thing that actually generates volume
vol = vtk.vtkVolume()
vol.SetMapper(volmap)
vol.SetProperty(volprop)

# set up renderer
ren.AddViewProp(vol)
ren.ResetCamera()

# setting for depth peeling
renwin.SetAlphaBitPlanes(1)
renwin.SetMultiSamples(0)
ren.SetUseDepthPeeling(1)
ren.SetMaximumNumberOfPeels(100)
ren.SetOcclusionRatio(0.1)

## create and set animation
class animation_callback():
    def __init__(self,*args):
        # get references to image and movie
        global imgFilter
        global movieWriter
        self.__imgFilter = imageFilter
        self.__mWriter = movieWriter
        # internal counter of number of times the timer has run
        self.__timerCount = 0
        self.__timerCountLim = 0
        # timer id
        self.__timerID = 0
        # resolutions for iteration
        self.clip_res = 0.1
        self.filt_res = 0.1
        self.rad_res = 0.1
        self.rot_res = 0.1
        self.ecc_res = 0.1
    def setTimerID(self,tid):
        self.__timerID = tid
        print("Timer ID Set! {}".format(self.__timerID))
    def setTimerCountLimit(self,lim):
        self.__timerCountLim = lim
    def writeToFile(self):
        print("Writing to file")
        # get the render window assigned to image converter
        ww = self.__imgFilter.GetInput()
        # render new data
        ww.Render()
        # get rendered window as image
        self.__imgFilter.Modified()
        self.__imgFilter.Update()
        # write image to file
        self.__mWriter.Write()
        self.__timerCount+=1;
        # check timer count
        # if we've run it the required number of times, destroy the timer
        # which will end the recording
        if self.__timerCount>=self.__timerCountLim:
            print("Reached count limit, destroying timer")
            ww.GetInteractor().DestroyTimer(self.__timerID)
            # close movie writer
            self.__mWriter.End()
            global movieWriter
            movieWriter.End()
    def index_callback(self,obj,event):
        print("Index callback!")
        global sl
        global ff
        ff += 1
        # move slider so it is part of the animation
        sl.GetRepresentation().SetValue(ff)
        sl.Modified()
        # increment global frame counter
        sl.InvokeEvent("EndInteractionEvent")
        # write vol data to file
        # render results
        self.writeToFile()
           
    def rotation_callback(self,obj,event):
        print("Rotation callback!")
        # get the renderers in the window
        cc = obj.GetRenderWindow().GetRenderers()
        # get the first one
        rr = cc.GetFirstRenderer()
        # get camera of renderer
        cm = rr.GetActiveCamera()
        # rotate camera
        cm.Azimuth(self.rot_res)
        # write results to file
        self.writeToFile()

    def filter_callback(self,obj,event):
        print("Filter callback!")
        global filt
        global slfilt
        filt+=self.filt_res
        # update slider to represent value
        slfilt.GetRepresentation().SetValue(filt)
        # update filter slider to match what's being displayed
        # indicate that the point source has been modified causing it to be updated
        slfilt.Modified()
        slfilt.InvokeEvent("EndInteractionEvent")
        self.writeToFile()

    def clip_callback(self,obj,event):
        print("Clip callback!")
        global clip
        global slclip
        clip+=self.clip_res
        slclip.GetRepresentation().SetValue(clip)
        slclip.Modified()
        slclip.InvokeEvent("EndInteractionEvent")
        self.writeToFile()

    def radius_callback(self,obj,event):
        print("Radius callback!")
        global gsplat
        # increment the gaussian radius value by the currently set resolution
        gsplat.SetRadius(gsplat.GetRadius()+self.rad_res)
        # force gaussian splat object to update
        gsplat.Update()
        # write rendered result to file
        self.writeToFile()

    def ecc_callback(self,obj,event):
        print("Ecc callback!")
        global gsplat
        # increment the gaussian radius value by the currently set resolution
        gsplat.SetEccentricity(gsplat.GetEccentricity()+self.ecc_res)
        # force gaussian splat object to update
        gsplat.Update()
        # write rendered result to file
        self.writeToFile()

## animation objects
print("Setting up animation objects...")
# window->image
imageFilter = vtk.vtkWindowToImageFilter()
imageFilter.SetInput(renwin)
imageFilter.SetInputBufferTypeToRGB()
imageFilter.ReadFrontBufferOff()
print("Setting up movie writer")
## movie writer
movieWriter = vtk.vtkAVIWriter()
# set the window to read into the writer
movieWriter.SetInputConnection(imageFilter.GetOutputPort())

print("Setting up timer for movie writer")
# initialize renderer so the timer callback can be added
iren.Initialize()
# create instance of class that handles callback passing references
# to imagefilter and movie writer
animcall = animation_callback(imageFilter,movieWriter)

############ ITERATE INDEX ############
iren.AddObserver("TimerEvent",animcall.index_callback)
# create timer to generate timer events
tid = iren.CreateRepeatingTimer(100)
# pass id into callback manager so it can destroy the timer when it's finished
animcall.setTimerID(tid)
animcall.setTimerCountLimit(150000-starter_f)
#animcall.setTimerCountLimit(10)
# Create file for volume data
# will be closed at end
print("Opening data file")
movieWriter.SetFileName("gaussiansplat-temperaturerotate-index.avi") # set filenanme
movieWriter.SetRate(1600)# set frame rate
################################################

########## ROTATE ANIMATION ############
### assign observer manager timer event
##iren.AddObserver("TimerEvent",animcall.rotation_callback)
### set repeating timer to generate TimerEvents
### get id and give it to animation callback manager
##tid = iren.CreateRepeatingTimer(1)
##animcall.setTimerID(tid)
### hide the sliders
###slrep.SetVisibility(0)
###slrotrep.SetVisibility(0)
###slfiltrep.SetVisibility(0)
### set filenanme
##movieWriter.SetFileName("gaussiansplat-temperaturerotate-rotation.avi") 
##animcall.rot_res = 360.0/1000.0
##animcall.setTimerCountLimit(1000.0)
##movieWriter.SetRate(100)# set frame rate
##############################################

############ FILTER ANIMATION ############
### assign observer manager timer event
##iren.AddObserver("TimerEvent",animcall.filter_callback)
##filt=0.0
### set repeating timer to generate TimerEvents
### get id and give it to animation callback manager
##tid = iren.CreateRepeatingTimer(1)
##animcall.setTimerID(tid)
### hide the sliders
###slrep.SetVisibility(0)
###slrotrep.SetVisibility(0)
###slfiltrep.SetVisibility(0)
##movieWriter.SetFileName("gaussiansplat-temperaturerotate-filter.avi")
##animcall.filt_res = 1/60.0
##animcall.setTimerCountLimit(60)
##movieWriter.SetRate(10)# set frame rate
################################################

############ CLIPPING ANIMATION ############
### assign observer manager timer event
##iren.AddObserver("TimerEvent",animcall.clip_callback)
### reset clip value
##clip=0.0
### reset slider
##slcliprep.SetValue(clip)
##slcliprep.Modified()
### set repeating timer to generate TimerEvents
### get id and give it to animation callback manager
##tid = iren.CreateRepeatingTimer(100)
##animcall.setTimerID(tid)
### hide the sliders
###slrep.SetVisibility(0)
###slrotrep.SetVisibility(0)
###slfiltrep.SetVisibility(0)
##movieWriter.SetFileName("gaussiansplat-temperaturerotate-clip.avi")
### set resolution of clipping
##animcall.clip_res = 1/60.0
##animcall.setTimerCountLimit(60)
##movieWriter.SetRate(10)# set frame rate
################################################

############## RADIUS ANIMATION ############
### assign observer manager timer event
##iren.AddObserver("TimerEvent",animcall.radius_callback)
### set repeating timer to generate TimerEvents
### get id and give it to animation callback manager
##tid = iren.CreateRepeatingTimer(1)
##animcall.setTimerID(tid)
### hide the sliders
###slrep.SetVisibility(0)
###slrotrep.SetVisibility(0)
###slfiltrep.SetVisibility(0)
##movieWriter.SetFileName("gaussiansplat-temperaturerotate-radius.avi")
### set resolution of clipping
##gsplat.SetRadius(0.01)
##gsplat.Update()
##animcall.rad_res = 0.01
##animcall.setTimerCountLimit(50)
##movieWriter.SetRate(10)# set frame rate
##################################################

############## ECCENTRICITY ANIMATION ############
### assign observer manager timer event
##iren.AddObserver("TimerEvent",animcall.ecc_callback)
### set repeating timer to generate TimerEvents
### get id and give it to animation callback manager
##tid = iren.CreateRepeatingTimer(1)
##animcall.setTimerID(tid)
### hide the sliders
###slrep.SetVisibility(0)
###slrotrep.SetVisibility(0)
###slfiltrep.SetVisibility(0)
##movieWriter.SetFileName("gaussiansplat-temperaturerotate-eccentricity.avi")
### set resolution of clipping
##gsplat.SetEccentricity(1.0)
##gsplat.Update()
##animcall.setTimerCountLimit(50)
##movieWriter.SetRate(10)# set frame rate
##################################################

# set bk color
print("Setting background color")
ren.SetBackground(0.0,0.0,0.0)
renwin.SetSize(500, 500)
renwin.Render()

cam1 = ren.GetActiveCamera()
cam1.Zoom(2.0)
# rotate piece by 180 degrees so the laser part is at the top and the
# heating area is at the bottom
cam1.Roll(180.0)

print("Starting recording")
movieWriter.Start()
# start process
iren.Start()
movieWriter.End()
print("Ending recording")
