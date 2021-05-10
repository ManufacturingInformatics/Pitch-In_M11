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
from scipy.interpolate import Rbf

starter_f = 93000
ff = starter_f
filt = 0.6
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

def getDataInCT(frame,ct):
    mask = np.zeros(frame.shape,np.uint8)
    mask = cv2.drawContours(mask,[ct],0,1,-1)
    return frame*mask

# convert numpy float array -> vtkFloatArray
#vk_data = numpy_support.numpy_to_vtk(num_array=ct_data_filt.ravel(), deep=True, array_type=vtk.VTK_FLOAT)

# dataset that will be updated by function readPoints
pointSource = vtk.vtkProgrammableSource()
def readPoints():
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
    img_ct = frame[bb[1]:bb[1]+bb[3],bb[0]:bb[0]+bb[2]]
    # get the location of the data points
    xx,yy = np.where(img_ct>20.0)
    # get the respective values
    zz = img_ct[xx,yy]
    # have the height touch 0
    zz -= zz.min()
    # generate and add mirroring data
    xx = np.concatenate((xx,xx))
    yy = np.concatenate((yy,yy))
    zz = np.concatenate((zz,-zz))
    # update points set
    for x,y,z in zip(xx,yy,zz):
        points.InsertNextPoint(float(z),float(y),float(x))
##
def readPointsRotRbf():
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

    lim = zz.min() + (zz.max()-zz.min())*filt
    for x,y,z in zip(xxr[dd>=lim],yyr[dd>=lim],zzr[dd>=lim]):
        points.InsertNextPoint(float(z),float(y),float(x))

######
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
    lim = np.diff(dd).max()*0.5
    print(lim)
    print(xxr[dd>lim].shape,yyr[dd>lim].shape,zzr[dd>lim].shape)
    print(dd.min(),dd.max())
    # background values are a base value plus some noise
    # to ignore tha background, we ignore values less than 1
    # this should reveal the values that stand out from the bk
    for x,y,z in zip(xxr[dd>lim],yyr[dd>lim],zzr[dd>lim]):
        points.InsertNextPoint(float(z),float(y),float(x))  

def readPointsRotEdge():
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

    edge = []
    # for each of the unique column values
    for c in set(yy):
        # get the highest row value, i.e. location of edge value
        edge.append(xx[yy==c].max()+bb[3]//2)

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
    for r in np.arange(0.0,360.0,10.0):
        # generate rotation object
        rot = R.from_euler('xyz',[0.0,r,0.0],degrees=True)
        # apply to matrix
        vv = rot.apply(data)
        # udpate position matrix
        xxr = np.concatenate((xxr,vv[:,0]))
        yyr = np.concatenate((yyr,vv[:,1]))
        zzr = np.concatenate((zzr,vv[:,2]))
        dd = np.concatenate((dd,zz))
        
    for x,y,z in zip(xxr,yyr,zzr):
        points.InsertNextPoint(float(z),float(y),float(x))

def readPointsRot():
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

    for x,y,z in zip(xxr,yyr,zzr):
        points.InsertNextPoint(float(z),float(y),float(x))

def errorHandler(obj,event):
    global pointSource
    global ff
    print("Error callback! Skipping frame ",ff)
    # in the event of an error e.g. no points to read
    # increment frame value
    ff +=1
    # collect new frame
    pointSource.Modified()

pointSource.SetExecuteMethod(readPointsRotRbf)
## convert vtkFloatArray to vtkPointData
##image = vtk.vtkImageData()
##image.SetDimensions(ct_data_filt.shape)
##points = image.GetPointData()
##points.SetScalars(vk_data)

# create surface reconstruction filter
surf = vtk.vtkSurfaceReconstructionFilter()
 #provide input to the surface
surf.SetInputConnection(pointSource.GetOutputPort())
surf.AddObserver("ErrorEvent",errorHandler)

# create contour filter
cf = vtk.vtkContourFilter()
cf.AddObserver("ErrorEvent",errorHandler)
cf.SetInputConnection(surf.GetOutputPort())
cf.SetValue(0,0.0)

# sometimes artifacts are introduced into the surface that intefere with the surrounding elements
# they can be removed
reverse = vtk.vtkReverseSense()
reverse.AddObserver("ErrorEvent",errorHandler)
reverse.SetInputConnection(cf.GetOutputPort())
reverse.ReverseCellsOn()
reverse.ReverseNormalsOn()

## setup plotting 
mm = vtk.vtkPolyDataMapper()
mm.SetInputConnection(cf.GetOutputPort())
mm.ScalarVisibilityOff()

surfaceActor = vtk.vtkActor()
surfaceActor.SetMapper(mm)
# set color properties
surfaceActor.GetProperty().SetDiffuseColor(1.0000, 0.3882, 0.2784)
surfaceActor.GetProperty().SetSpecularColor(1, 1, 1)
surfaceActor.GetProperty().SetSpecular(.4)
surfaceActor.GetProperty().SetSpecularPower(50)

# Create the RenderWindow, Renderer and both Actors
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

# Add the actors to the renderer, set the background and size
ren.AddActor(surfaceActor)
ren.SetBackground(1, 1, 1)
renWin.SetSize(400, 400)
ren.GetActiveCamera().SetFocalPoint(0, 0, 0)
ren.GetActiveCamera().SetPosition(1, 0, 0)
ren.GetActiveCamera().SetViewUp(0, 0, 1)
ren.ResetCamera()
ren.GetActiveCamera().Azimuth(20)
ren.GetActiveCamera().Elevation(30)
ren.GetActiveCamera().Dolly(1.2)
ren.ResetCameraClippingRange()

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
        global ff
        ff += 1
        print(ff)
        pointSource.Modified()
        # get the renderers in the window
        cc = obj.GetRenderWindow().GetRenderers()
        # get the first one
        rr = cc.GetFirstRenderer()
        # get camera of renderer
        rr.ResetCamera()
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
        filt+=self.filt_res
        pointSource.Modified()
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

## animation objects
print("Setting up animation objects...")
# window->image
imageFilter = vtk.vtkWindowToImageFilter()
imageFilter.SetInput(renWin)
imageFilter.SetInputBufferTypeToRGB()
imageFilter.ReadFrontBufferOff()
print("Setting up movie writer")
## movie writer
movieWriter = vtk.vtkAVIWriter()
# set the window to read into the writer
movieWriter.SetInputConnection(imageFilter.GetOutputPort())

iren.Initialize()

animcall = animation_callback(imageFilter,movieWriter)

############## ITERATE INDEX ############
##iren.AddObserver("TimerEvent",animcall.index_callback)
### create timer to generate timer events
##tid = iren.CreateRepeatingTimer(1)
### pass id into callback manager so it can destroy the timer when it's finished
##animcall.setTimerID(tid)
##animcall.setTimerCountLimit(150000-starter_f)
###animcall.setTimerCountLimit(10)
### Create file for volume data
### will be closed at end
##print("Opening data file")
##movieWriter.SetFileName("surfacerecon-temperaturerotate-index.avi") # set filenanme
##movieWriter.SetRate(118)# set frame rate
##################################################

############ ITERATE ROTATION ############
iren.AddObserver("TimerEvent",animcall.rotation_callback)
# create timer to generate timer events
tid = iren.CreateRepeatingTimer(1)
# pass id into callback manager so it can destroy the timer when it's finished
animcall.setTimerID(tid)
animcall.rot_res = 0.1
animcall.setTimerCountLimit(360/animcall.rot_res)
#animcall.setTimerCountLimit(10)
# Create file for volume data
# will be closed at end
print("Opening data file")
movieWriter.SetFileName("surfacerecon-temperaturerotate-rotation.avi") # set filenanme
movieWriter.SetRate(int(3600/60))# set frame rate
################################################

renWin.Render()

movieWriter.Start()
iren.Start()
movieWriter.End()
