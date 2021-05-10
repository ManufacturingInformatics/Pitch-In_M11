import vtk
import glob
import os
import matplotlib.cm
import h5py
import numpy as np

## setting up the renderer and window
print("Setting up renderer and window")
ren = vtk.vtkRenderer()
# set background color
ren.SetBackground(0.0,0.0,0.0)
renwin = vtk.vtkRenderWindow()
renwin.AddRenderer(ren)
renwin.SetSize(500,500)
iren = vtk.vtkRenderWindowInteractor()
# define interactor
iren.SetRenderWindow(renwin)

# hide warnings window
vtk.vtkOutputWindow.GetInstance().SetGlobalWarningDisplay(0)

# temperature reference value for marching cubes
temp_ref = 15.0 # C

# get list of filenames
files = glob.glob(r"D:\BEAM\RPiWaifu\Pngs"+r"\*.png")
# select subset
ff = []
# remove known bad files
bad_files = [11348]
for i in range(0,len(files),1):
    if i not in bad_files:
        ff.append(files[i])

print("Getting dataset limits")
path = "D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20.hdf5"
with h5py.File(path,'r') as file:
    mmax,mmin = np.nanmax(file['pi-camera-1'][()],axis=(0,1,2)),np.nanmin(file['pi-camera-1'][()],axis=(0,1,2))

def norm2image(T,vmax=mmax,vmin=mmin):
    return int((T-vmin)*((2**16)/(vmax-vmin)))

num_frames = len(ff)
print("Number of frames: ",len(ff))
# setup vtk file names list
print("Sorting out VTK file list")
filePath = vtk.vtkStringArray()
files.sort(key=lambda x : int(os.path.splitext(os.path.basename(x))[0].split('f')[1]))
filePath.SetNumberOfValues(len(ff))
for i in range(0,len(ff),1):
    filePath.SetValue(i,ff[i])

# setup vtk png reader
print("Setting up reader")
reader = vtk.vtkPNGReader()
#reader.AddObserver("ProgressEvent",lambda obj,event : print("{:.2f}".format(obj.GetProgress()),end='\r',sep=''))
reader.SetFileNames(filePath)
# set how the images are stacked next to each other
reader.SetDataSpacing(1,1,.1)
# calling update gets the reader to read in the images
reader.Update()

print("Setting up flying edges")
fedge = vtk.vtkFlyingEdges3D()
fedge.SetInputConnection(reader.GetOutputPort())
bb = reader.GetOutput().GetScalarRange()
# generate 5 contours across the range
fedge.GenerateValues(5,bb[0],bb[1])
fedge.ComputeNormalsOn()
fedge.ComputeGradientsOn()
fedge.Update()

print("Setting up Stripper and extractor")
# create triangle strips from isosurface
# faster rendering
stripper = vtk.vtkStripper()
stripper.SetInputConnection(fedge.GetOutputPort())
# update so we can get bounds
stripper.Update()

print("Setting up cropper")
## setup cropping area
# generate data for cropping region
cubeSource = vtk.vtkCubeSource()
bounds = stripper.GetOutput().GetBounds()
def click2Index(clickz,bb=bounds,ff=num_frames):
    # click is the world position coordinates where the click occured
    # bounds is the limit of the shape
    # the click z position is normalized to the image index range
    # [zmin, zmax] |-> [num_frames, 0]
    return int((clickz-bb[-2])*((-ff)/(bb[-1]-bb[-2]))+ff)

def index2Click(index,bb=bounds,ff=num_frames):
    return float((index-ff)*((bb[-1]-bb[-2])/(-ff))+bb[-2])

cubeSource.SetBounds(bounds[0],bounds[1],bounds[2],bounds[3],bounds[5],bounds[5]-0.1)
cubeSource.Update()
## convert poly data into implicit function
box = vtk.vtkBox()
box.SetBounds(cubeSource.GetOutput().GetBounds())
box.Modified()

# pass implicit function to clipping object
clip = vtk.vtkClipPolyData()
clip.SetClipFunction(box)
# setup input
clip.SetInputConnection(stripper.GetOutputPort())
clip.GenerateClippedOutputOn()
clip.GenerateClipScalarsOn()
clip.Update()

print("Setting up clipping mapper")
# remove temporary connection to marching cubes
mapper = vtk.vtkPolyDataMapper()
# attach mapper to clip result
mapper.SetInputConnection(clip.GetClippedOutputPort())
# take the data and generate poly data
mapper.ScalarVisibilityOff()

print("Setting up clipping actor")
# pass connection onto actor to produce the object
actor = vtk.vtkActor()
actor.SetMapper(mapper)
# setup color
#actor.GetProperty().SetColor(0.0,1.0,0.0)
actor.GetProperty().SetSpecular(.3)
actor.GetProperty().SetSpecularPower(20)
# opacity is set to one as clipping shows a single frame which is a poly data outline
actor.GetProperty().SetOpacity(1.0)
# add actors to renderer
ren.AddActor(actor)

print("Setting up wireframe for cropping area")
# setup mapper to draw cube
cubeMapper = vtk.vtkPolyDataMapper()
cubeMapper.SetInputConnection(cubeSource.GetOutputPort())
cubeMapper.Update()
# setup actor to render cube
cubeActor = vtk.vtkActor()
cubeActor.SetMapper(cubeMapper)
# set actor to show cube as wireframe
cubeActor.GetProperty().SetRepresentationToWireframe()
cubeActor.GetProperty().SetOpacity(1.0)
cubeActor.GetProperty().SetColor(1.0,0.0,0.0) # set color of wireframe
# add actor to display wireframe
ren.AddActor(cubeActor)

## create filter slider to adjust cropping
print("Setting up clipping slider")
slcliprep = vtk.vtkSliderRepresentation2D()
slcliprep.GetPoint1Coordinate().SetCoordinateSystemToDisplay()
slcliprep.GetPoint1Coordinate().SetValue(300,40)
slcliprep.GetPoint2Coordinate().SetCoordinateSystemToDisplay()
slcliprep.GetPoint2Coordinate().SetValue(470,40)
slcliprep.SetMinimumValue(0.0) # limits of slider
slcliprep.SetMaximumValue(1.0)
slcliprep.SetValue(0.0)
slcliprep.SetLabelFormat("%0.2f") # format the index is displayed in
slcliprep.SetTitleText("Cropping") # slider title
slcliprep.GetTitleProperty().SetColor(1,1,1) # set text to white
slcliprep.GetSliderProperty().SetColor(0,1,0) # Set nob to red
slcliprep.GetLabelProperty().SetColor(1,1,1) # set color of value label text
slcliprep.GetSelectedProperty().SetColor(0,0,1) # set color when selected
slcliprep.GetTubeProperty().SetColor(1,1,1) # set the bit that moves to white
slcliprep.GetCapProperty().SetColor(1,1,1) # set the end points to white

slclip = vtk.vtkSliderWidget()
slclip.SetRepresentation(slcliprep)
slclip.SetInteractor(iren)
slclip.SetAnimationModeToJump()
slclip.EnabledOn()

def set_clip(obj,event):
    print("Clip callback!")
    ## update the wireframe cube centre to show change
    # get the centre of the cube
    # convert to list so it can be edited
    cc = list(cubeSource.GetCenter())
    # update centre according to slider value
    # slider dictates how far along the z axis the cropping goes as a percentage
    # cube centre is z start + set percentage of range
    cc[2] = bounds[5]-(obj.GetRepresentation().GetValue()*(bounds[5]-bounds[4]))+0.05
    # assign modified centre
    cubeSource.SetCenter(cc)
    cubeSource.Update()
    ## update actual clipping
    # calculate new bounds for clipping
    newz = bounds[5]-(obj.GetRepresentation().GetValue()*(bounds[5]-bounds[4]))
    # update box bounds
    box.SetBounds(bounds[0],bounds[1],bounds[2],bounds[3],newz,newz+0.1)
    # update clip
    clip.Update()
    #
    renwin.Render()
    renwin.Modified()
    # reset the camera position to show contents
    ren.ResetCamera()
    
slclip.AddObserver("EndInteractionEvent",set_clip)
#slclip.InvokeEvent("EndInteractionEvent")

# reset camera to show clipped data
ren.ResetCamera()

## setting up mass properties
mp = vtk.vtkMassProperties()
# set connection to clipped output of dataset
clip.Update()
mp.SetInputConnection(clip.GetClippedOutputPort())
def printUpdatedMassP(obj,event):
    print(obj.GetSurfaceArea())
mp.AddObserver("ModifiedEvent",printUpdatedMassP)
mp.Update()
mp.Modified()

## setting up timer to animate and generate surface area data
# before starting reset slider
slclip.GetRepresentation().SetValue(0.0)
iren.Initialize()
# create timer to generate events
tid = iren.CreateRepeatingTimer(1)
# ieration resolution
res = (bounds[5]-bounds[4])/len(ff)
print("Iterative res: {}".format(res))
# open file to store surface area values
surface_file = open("waifu2x-flyingedge-surfacearea-ref{}.csv".format(int(temp_ref)),'w')
# frame counter
ii = 0

print("Setting up recording objects")
imageFilter = vtk.vtkWindowToImageFilter()
imageFilter.SetInput(renwin)
imageFilter.SetInputBufferTypeToRGB()
imageFilter.ReadFrontBufferOff()

movieWriter = vtk.vtkAVIWriter()
movieWriter.SetInputConnection(imageFilter.GetOutputPort())
movieWriter.SetFileName("waifu2x-flyingedge-surfacearea-ref{}.avi".format(int(temp_ref)))
movieWriter.SetRate(100)

# function to write values to the file
def writeSurfaceArea(obj,event):
    print("Animation callback!")
    global ii
    print(ii)
    cc = list(cubeSource.GetCenter())
    # calculate centre of the clipping cube by working back from index
    cc[2] = index2Click(ii)
    # assign modified centre
    cubeSource.SetCenter(cc)
    cubeSource.Update()
    ## update actual clipping
    # update box bounds
    box.SetBounds(bounds[0],bounds[1],bounds[2],bounds[3],cc[2]-0.05,cc[2]+0.05)
    # update clip
    clip.Update()
    # update mass properties
    mp.Update()
    # update display
    renwin.Render()
    #renwin.Modified()
    surface_file.write("{}\n".format(mp.GetSurfaceArea()))
    # reset the camera position to show contents
    ren.ResetCamera()
    # save video to file
    global imageFilter
    global movieWriter
    imageFilter.Modified()
    imageFilter.Update()
    movieWriter.Write()

    if ii<len(ff):
        ii+=1
    else:
        print("Reached limit!")
        print("Killing timer!")
        global tid
        iren.DestroyTimer(tid)
        if not surface_file.closed:
            print("Closing file")
            surface_file.close()
            
print("Setting up animation callback!")
# add observer to respond to call
iren.AddObserver("TimerEvent",writeSurfaceArea)

# initialize window interactor
iren.Initialize()
print("Starting...")
renwin.SetWindowName("Waifu2x Image Slices Cropper")
# generate window
renwin.Render()
# start movie writer
movieWriter.Start()
# start interactor
iren.Start()
# if the data file is still open for some reason
# ensure the file is closed
if not surface_file.closed:
    print("Emergency surface file close!")
    surface_file.close()
movieWriter.End


