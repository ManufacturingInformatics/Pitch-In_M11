import vtk
import glob
import os
import matplotlib.cm
import h5py
import numpy as np

# get list of filenames
files = glob.glob(r"D:\BEAM\RPiWaifu\Pngs"+r"\*.png")

# select subset
ff = []
for i in range(0,len(files),1):
    ff.append(files[i])

print("Getting dataset limits")
path = "D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20.hdf5"
#path = "D:\BEAM\RPiWaifu\pi-camera-data-127001-2019-10-14T12-41-20-waifu-noise-scale-91398-150000.hdf5"
if "waifu" not in path:
    with h5py.File(path,'r') as file:
        mmax,mmin = np.nanmax(file['pi-camera-1'][()],axis=(0,1,2)),np.nanmin(file['pi-camera-1'][()],axis=(0,1,2))
else:
    with h5py.File(path,'r') as file:
        mmax,mmin = np.nanmax(file['pi-camera-1-scale'][()],axis=(0,1,2)),np.nanmin(file['pi-camera-1-scale'][()],axis=(0,1,2))



def norm2image(T,vmax=mmax,vmin=mmin):
    return int((T-vmin)*((2**16)/(vmax-vmin)))

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
fedge.GenerateValues(5,bb[0],bb[1])
fedge.ComputeNormalsOn()
fedge.ComputeGradientsOn()
fedge.Update()

print("Setting up Stripper and extractor")
# create triangle strips from isosurface
# faster rendering
stripper = vtk.vtkStripper()
stripper.SetInputConnection(fedge.GetOutputPort())
# take the data and generate poly data
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(stripper.GetOutputPort())
mapper.ScalarVisibilityOff()
# pass connection onto actor to produce the object
actor = vtk.vtkActor()
actor.SetMapper(mapper)
# setup color
#actor.GetProperty().SetColor(0.0,1.0,0.0)
actor.GetProperty().SetSpecular(.3)
actor.GetProperty().SetSpecularPower(20)
actor.GetProperty().SetOpacity(.5)

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
# add actor to renderer
ren.AddActor(actor)
# initialize window interactor
iren.Initialize()

print("Setting up animation")
ren.ResetCamera()
# get current position
pos = ren.GetActiveCamera().GetPosition()
# get bounds of the object
bounds = fedge.GetOutput().GetBounds()
# set initial camera position to be end of stack
ren.GetActiveCamera().SetPosition(pos[0],pos[1],bounds[5])
ren.Modified()
renwin.Render()
# update known target position
pos = ren.GetActiveCamera().GetPosition()
## set resolution based on desired time assuming timer of 1 ms
# target time
des_t_s = 60
# velocity based on estimated travel distance, d/s
v = (bounds[4]- pos[2])/des_t_s
# distance moved in one time jump given target velocity
move_res = v*0.001
print("Target resolution: ",move_res)

imageFilter = vtk.vtkWindowToImageFilter()
imageFilter.SetInput(renwin)
imageFilter.SetInputBufferTypeToRGB()
imageFilter.ReadFrontBufferOff()

movieWriter = vtk.vtkAVIWriter()
movieWriter.SetInputConnection(imageFilter.GetOutputPort())
# make camera
tid = iren.CreateRepeatingTimer(1)
# function for moving the camera along the z-axis
def moveAlongZ(obj,event):
    #print("Moving time event")
    global pos
    global imageFilter
    imageFilter.Modified()
    imageFilter.Update()
    global movieWriter
    movieWriter.Write()
    global ren
    # get first renderer
    cc = obj.GetRenderWindow().GetRenderers().GetFirstRenderer()
    # get camera
    cam = cc.GetActiveCamera()
    # get current camera position
    pos = cam.GetPosition()
    #print("Progress: {}".format(100-(pos[2]-bounds[4])/(bounds[5]-bounds[4])*100))
    # update camera position
    global move_res
    cam.SetPosition(pos[0],pos[1],pos[2]+move_res)
    ren.Modified()
    ww = imageFilter.GetInput()
    ww.Render()
    imageFilter.Modified()
    imageFilter.Update()
    # get new position
    pos = ren.GetActiveCamera().GetPosition()
    # check if we've reached the end
    print(pos[2])
    if pos[2]<bounds[4]:
        print("Destroying timer")
        iren.DestroyTimer(tid)
        movieWriter.End()

def recordActions(obj,event):
    global imageFilter
    imageFilter.Modified()
    imageFilter.Update()
    global movieWriter
    movieWriter.Write()

## move camera backwards through the dataset centre of the stack
# set timer event handler
#movieWriter.SetFileName("imagestack-flyingedges-travel-z-reverse.avi")
#movieWriter.SetRate(1000)
#iren.AddObserver("TimerEvent",moveAlongZ)

# record actions for an indeterminate amount of time
#movieWriter.SetFileName("imagestack-flyingedges-travel-z-custom.avi")
#movieWriter.SetRate(60)
#iren.AddObserver("TimerEvent",recordActions)

# event handler to print camera possition every time it's modified
#iren.AddObserver("ModifiedEvent",lambda obj,event : print(ren.GetActiveCamera().GetPosition()))
print("Starting...")
# generate window
renwin.Render()
renwin.SetWindowName("Image Stack -> Flying Edges")
movieWriter.Start()
# start interactor
iren.Start()
movieWriter.End()


